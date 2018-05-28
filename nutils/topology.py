# Copyright (c) 2014 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
The topology module defines the topology objects, notably the
:class:`StructuredTopology` and :class:`UnstructuredTopology`. Maintaining
strict separation of topological and geometrical information, the topology
represents a set of elements and their interconnectivity, boundaries,
refinements, subtopologies etc, but not their positioning in physical space. The
dimension of the topology represents the dimension of its elements, not that of
the the space they are embedded in.

The primary role of topologies is to form a domain for :mod:`nutils.function`
objects, like the geometry function and function bases for analysis, as well as
provide tools for their construction. It also offers methods for integration and
sampling, thus providing a high level interface to operations otherwise written
out in element loops. For lower level operations topologies can be used as
:mod:`nutils.element` iterators.
"""

from . import element, function, util, numpy, parallel, log, config, numeric, cache, transform, warnings, matrix, types, sample, points, _
import functools, collections.abc, itertools, functools, operator, numbers, pathlib

_identity = lambda x: x

class TransformsTuple(types.Singleton):

  __slots__ = '_transforms', '_sorted', '_indices', '_fromdims'

  @types.apply_annotations
  def __init__(self, transforms:types.tuple[transform.canonical], fromdims:types.strictint):
    assert all(trans[-1].fromdims == fromdims for trans in transforms)
    self._transforms = transforms
    self._sorted = numpy.empty([len(self._transforms)], dtype=object)
    for i, trans in enumerate(self._transforms):
      self._sorted[i] = tuple(map(id, trans))
    self._indices = numpy.argsort(self._sorted)
    self._sorted = self._sorted[self._indices]
    self._fromdims = fromdims
    super().__init__()

  def __iter__(self):
    return iter(self._transforms)

  def __getitem__(self, item):
    return self._transforms[item]

  def __len__(self):
    return len(self._transforms)

  def index(self, trans):
    index, tail = self.index_with_tail(trans)
    if tail:
      raise ValueError
    return index

  def index_with_tail(self, trans):
    head, tail = transform.promote(trans, self._fromdims)
    headid_array = numpy.empty((), dtype=object)
    headid_array[()] = headid = tuple(map(id, head))
    i = numpy.searchsorted(self._sorted, headid_array, side='right') - 1
    if i < 0:
      raise ValueError
    match = self._sorted[i]
    if headid[:len(match)] != match:
      raise ValueError
    return self._indices[i], head[len(match):] + tail

  def contains(self, trans):
    try:
      self.index(trans)
    except ValueError:
      return False
    else:
      return True

  __contains__ = contains

  def contains_with_tail(self, trans):
    try:
      self.index_with_tail(trans)
    except ValueError:
      return False
    else:
      return True

class Topology(types.Singleton):
  'topology base class'

  __slots__ = 'ndims',
  __cache__ = 'border_transforms', 'simplex', 'boundary', 'interfaces'

  # subclass needs to implement: .references, .transforms, .opposites

  @types.apply_annotations
  def __init__(self, ndims:types.strictint):
    super().__init__()
    assert ndims >= 0
    self.ndims = ndims

  def __str__(self):
    'string representation'

    return '{}(#{})'.format(self.__class__.__name__, len(self))

  def __len__(self):
    return len(self.references)

  def __iter__(self):
    return iter(map(element.Element, self.references, self.transforms, self.opposites))

  def getitem(self, item):
    return EmptyTopology(self.ndims)

  def __getitem__(self, item):
    if not isinstance(item, tuple):
      item = item,
    if all(it in (...,slice(None)) for it in item):
      return self
    topo = self.getitem(item) if len(item) != 1 or not isinstance(item[0],str) \
       else functools.reduce(operator.or_, map(self.getitem, item[0].split(',')), EmptyTopology(self.ndims))
    if not topo:
      raise KeyError(item)
    return topo

  def __invert__(self):
    return OppositeTopology(self)

  def __or__(self, other):
    assert isinstance(other, Topology) and other.ndims == self.ndims
    return other if not self \
      else self if not other \
      else NotImplemented if isinstance(other, UnionTopology) \
      else UnionTopology((self,other))

  __ror__ = lambda self, other: self.__or__(other)

  def __and__(self, other):
    # Strategy: loop over combined elements sorted by .transform while keeping
    # track of the origin (mine=True for self, mine=False for other), and
    # select an element if it is equal to or a refinement of the previous
    # (hold) element and it originates from the other topology (mine == need).
    # Hold is not updated in case of a match because it might match multiple
    # children.
    references = []
    transforms = []
    opposites = []
    need = None
    for ref, trans, opp, mine in sorted([(ref, trans, opp, True) for ref, trans, opp in zip(self.references, self.transforms, self.opposites)] + [(ref, trans, opp, False) for ref, trans, opp in zip(other.references, other.transforms, other.opposites)], key=lambda v: v[1]):
      if mine == need and trans[:len(holdtrans)] == holdtrans:
        assert opp[:len(holdopp)] == holdopp
        references.append(ref)
        transforms.append(trans)
        opposites.append(opp)
      else:
        holdtrans = trans
        holdopp = opp
        need = not mine
    return UnstructuredTopology(self.ndims, references, transforms, opposites)

  __rand__ = lambda self, other: self.__and__(other)

  def __add__(self, other):
    return self | other

  def __contains__(self, element):
    try:
      ielem = self.transforms.index(element.transform)
    except ValueError:
      return False
    return self.references[ielem] == element.reference and self.opposites[ielem] == element.opposite

  def __sub__(self, other):
    assert isinstance(other, Topology) and other.ndims == self.ndims
    return other.__rsub__(self)

  def __rsub__(self, other):
    assert isinstance(other, Topology) and other.ndims == self.ndims
    return other - other.subset(self, newboundary=getattr(self,'boundary',None))

  def __mul__(self, other):
    return ProductTopology(self, other)

  @property
  def border_transforms(self):
    indices = set()
    for btrans in self.boundary.transforms:
      try:
        ielem, tail = self.transforms.index_with_tail(btrans)
      except ValueError:
        pass
      else:
        indices.add(ielem)
    return TransformsTuple(tuple(self.transforms[ielem] for ielem in sorted(indices)), self.ndims)

  @property
  def refine_iter(self):
    topo = self
    for irefine in log.count('refinement level'):
      yield topo
      topo = topo.refined

  def basis(self, name, *args, **kwargs):
    if self.ndims == 0:
      return function.asarray([1])
    split = name.split('-', 1)
    if len(split) == 2 and split[0] in ('h', 'th'):
      name = split[1] # default to non-hierarchical bases
      if split[0] == 'th':
        kwargs.pop('truncation_tolerance', None)
    f = getattr(self, 'basis_' + name)
    return f(*args, **kwargs)

  def sample(self, ischeme, degree):
    'Create sample.'

    points = tuple(ref.getpoints(ischeme, degree) for ref in self.references)
    offset = numpy.cumsum([0] + [p.npoints for p in points])
    return sample.Sample((self.transforms, self.opposites), points, map(numpy.arange, offset[:-1], offset[1:]))

  @util.single_or_multiple
  def elem_eval(self, funcs, ischeme, separate=False, geometry=None, asfunction=False, *, edit=None, title='elem_eval', **kwargs):
    'element-wise evaluation'

    if geometry is not None:
      warnings.deprecation('elem_eval will be removed in future, use integrate_elementwise instead')
      return self.integrate_elementwise(funcs, ischeme=ischeme, geometry=geometry, asfunction=asfunction, edit=edit, **kwargs)
    if edit is not None:
      funcs = [edit(func) for func in funcs]
    warnings.deprecation('elem_eval will be removed in future, use sample(...).eval instead')
    sample = self.sample(*element.parse_legacy_ischeme(ischeme))
    retvals = sample.eval(funcs, title=title, **kwargs)
    return [sample.asfunction(retval) for retval in retvals] if asfunction \
      else [[retval[index] for index in sample.index] for retval in retvals] if separate \
      else retvals

  @util.single_or_multiple
  def integrate_elementwise(self, funcs, *, asfunction=False, **kwargs):
    'element-wise integration'

    ielem = function.FindTransform(self.transforms, function.TRANS).index
    with matrix.backend('numpy'):
      retvals = self.integrate([function.Inflate(function.asarray(func)[_], dofmap=ielem[_], length=len(self), axis=0) for func in funcs], **kwargs)
    retvals = [retval.export('dense') if len(retval.shape) == 2 else retval for retval in retvals]
    return [function.elemwise(self.transforms, retval, shape=retval.shape[1:]) for retval in retvals] if asfunction \
      else retvals

  @util.single_or_multiple
  def elem_mean(self, funcs, geometry=None, ischeme='gauss', degree=None, **kwargs):
    ischeme, degree = element.parse_legacy_ischeme(ischeme if degree is None else ischeme + str(degree))
    area, *integrals = self.integrate_elementwise((1,)+funcs, ischeme=ischeme, degree=degree, geometry=geometry, **kwargs)
    return [integral / area[(slice(None),)+(_,)*(integral.ndim-1)] for integral in integrals]

  @util.single_or_multiple
  def integrate(self, funcs, ischeme='gauss', degree=None, geometry=None, edit=None, *, arguments=None, title='integrate'):
    'integrate functions'

    ischeme, degree = element.parse_legacy_ischeme(ischeme if degree is None else ischeme + str(degree))
    if geometry is not None:
      funcs = [func * function.J(geometry, self.ndims) for func in funcs]
    if edit is not None:
      funcs = [edit(func) for func in funcs]
    return self.sample(ischeme, degree).integrate(funcs, arguments=arguments, title=title)

  def integral(self, func, ischeme='gauss', degree=None, geometry=None, edit=None):
    'integral'

    ischeme, degree = element.parse_legacy_ischeme(ischeme if degree is None else ischeme + str(degree))
    if geometry is not None:
      func = func * function.J(geometry, self.ndims)
    if edit is not None:
      funcs = edit(func)
    return self.sample(ischeme, degree).integral(func)

  def projection(self, fun, onto, geometry, **kwargs):
    'project and return as function'

    weights = self.project(fun, onto, geometry, **kwargs)
    return onto.dot(weights)

  @log.title
  def project(self, fun, onto, geometry, ischeme='gauss', degree=None, droptol=1e-12, exact_boundaries=False, constrain=None, verify=None, ptype='lsqr', edit=None, *, arguments=None, **solverargs):
    'L2 projection of function onto function space'

    log.debug('projection type:', ptype)

    if degree is not None:
      ischeme += str(degree)
    if constrain is None:
      constrain = util.NanVec(onto.shape[0])
    else:
      constrain = constrain.copy()
    if exact_boundaries:
      constrain |= self.boundary.project(fun, onto, geometry, constrain=constrain, title='boundaries', ischeme=ischeme, droptol=droptol, ptype=ptype, edit=edit, arguments=arguments)
    assert isinstance(constrain, util.NanVec)
    assert constrain.shape == onto.shape[:1]

    avg_error = None # setting this depends on projection type

    if ptype == 'lsqr':
      assert ischeme is not None, 'please specify an integration scheme for lsqr-projection'
      fun2 = function.asarray(fun)**2
      if len(onto.shape) == 1:
        Afun = function.outer(onto)
        bfun = onto * fun
      elif len(onto.shape) == 2:
        Afun = function.outer(onto).sum(2)
        bfun = function.sum(onto * fun, -1)
        if fun2.ndim:
          fun2 = fun2.sum(-1)
      else:
        raise Exception
      assert fun2.ndim == 0
      A, b, f2, area = self.integrate([Afun,bfun,fun2,1], geometry=geometry, ischeme=ischeme, edit=edit, arguments=arguments, title='building system')
      N = A.rowsupp(droptol)
      if numpy.equal(b, 0).all():
        constrain[~constrain.where&N] = 0
        avg_error = 0.
      else:
        solvecons = constrain.copy()
        solvecons[~(constrain.where|N)] = 0
        u = A.solve(b, constrain=solvecons, **solverargs)
        constrain[N] = u[N]
        err2 = f2 - numpy.dot(2*b-A.matvec(u), u) # can be negative ~zero due to rounding errors
        avg_error = numpy.sqrt(err2) / area if err2 > 0 else 0

    elif ptype == 'convolute':
      assert ischeme is not None, 'please specify an integration scheme for convolute-projection'
      if len(onto.shape) == 1:
        ufun = onto * fun
        afun = onto
      elif len(onto.shape) == 2:
        ufun = function.sum(onto * fun, axis=-1)
        afun = function.norm2(onto)
      else:
        raise Exception
      u, scale = self.integrate([ufun, afun], geometry=geometry, ischeme=ischeme, edit=edit, arguments=arguments)
      N = ~constrain.where & (scale > droptol)
      constrain[N] = u[N] / scale[N]

    elif ptype == 'nodal':

      ## data = function.Tuple([fun, onto])
      ## F = W = 0
      ## for elem in self:
      ##   f, w = data(elem, 'bezier2')
      ##   W += w.sum(axis=-1).sum(axis=0)
      ##   F += numeric.contract(f[:,_,:], w, axis=[0,2])
      ## I = (W!=0)

      F = numpy.zeros(onto.shape[0])
      W = numpy.zeros(onto.shape[0])
      I = numpy.zeros(onto.shape[0], dtype=bool)
      fun = function.zero_argument_derivatives(function.asarray(fun))
      data = function.Tuple(function.Tuple([fun, onto_f.simplified, function.Tuple(onto_ind)]) for onto_ind, onto_f in function.blocks(function.zero_argument_derivatives(onto)))
      for ref, trans, opp in zip(self.references, self.transforms, self.opposites):
        ipoints, iweights = ref.getischeme('bezier2')
        for fun_, onto_f_, onto_ind_ in data.eval(_transforms=(trans, opp), _points=ipoints, **arguments or {}):
          onto_f_ = onto_f_.swapaxes(0,1) # -> dof axis, point axis, ...
          indfun_ = fun_[(slice(None),)+numpy.ix_(*onto_ind_[1:])]
          assert onto_f_.shape[0] == len(onto_ind_[0])
          assert onto_f_.shape[1:] == indfun_.shape
          W[onto_ind_[0]] += onto_f_.reshape(onto_f_.shape[0],-1).sum(1)
          F[onto_ind_[0]] += (onto_f_ * indfun_).reshape(onto_f_.shape[0],-1).sum(1)
          I[onto_ind_[0]] = True

      I[constrain.where] = False
      constrain[I] = F[I] / W[I]

    else:
      raise Exception('invalid projection {!r}'.format(ptype))

    numcons = constrain.where.sum()
    info = 'constrained {}/{} dofs'.format(numcons, constrain.size)
    if avg_error is not None:
      info += ', error {:.2e}/area'.format(avg_error)
    log.info(info)
    if verify is not None:
      assert numcons == verify, 'number of constraints does not meet expectation: {} != {}'.format(numcons, verify)

    return constrain

  @property
  def simplex(self):
    references = []
    transforms = []
    opposites = []
    for ref, trans, opp in zip(self.references, self.transforms, self.opposites):
      for simplextrans, simplexref in ref.simplices:
        references.append(simplexref)
        transforms.append(trans+(simplextrans,))
        opposites.append(opp+(simplextrans,))
    return UnstructuredTopology(self.ndims, references, transforms, opposites)

  def refined_by(self, refine):
    'create refined space by refining dofs in existing one'

    refine = TransformsTuple(tuple(item.transform if isinstance(item,element.Element) else item for item in refine), self.ndims)
    refined = []
    for ref, trans in zip(self.references, self.transforms):
      if trans in refine:
        refined.extend(trans+(ctrans,) for ctrans, cref in ref.children if cref)
      else:
        refined.append(trans)
    return self.hierarchical(refined)

  def hierarchical(self, refined):
    return HierarchicalTopology(self, refined)

  @property
  def refined(self):
    return RefinedTopology(self)

  def refine(self, n):
    'refine entire topology n times'

    if numpy.iterable(n):
      assert len(n) == self.ndims
      assert all(ni == n[0] for ni in n)
      n = n[0]
    return self if n <= 0 else self.refined.refine(n-1)

  @log.title
  def trim(self, levelset, maxrefine, ndivisions=8, name='trimmed', leveltopo=None, *, arguments=None):
    'trim element along levelset'

    if arguments is None:
      arguments = {}

    fcache = cache.WrapperCache()
    levelset = function.zero_argument_derivatives(levelset).simplified
    if leveltopo is None:
      ischeme = 'vertex{}'.format(maxrefine)
      refs = [ref.trim(levelset.eval(_transforms=(trans, opp), _points=ref.getischeme(ischeme)[0], _cache=fcache, **arguments), maxrefine=maxrefine, ndivisions=ndivisions) for ref, trans, opp in log.zip('elem', self.references, self.transforms, self.opposites)]
    else:
      log.info('collecting leveltopo elements')
      bins = [[] for ielem in range(len(self))]
      for trans in leveltopo.transforms:
        ielem, tail = self.transforms.index_with_tail(trans)
        bins[ielem].append(tail)
      refs = []
      for ref, trans, ctransforms in log.zip('elem', self.references, self.transforms, bins):
        levels = numpy.empty(ref.nvertices_by_level(maxrefine))
        cover = list(fcache[ref.vertex_cover](tuple(sorted(ctransforms)), maxrefine))
        # confirm cover and greedily optimize order
        mask = numpy.ones(len(levels), dtype=bool)
        while mask.any():
          imax = numpy.argmax([mask[indices].sum() for ctrans, points, indices in cover])
          ctrans, points, indices = cover.pop(imax)
          levels[indices] = levelset.eval(_transforms=(trans + ctrans,), _points=points, _cache=fcache, **arguments)
          mask[indices] = False
        refs.append(ref.trim(levels, maxrefine=maxrefine, ndivisions=ndivisions))
    log.debug('cache', fcache.stats)
    return SubsetTopology(self, refs, newboundary=name)

  def subset(self, topo, newboundary=None, strict=False):
    'intersection'
    refs = [ref.empty for ref in self.references]
    for ref, trans in zip(topo.references, topo.transforms):
      try:
        ielem = self.transforms.index(trans)
      except ValueError:
        assert not strict, 'elements do not form a strict subset'
      else:
        subref = self.references[ielem] & ref
        if strict:
          assert subref == ref, 'elements do not form a strict subset'
        refs[ielem] = subref
    if not any(refs):
      return EmptyTopology(self.ndims)
    return SubsetTopology(self, refs, newboundary)

  def withgroups(self, vgroups={}, bgroups={}, igroups={}, pgroups={}):
    return WithGroupsTopology(self, vgroups, bgroups, igroups, pgroups) if vgroups or bgroups or igroups or pgroups else self

  withsubdomain  = lambda self, **kwargs: self.withgroups(vgroups=kwargs)
  withboundary   = lambda self, **kwargs: self.withgroups(bgroups=kwargs)
  withinterfaces = lambda self, **kwargs: self.withgroups(igroups=kwargs)
  withpoints     = lambda self, **kwargs: self.withgroups(pgroups=kwargs)

  @log.title
  @util.single_or_multiple
  def elem_project(self, funcs, degree, ischeme=None, check_exact=False, *, arguments=None):

    if arguments is None:
      arguments = {}

    if ischeme is None:
      ischeme = 'gauss{}'.format(degree*2)

    blocks = function.Tuple([function.Tuple([function.Tuple((function.Tuple(ind), f.simplified))
      for ind, f in function.blocks(function.zero_argument_derivatives(func))])
        for func in funcs])

    bases = {}
    extractions = [[] for ifunc in range(len(funcs))]

    for ref, trans, opp in log.zip('elem', self.references, self.transforms, self.opposites):

      try:
        points, projector, basis = bases[ref]
      except KeyError:
        points, weights = ref.getischeme(ischeme)
        coeffs = ref.get_poly_coeffs('bernstein', degree=degree)
        basis = numeric.poly_eval(coeffs[_], points)
        npoints, nfuncs = basis.shape
        A = numeric.dot(weights, basis[:,:,_] * basis[:,_,:])
        projector = numpy.linalg.solve(A, basis.T * weights)
        bases[ref] = points, projector, basis

      for ifunc, ind_val in enumerate(blocks.eval(_transforms=(trans, opp), _points=points, **arguments)):

        if len(ind_val) == 1:
          (allind, sumval), = ind_val
        else:
          allind, where = zip(*[numpy.unique([i for ind, val in ind_val for i in ind[iax]], return_inverse=True) for iax in range(funcs[ifunc].ndim)])
          sumval = numpy.zeros([len(n) for n in (points,) + allind])
          for ind, val in ind_val:
            I, where = zip(*[(w[:len(n)], w[len(n):]) for w, n in zip(where, ind)])
            numpy.add.at(sumval, numpy.ix_(range(len(points)), *I), val)
          assert not any(where)

        ex = numeric.dot(projector, sumval)
        if check_exact:
          numpy.testing.assert_almost_equal(sumval, numeric.dot(basis, ex), decimal=15)

        extractions[ifunc].append((allind, ex))

    return extractions

  @log.title
  def volume(self, geometry, ischeme='gauss', degree=1, *, arguments=None):
    return self.integrate(1, geometry=geometry, ischeme=ischeme, degree=degree, arguments=arguments)

  @log.title
  def check_boundary(self, geometry, elemwise=False, ischeme='gauss', degree=1, tol=1e-15, print=print, *, arguments=None):
    if elemwise:
      for ref in self.references:
        ref.check_edges(tol=tol, print=print)
    volume = self.volume(geometry, ischeme=ischeme, degree=degree, arguments=arguments)
    zeros, volumes = self.boundary.integrate([geometry.normal(), geometry * geometry.normal()], geometry=geometry, ischeme=ischeme, degree=degree, arguments=arguments)
    if numpy.greater(abs(zeros), tol).any():
      print('divergence check failed: {} != 0'.format(zeros))
    if numpy.greater(abs(volumes - volume), tol).any():
      print('divergence check failed: {} != {}'.format(volumes, volume))

  def volume_check(self, geometry, ischeme='gauss', degree=1, decimal=15, *, arguments=None):
    warnings.deprecation('volume_check will be removed in future, us check_boundary instead')
    self.check_boundary(geometry=geometry, ischeme=ischeme, degree=degree, tol=10**-decimal, arguments=arguments)

  def indicator(self, subtopo):
    if isinstance(subtopo, str):
      subtopo = self[subtopo]
    values = types.frozenarray([int(trans in subtopo.transforms) for trans in self.transforms])
    assert len(subtopo) == values.sum(0), '{} is not a proper subtopology of {}'.format(subtopo, self)
    return function.Get(values, axis=0, item=function.FindTransform(self.transforms, function.Promote(self.ndims, trans=function.TRANS)).index)

  def select(self, indicator, ischeme='bezier2', *, arguments=None):
    values = self.elem_eval(indicator, ischeme, separate=True, arguments=arguments)
    selected = tuple(i for i, value in enumerate(values) if numpy.greater(value, 0).any())
    return UnstructuredTopology(self.ndims, (self.references[i] for i in selected), (self.transforms[i] for i in selected), (self.opposites[i] for i in selected))

  def prune_basis(self, basis):
    used = numpy.zeros(len(basis), dtype=bool)
    for axes, func in function.blocks(basis):
      dofmap = axes[0]
      for trans in self.transforms:
        dofs = dofmap.eval(_transforms=(trans,))
        used[dofs] = True
    return function.mask(basis, used)

  def locate(self, geom, coords, ischeme='vertex', scale=1, tol=1e-12, eps=0, maxiter=100, *, arguments=None):
    '''Create a sample based on physical coordinates.

    In a finite element application, functions are commonly evaluated in points
    that are defined on the topology. The reverse, finding a point on the
    topology based on a function value, is often a nonlinear process and as
    such involves Newton iterations. The ``locate`` function facilitates this
    search process and produces a :class:`nutils.sample.Sample` instance that
    can be used for the subsequent evaluation of any function in the given
    physical points.

    Example:

    >>> from . import mesh
    >>> domain, geom = mesh.rectilinear([2,1])
    >>> sample = domain.locate(geom, [[1.5, .5]])
    >>> sample.eval(geom).tolist()
    [[1.5, 0.5]]

    Locate has a long list of arguments that can be used to steer the nonlinear
    search process, but the default values should be fine for reasonably
    standard situations.

    Args
    ----
    geom : 1-dimensional :class:`nutils.function.Array`
        Geometry function of length ``ndims``.
    coords : 2-dimensional :class:`float` array
        Array of coordinates with ``ndims`` columns.
    ischeme : :class:`str` (default: "vertex")
        Sample points used to determine bounding boxes.
    scale : :class:`float` (default: 1)
        Bounding box amplification factor, useful when element shapes are
        distorted. Setting this to >1 can increase computational effort but is
        otherwise harmless.
    tol : :class:`float` (default: 1e-12)
        Newton tolerance.
    eps : :class:`float` (default: 0)
        Epsilon radius around element within which a point is considered to be
        inside.
    maxiter : :class:`int` (default: 100)
        Maximum allowed number of Newton iterations.
    arguments : :class:`dict` (default: None)
        Arguments for function evaluation.

    Returns
    -------
    located : :class:`nutils.sample.Sample`
    '''

    nprocs = min(config.nprocs, len(self))
    if arguments is None:
      arguments = {}
    if geom.ndim == 0:
      geom = geom[_]
      coords = coords[...,_]
    assert geom.shape == (self.ndims,)
    coords = numpy.asarray(coords, dtype=float)
    assert coords.ndim == 2 and coords.shape[1] == self.ndims
    vertices = self.elem_eval(geom, ischeme=ischeme, separate=True, arguments=arguments)
    bboxes = numpy.array([numpy.mean(v,axis=0) * (1-scale) + numpy.array([numpy.min(v,axis=0), numpy.max(v,axis=0)]) * scale
      for v in vertices]) # nelems x {min,max} x ndims
    vref = element.getsimplex(0)
    ielems = parallel.shempty(len(coords), dtype=int)
    xis = parallel.shempty((len(coords),len(geom)), dtype=float)
    for ipoint, coord in parallel.pariter(log.enumerate('point', coords), nprocs=nprocs):
      ielemcandidates, = numpy.logical_and(numpy.greater_equal(coord, bboxes[:,0,:]), numpy.less_equal(coord, bboxes[:,1,:])).all(axis=-1).nonzero()
      for ielem in sorted(ielemcandidates, key=lambda i: numpy.linalg.norm(bboxes[i].mean(0)-coord)):
        converged = False
        ref = self.references[ielem]
        xi, w = ref.getischeme('gauss1')
        xi = (numpy.dot(w,xi) / w.sum())[_] if len(xi) > 1 else xi.copy()
        J = function.localgradient(geom, self.ndims)
        geom_J = function.Tuple((function.zero_argument_derivatives(geom), function.zero_argument_derivatives(J))).simplified
        for iiter in range(maxiter):
          coord_xi, J_xi = geom_J.eval(_transforms=(self.transforms[ielem], self.opposites[ielem]), _points=xi, **arguments)
          err = numpy.linalg.norm(coord - coord_xi)
          if err < tol:
            converged = True
            break
          if iiter and err > prev_err:
            break
          prev_err = err
          xi += numpy.linalg.solve(J_xi, coord - coord_xi)
        if converged and ref.inside(xi[0], eps=eps):
          ielems[ipoint] = ielem
          xis[ipoint], = xi
          break
      else:
        raise LocateError('failed to locate point: {}'.format(coord))
    transforms = []
    opposites = []
    points_ = []
    index = []
    for ielem in numpy.unique(ielems):
      w, = numpy.equal(ielems, ielem).nonzero()
      transforms.append(self.transforms[ielem])
      opposites.append(self.opposites[ielem])
      points_.append(points.CoordsPoints(xis[w]))
      index.append(w)
    return sample.Sample((tuple(transforms), tuple(opposites)), points_, index)

  def supp(self, basis, mask=None):
    if mask is None:
      mask = numpy.ones(len(basis), dtype=bool)
    elif isinstance(mask, list) or numeric.isarray(mask) and mask.dtype == int:
      tmp = numpy.zeros(len(basis), dtype=bool)
      tmp[mask] = True
      mask = tmp
    else:
      assert numeric.isarray(mask) and mask.dtype == bool and mask.shape == basis.shape[:1]
    indfunc = function.Tuple([ind[0] for ind, f in function.blocks(basis)])
    subset = []
    for ielem, trans in enumerate(self.transforms):
      try:
        ind, = numpy.concatenate(indfunc.eval(_transforms=(trans,)), axis=1)
      except function.EvaluationError:
        pass
      else:
        if mask[ind].any():
          subset.append(ielem)
    if not subset:
      return EmptyTopology(self.ndims)
    return self.subset(UnstructuredTopology(self.ndims, (self.references[i] for i in subset), (self.transforms[i] for i in subset), (self.opposites[i] for i in subset)), newboundary='supp', strict=True)

  def revolved(self, geom):
    assert geom.ndim == 1
    revdomain = self * RevolutionTopology()
    angle, = function.rootcoords(1)
    geom, angle = function.bifurcate(geom, angle)
    revgeom = function.concatenate([geom[0] * function.trignormal(angle), geom[1:]])
    simplify = function.replace(lambda op: function.zeros(()) if op is angle else None)
    return revdomain, revgeom, simplify

  def extruded(self, geom, nelems, periodic=False, bnames=('front','back')):
    assert geom.ndim == 1
    root = transform.Identifier(self.ndims+1, 'extrude')
    extopo = self * StructuredLine(root, i=0, j=nelems, periodic=periodic, bnames=bnames)
    exgeom = function.concatenate(function.bifurcate(geom, function.rootcoords(1)))
    return extopo, exgeom

  @property
  @log.title
  def boundary(self):
    references = []
    transforms = []
    for ielem, (ioppelems, elemref, elemtrans) in enumerate(zip(self.connectivity, self.references, self.transforms)):
      for (edgetrans, edgeref), ioppelem in zip(elemref.edges, ioppelems):
        if edgeref:
          if ioppelem == -1:
            references.append(edgeref)
            transforms.append(elemtrans+(edgetrans,))
          else:
            ioppedge = self.connectivity[ioppelem].index(ielem)
            ref = edgeref - self.references[ioppelem].edge_refs[ioppedge]
            if ref:
              references.append(ref)
              transforms.append(elemtrans+(edgetrans,))
    return UnstructuredTopology(self.ndims-1, references, transforms, transforms)

  @property
  @log.title
  def interfaces(self):
    references = []
    transforms = []
    opposites = []
    for ielem, (ioppelems, elemref, elemtrans) in enumerate(zip(self.connectivity, self.references, self.transforms)):
      for (edgetrans, edgeref), ioppelem in zip(elemref.edges, ioppelems):
        if edgeref and -1 < ioppelem < ielem:
          ioppedge = self.connectivity[ioppelem].index(ielem)
          oppedgetrans, oppedgeref = self.references[ioppelem].edges[ioppedge]
          ref = oppedgeref and edgeref & oppedgeref
          if ref:
            references.append(ref)
            transforms.append(elemtrans+(edgetrans,))
            opposites.append(self.transforms[ioppelem]+(oppedgetrans,))
    return UnstructuredTopology(self.ndims-1, references, transforms, opposites)

  def basis_spline(self, degree):
    assert degree == 1
    return self.basis('std', degree)

  def basis_discont(self, degree):
    'discontinuous shape functions'

    assert numeric.isint(degree) and degree >= 0
    coeffs = []
    nmap = []
    ndofs = 0
    for ref in self.references:
      elemcoeffs = ref.get_poly_coeffs('bernstein', degree=degree)
      coeffs.append(elemcoeffs)
      nmap.append(types.frozenarray(ndofs + numpy.arange(len(elemcoeffs)), copy=False))
      ndofs += len(elemcoeffs)
    degrees = set(n-1 for c in coeffs for n in c.shape[1:])
    return function.polyfunc(coeffs, nmap, ndofs, self.transforms)

  def _basis_c0_structured(self, name, degree):
    'C^0-continuous shape functions with lagrange stucture'

    assert numeric.isint(degree) and degree >= 0

    if degree == 0:
      raise ValueError('Cannot build a C^0-continuous basis of degree 0.  Use basis \'discont\' instead.')

    coeffs = [ref.get_poly_coeffs(name, degree=degree) for ref in self.references]
    offsets = numpy.cumsum([0] + [len(c) for c in coeffs])
    dofmap = numpy.repeat(-1, offsets[-1])
    for ielem, ioppelems in enumerate(self.connectivity):
      for iedge, jelem in enumerate(ioppelems): # loop over element neighbors and merge dofs
        if jelem < ielem:
          continue # either there is no neighbor along iedge or situation will be inspected from the other side
        jedge = self.connectivity[jelem].index(ielem)
        idofs = offsets[ielem] + self.references[ielem].get_edge_dofs(degree, iedge)
        jdofs = offsets[jelem] + self.references[jelem].get_edge_dofs(degree, jedge)
        for idof, jdof in zip(idofs, jdofs):
          while dofmap[idof] != -1:
            idof = dofmap[idof]
          while dofmap[jdof] != -1:
            jdof = dofmap[jdof]
          if idof != jdof:
            dofmap[max(idof, jdof)] = min(idof, jdof) # create left-looking pointer
    # assign dof numbers left-to-right
    ndofs = 0
    for i, n in enumerate(dofmap):
      if n == -1:
        dofmap[i] = ndofs
        ndofs += 1
      else:
        dofmap[i] = dofmap[n]

    elem_slices = map(slice, offsets[:-1], offsets[1:])
    dofs = tuple(types.frozenarray(dofmap[s]) for s in elem_slices)
    return function.polyfunc(coeffs, dofs, ndofs, self.transforms)

  def basis_lagrange(self, degree):
    'lagrange shape functions'
    return self._basis_c0_structured('lagrange', degree)

  def basis_bernstein(self, degree):
    'bernstein shape functions'
    return self._basis_c0_structured('bernstein', degree)

  basis_std = basis_bernstein

stricttopology = types.strict[Topology]

class LocateError(Exception):
  pass

class WithGroupsTopology(Topology):
  'item topology'

  __slots__ = 'basetopo', 'vgroups', 'bgroups', 'igroups', 'pgroups'
  __cache__ = 'refined',

  @types.apply_annotations
  def __init__(self, basetopo:stricttopology, vgroups:types.frozendict={}, bgroups:types.frozendict={}, igroups:types.frozendict={}, pgroups:types.frozendict={}):
    assert vgroups or bgroups or igroups or pgroups
    self.basetopo = basetopo
    self.vgroups = vgroups
    self.bgroups = bgroups
    self.igroups = igroups
    self.pgroups = pgroups
    super().__init__(basetopo.ndims)
    assert all(topo is Ellipsis or isinstance(topo, str) or isinstance(topo, Topology) and topo.ndims == basetopo.ndims and set(self.basetopo.transforms).issuperset(topo.transforms) for topo in self.vgroups.values())

  def withgroups(self, vgroups={}, bgroups={}, igroups={}, pgroups={}):
    args = []
    for groups, newgroups in (self.vgroups,vgroups), (self.bgroups,bgroups), (self.igroups,igroups), (self.pgroups,pgroups):
      groups = groups.copy()
      groups.update(newgroups)
      args.append(groups)
    return WithGroupsTopology(self.basetopo, *args)

  def __len__(self):
    return len(self.basetopo)

  def getitem(self, item):
    if not isinstance(item, str):
      return self.basetopo.getitem(item)
    try:
      itemtopo = self.vgroups[item]
    except KeyError:
      return self.basetopo.getitem(item)
    else:
      return itemtopo if isinstance(itemtopo, Topology) else self.basetopo[itemtopo]

  @property
  def border_transforms(self):
    return self.basetopo.border_transforms

  @property
  def connectivity(self):
    return self.basetopo.connectivity

  @property
  def references(self):
    return self.basetopo.references

  @property
  def transforms(self):
    return self.basetopo.transforms

  @property
  def opposites(self):
    return self.basetopo.opposites

  @property
  def boundary(self):
    return self.basetopo.boundary.withgroups(self.bgroups)

  @property
  def interfaces(self):
    baseitopo = self.basetopo.interfaces
    # last minute orientation fix
    igroups = {}
    for name, itopo in self.igroups.items():
      references = []
      transforms = []
      opposites = []
      for ref, trans, opp in zip(itopo.references, itopo.transforms, itopo.opposites):
        references.append(ref)
        if trans in baseitopo.transforms:
          transforms.append(trans)
          opposites.append(opp)
        else:
          transforms.append(opp)
          opposites.append(trans)
      igroups[name] = UnstructuredTopology(self.ndims-1, references, transforms, opposites)
    return baseitopo.withgroups(igroups)

  @property
  def points(self):
    return UnstructuredTopology(0, itertools.chain.from_iterable(ptopo.references for ptopo in self.pgroups.values()), itertools.chain.from_iterable(ptopo.transforms for ptopo in self.pgroups.values()), itertools.chain.from_iterable(ptopo.opposites for ptopo in self.pgroups.values())).withgroups(self.pgroups)

  def basis(self, name, *args, **kwargs):
    return self.basetopo.basis(name, *args, **kwargs)

  @property
  def refined(self):
    groups = [{name: topo.refined if isinstance(topo,Topology) else topo for name, topo in groups.items()} for groups in (self.vgroups,self.bgroups,self.igroups,self.pgroups)]
    return self.basetopo.refined.withgroups(*groups)

class OppositeTopology(Topology):
  'opposite topology'

  __slots__ = 'basetopo',

  def __init__(self, basetopo):
    self.basetopo = basetopo
    super().__init__(basetopo.ndims)

  def getitem(self, item):
    return ~(self.basetopo.getitem(item))

  def __len__(self):
    return len(self.basetopo)

  @property
  def references(self):
    return self.basetopo.references

  @property
  def transforms(self):
    return self.basetopo.opposites

  @property
  def opposites(self):
    return self.basetopo.transforms

  def __invert__(self):
    return self.basetopo

class EmptyTopology(Topology):
  'empty topology'

  __slots__ = ()

  def __len__(self):
    return 0

  def __or__(self, other):
    assert self.ndims == other.ndims
    return other

  def __rsub__(self, other):
    return other

  @property
  def references(self):
    return ()

  @property
  def transforms(self):
    return TransformsTuple((), self.ndims)

  opposites = transforms

class Point(Topology):
  'point'

  __slots__ = 'references', 'transforms', 'opposites'

  @types.apply_annotations
  def __init__(self, trans:transform.stricttransform, opposite:transform.stricttransform=None):
    assert trans[-1].fromdims == 0
    self.references = element.getsimplex(0),
    self.transforms = TransformsTuple((trans,), 0)
    self.opposites = self.transforms if opposite is None else TransformsTuple((opposite,), 0)
    super().__init__(ndims=0)

class StructuredLine(Topology):
  'structured topology'

  __slots__ = 'root', 'i', 'j', 'periodic', 'bnames'
  __cache__ = 'references', 'transforms', 'opposites', 'boundary', 'interfaces'

  @types.apply_annotations
  def __init__(self, root:transform.stricttransformitem, i:types.strictint, j:types.strictint, periodic:bool=False, bnames:types.tuple[types.strictstr]=None):
    'constructor'

    assert j > i
    assert not bnames or len(bnames) == 2
    self.root = root
    self.i = i
    self.j = j
    self.periodic = periodic
    self.bnames = bnames or ()
    super().__init__(ndims=1)

  @property
  def references(self):
    return (element.getsimplex(1),)*(self.j-self.i)

  @property
  def transforms(self):
    return TransformsTuple(tuple((self.root, transform.Shift([float(offset)])) for offset in range(self.i, self.j)), 1)

  opposites = transforms

  def __len__(self):
    return self.j - self.i

  @property
  def boundary(self):
    if self.periodic:
      return EmptyTopology(ndims=0)
    outsideleft = self.root, transform.Shift([float(self.i-1)])
    outsideright = self.root, transform.Shift([float(self.j)])
    right, left = element.LineReference().edge_transforms
    bnd = Point(self.transforms[0] + (left,), outsideleft + (right,)), Point(self.transforms[-1] + (right,), outsideright + (left,))
    return UnionTopology(bnd, self.bnames)

  @property
  def interfaces(self):
    right, left = element.LineReference().edge_transforms
    return UnionTopology(tuple(Point(b+(left,), a+(right,)) for a, b in util.pairwise(self.transforms, periodic=self.periodic)))

  @classmethod
  def _bernstein_poly(cls, degree):
    'bernstein polynomial coefficients'


  @classmethod
  def _spline_coeffs(cls, p, n):
    'spline polynomial coefficients'

    assert p >= 0, 'invalid polynomial degree {}'.format(p)
    if p == 0:
      assert n == -1
      return numpy.array([[[1.]]])

    assert 1 <= n < 2*p
    extractions = numpy.empty((n, p+1, p+1))
    extractions[0] = numpy.eye(p+1)
    for i in range(1, n):
      extractions[i] = numpy.eye(p+1)
      for j in range(2, p+1):
        for k in reversed(range(j, p+1)):
          alpha = 1. / min(2+k-j, n-i+1)
          extractions[i-1,:,k] = alpha * extractions[i-1,:,k] + (1-alpha) * extractions[i-1,:,k-1]
        extractions[i,-j-1:-1,-j-1] = extractions[i-1,-j:,-1]

    # magic bernstein triangle
    poly = numpy.zeros([p+1,p+1], dtype=int)
    for k in range(p//2+1):
      poly[k,k] = root = (-1)**p if k == 0 else (poly[k-1,k] * (k*2-1-p)) / k
      for i in range(k+1,p+1-k):
        poly[i,k] = poly[k,i] = root = (root * (k+i-p-1)) / i
    poly = poly[::-1].astype(float)

    return types.frozenarray(numeric.contract(extractions[:,_,:,:], poly[_,:,_,:], axis=-1).transpose(0,2,1), copy=False)

  def basis_spline(self, degree, periodic=None, removedofs=None):
    'spline from vertices'

    if periodic is None:
      periodic = self.periodic

    if numpy.iterable(degree):
      degree, = degree

    if numpy.iterable(removedofs):
      removedofs, = removedofs

    strides = 1, 1
    shape = len(self), degree+1
    ndofs = sum(s*(n-1) for s, n in zip(strides, shape))+1
    dofs = numpy.arange(ndofs)
    if periodic and degree > 0:
      assert ndofs >= 2 * degree
      dofs[-degree:] = dofs[:degree]
      ndofs -= degree
    dofs = numpy.lib.stride_tricks.as_strided(dofs, shape=shape, strides=tuple(s*dofs.strides[0] for s in strides))
    dofs = types.frozenarray(dofs, copy=False)

    p = degree
    n = 2*p-1
    nelems = len(self)
    if periodic:
      if nelems == 1: # periodicity on one element can only mean a constant
        coeffs = list(self._spline_coeffs(0, n))
        dofs = types.frozenarray([[0]], copy=False)
      else:
        coeffs = list(self._spline_coeffs(p, n)[p-1:p]) * nelems
    else:
      coeffs = list(self._spline_coeffs(p, min(nelems,n)))
      if len(coeffs) < nelems:
        coeffs = coeffs[:p-1] + coeffs[p-1:p] * (nelems-2*(p-1)) + coeffs[p:]
    coeffs = types.frozenarray(coeffs, copy=False)

    func = function.polyfunc(coeffs, dofs, ndofs, self.transforms)
    if not removedofs:
      return func

    mask = numpy.ones(ndofs, dtype=bool)
    mask[list(removedofs)] = False
    return function.mask(func, mask)

  def basis_discont(self, degree):
    'discontinuous shape functions'

    ref = element.LineReference()
    coeffs = [ref.get_poly_coeffs('bernstein', degree=degree)]*len(self)
    ndofs = ref.get_ndofs(degree)
    dofs = types.frozenarray(numpy.arange(ndofs*len(self), dtype=int).reshape(len(self), ndofs), copy=False)
    return function.polyfunc(coeffs, dofs, ndofs*len(self), self.transforms)

  def basis_std(self, degree, periodic=None, removedofs=None):
    'spline from vertices'

    if periodic is None:
      periodic = self.periodic

    strides = max(1, degree), 1
    shape = len(self), degree+1
    ndofs = sum(s*(n-1) for s, n in zip(strides, shape))+1
    dofs = numpy.arange(ndofs)
    if periodic and degree > 0:
      dofs[-1] = dofs[0]
      ndofs -= 1
    dofs = numpy.lib.stride_tricks.as_strided(dofs, shape=shape, strides=tuple(s*dofs.strides[0] for s in strides))
    dofs = types.frozenarray(dofs, copy=False)

    coeffs = [element.LineReference().get_poly_coeffs('bernstein', degree=degree)]*len(self)
    func = function.polyfunc(coeffs, dofs, ndofs, self.transforms)
    if not removedofs:
      return func

    mask = numpy.ones(ndofs, dtype=bool)
    mask[list(removedofs)] = False
    return function.mask(func, mask)

  def __str__(self):
    'string representation'

    return '{}({}:{})'.format(self.__class__.__name__, self.i, self.j)

class Axis(types.Singleton):
  __slots__ = ()

class DimAxis(Axis):
  __slots__ = 'i', 'j', 'isperiodic'
  isdim = True
  @types.apply_annotations
  def __init__(self, i:types.strictint, j:types.strictint, isperiodic:bool):
    super().__init__()
    self.i = i
    self.j = j
    self.isperiodic = isperiodic

class BndAxis(Axis):
  __slots__ = 'i', 'j', 'ibound', 'side'
  isdim = False
  @types.apply_annotations
  def __init__(self, i:types.strictint, j:types.strictint, ibound:types.strictint, side:bool):
    super().__init__()
    self.i = i
    self.j = j
    self.ibound = ibound
    self.side = side

class StructuredTopology(Topology):
  'structured topology'

  __slots__ = 'root', 'axes', 'nrefine', 'shape', '_bnames'
  __cache__ = 'references', 'transforms', 'opposites', 'connectivity', 'boundary', 'interfaces'

  @types.apply_annotations
  def __init__(self, root:transform.stricttransformitem, axes:types.tuple[types.strict[Axis]], nrefine:types.strictint=0, bnames:types.tuple[types.strictstr]=None):
    'constructor'

    self.root = root
    self.axes = axes
    self.nrefine = nrefine
    self.shape = tuple(axis.j - axis.i for axis in self.axes if axis.isdim)
    if bnames is None:
      assert len(self.axes) <= 3
      bnames = ('left', 'right'), ('bottom', 'top'), ('front', 'back')
      bnames = itertools.chain.from_iterable(n for axis, n in zip(self.axes, bnames) if axis.isdim and not axis.isperiodic)
    self._bnames = tuple(bnames)
    assert len(self._bnames) == sum(2 for axis in self.axes if axis.isdim and not axis.isperiodic)
    assert all(isinstance(bname,str) for bname in self._bnames)
    super().__init__(len(self.shape))

  def __len__(self):
    return numpy.prod(self.shape, dtype=int)

  @property
  def references(self):
    return (util.product(element.getsimplex(1 if axis.isdim else 0) for axis in self.axes),)*len(self)

  def getitem(self, item):
    if not isinstance(item, tuple):
      return EmptyTopology(self.ndims)
    assert all(isinstance(it,slice) for it in item) and len(item) <= self.ndims
    if all(it == slice(None) for it in item): # shortcut
      return self
    axes = []
    idim = 0
    for axis in self.axes:
      if axis.isdim and idim < len(item):
        s = item[idim]
        start, stop, stride = s.indices(axis.j - axis.i)
        assert stride == 1
        assert stop > start
        if start > 0 or stop < axis.j - axis.i:
          axis = DimAxis(axis.i+start, axis.i+stop, isperiodic=False)
        idim += 1
      axes.append(axis)
    return StructuredTopology(self.root, axes, self.nrefine, bnames=self._bnames)

  @property
  def periodic(self):
    dimaxes = (axis for axis in self.axes if axis.isdim)
    return tuple(idim for idim, axis in enumerate(dimaxes) if axis.isdim and axis.isperiodic)

  @staticmethod
  def mktransforms(axes, root, nrefine):
    assert nrefine >= 0

    updim = []
    rmdims = numpy.zeros(len(axes), dtype=bool)
    for order, side, idim in sorted((axis.ibound, axis.side, idim) for idim, axis in enumerate(axes) if not axis.isdim):
      ref = util.product(element.getsimplex(0 if rmdim else 1) for rmdim in rmdims)
      iedge = (idim - rmdims[:idim].sum()) * 2 + 1 - side
      updim.append(ref.edge_transforms[iedge])
      rmdims[idim] = True

    grid = [numpy.arange(axis.i>>nrefine, ((axis.j-1)>>nrefine)+1) if axis.isdim else numpy.array([(axis.i-1 if axis.side else axis.j)>>nrefine]) for axis in axes]
    indices = numeric.broadcast(*numeric.ix(grid))
    transforms = numeric.asobjvector([transform.Shift(numpy.array(index, dtype=float))] for index in log.iter('elem', indices, indices.size)).reshape(indices.shape)

    if nrefine:
      scales = numeric.asobjvector([trans] for trans in (element.LineReference()**len(axes)).child_transforms).reshape((2,)*len(axes))
      for irefine in log.range('level', nrefine-1, -1, -1):
        offsets = numpy.array([r[0] for r in grid])
        grid = [numpy.arange(axis.i>>irefine,((axis.j-1)>>irefine)+1) if axis.isdim else numpy.array([(axis.i-1 if axis.side else axis.j)>>irefine]) for axis in axes]
        A = transforms[numpy.broadcast_arrays(*numeric.ix(r//2-o for r, o in zip(grid, offsets)))]
        B = scales[numpy.broadcast_arrays(*numeric.ix(r%2 for r in grid))]
        transforms = A + B

    shape = tuple(axis.j - axis.i for axis in axes if axis.isdim)
    return TransformsTuple(tuple(transform.canonical([root] + trans + updim) for trans in log.iter('canonical', transforms.flat)), len(shape))

  @property
  def transforms(self):
    return self.mktransforms(self.axes, self.root, self.nrefine)

  @property
  def opposites(self):
    nbounds = len(self.axes) - self.ndims
    if nbounds == 0:
      return self.transforms
    axes = [BndAxis(axis.i, axis.j, axis.ibound, not axis.side) if not axis.isdim and axis.ibound==nbounds-1 else axis for axis in self.axes]
    return self.mktransforms(axes, self.root, self.nrefine)

  @property
  def connectivity(self):
    connectivity = numpy.empty(self.shape+(self.ndims,2), dtype=int)
    connectivity[...] = -1
    ielems = numpy.arange(len(self)).reshape(self.shape)
    for idim in range(self.ndims):
      s = (slice(None),)*idim
      s1 = s + (slice(1,None),)
      s2 = s + (slice(0,-1),)
      connectivity[s2+(...,idim,0)] = ielems[s1]
      connectivity[s1+(...,idim,1)] = ielems[s2]
      if idim in self.periodic:
        connectivity[s+(-1,...,idim,0)] = ielems[s+(0,)]
        connectivity[s+(0,...,idim,1)] = ielems[s+(-1,)]
    return types.frozenarray(connectivity.reshape(len(self), self.ndims*2), copy=False)

  @property
  def boundary(self):
    'boundary'

    nbounds = len(self.axes) - self.ndims
    btopo = EmptyTopology(self.ndims-1)
    jdim = 0
    for idim, axis in enumerate(self.axes):
      if not axis.isdim or axis.isperiodic:
        continue
      btopos = [
        StructuredTopology(
          root=self.root,
          axes=self.axes[:idim] + (BndAxis(n,n if not axis.isperiodic else 0,nbounds,side),) + self.axes[idim+1:],
          nrefine=self.nrefine,
          bnames=self._bnames[:jdim*2]+self._bnames[jdim*2+2:])
        for side, n in enumerate((axis.i,axis.j)) ]
      btopo |= UnionTopology(btopos, self._bnames[jdim*2:jdim*2+2])
      jdim += 1
    return btopo

  @property
  def interfaces(self):
    'interfaces'

    assert self.ndims > 0, 'zero-D topology has no interfaces'
    itopos = []
    nbounds = len(self.axes) - self.ndims
    for idim, axis in enumerate(self.axes):
      if not axis.isdim:
        continue
      bndprops = [BndAxis(i, i, ibound=nbounds, side=True) for i in range(axis.i+1, axis.j)]
      if axis.isperiodic:
        assert axis.i == 0
        bndprops.append(BndAxis(axis.j, 0, ibound=nbounds, side=True))
      itopos.append(EmptyTopology(self.ndims-1) if not bndprops
                else UnionTopology(StructuredTopology(self.root, self.axes[:idim] + (axis,) + self.axes[idim+1:], self.nrefine) for axis in bndprops))
    assert len(itopos) == self.ndims
    return UnionTopology(itopos, names=['dir{}'.format(idim) for idim in range(self.ndims)])

  def _basis_spline(self, degree, knotvalues=None, knotmultiplicities=None, continuity=-1, periodic=None):
    'spline with structure information'

    if periodic is None:
      periodic = self.periodic

    if numeric.isint(degree):
      degree = [degree]*self.ndims

    assert len(degree) == self.ndims

    if knotvalues is None or isinstance(knotvalues[0], (int,float)):
      knotvalues = [knotvalues] * self.ndims
    else:
      assert len(knotvalues) == self.ndims

    if knotmultiplicities is None or isinstance(knotmultiplicities[0], int):
      knotmultiplicities = [knotmultiplicities] * self.ndims
    else:
      assert len(knotmultiplicities) == self.ndims

    if not numpy.iterable(continuity):
      continuity = [continuity] * self.ndims
    else:
      assert len(continuity) == self.ndims

    vertex_structure = numpy.array(0)
    stdelems = []
    dofshape = []
    slices = []
    cache = {}
    for idim in range(self.ndims):
      p = degree[idim]
      n = self.shape[idim]
      isperiodic = idim in periodic

      c = continuity[idim]
      if c < 0:
        c += p
      assert -1 <= c < p

      k = knotvalues[idim]
      if k is None: #Defaults to uniform spacing
        k = numpy.arange(n+1)
      else:
        k = numpy.array(k)
        while len(k) < n+1:
          k_ = numpy.empty(len(k)*2-1)
          k_[::2] = k
          k_[1::2] = (k[:-1] + k[1:]) / 2
          k = k_
        assert len(k) == n+1, 'knot values do not match the topology size'

      m = knotmultiplicities[idim]
      if m is None: #Defaults to open spline without internal repetitions
        m = numpy.repeat(p-c, n+1)
        if not isperiodic:
          m[0] = m[-1] = p+1
      else:
        m = numpy.array(m)
        assert min(m) >0 and max(m) <= p+1, 'incorrect multiplicity encountered'
        while len(m) < n+1:
          m_ = numpy.empty(len(m)*2-1, dtype=int)
          m_[::2] = m
          m_[1::2] = p-c
          m = m_
        assert len(m) == n+1, 'knot multiplicity do not match the topology size'

      if not isperiodic:
        nd = sum(m)-p-1
        npre  = p+1-m[0]  #Number of knots to be appended to front
        npost = p+1-m[-1] #Number of knots to be appended to rear
        m[0] = m[-1] = p+1
      else:
        assert m[0]==m[-1], 'Periodic spline multiplicity expected'
        assert m[0]<p+1, 'Endpoint multiplicity for periodic spline should be p or smaller'

        nd = sum(m[:-1])
        npre = npost = 0
        k = numpy.concatenate([k[-p-1:-1]+k[0]-k[-1], k, k[1:1+p]-k[0]+k[-1]])
        m = numpy.concatenate([m[-p-1:-1], m, m[1:1+p]])

      km = numpy.array([ki for ki,mi in zip(k,m) for cnt in range(mi)],dtype=float)
      assert len(km)==sum(m)
      assert nd>0, 'No basis functions defined. Knot vector too short.'

      stdelems_i = []
      slices_i = []
      offsets = numpy.cumsum(m[:-1])-p
      if isperiodic:
        offsets = offsets[p:-p]
      offset0 = offsets[0]+npre

      for offset in offsets:
        start = max(offset0-offset,0) #Zero unless prepending influence
        stop  = p+1-max(offset-offsets[-1]+npost,0) #Zero unless appending influence
        slices_i.append(slice(offset-offset0+start,offset-offset0+stop))
        lknots  = km[offset:offset+2*p] - km[offset] #Copy operation required
        if p: #Normalize for optimized caching
          lknots /= lknots[-1]
        key = (tuple(numeric.round(lknots*numpy.iinfo(numpy.int32).max)), p)
        try:
          coeffs = cache[key]
        except KeyError:
          coeffs = types.frozenarray(self._localsplinebasis(lknots, p).T, copy=False)
          cache[key] = coeffs
        stdelems_i.append(coeffs[start:stop])
      stdelems.append(stdelems_i)

      numbers = numpy.arange(nd)
      if isperiodic:
        numbers = numpy.concatenate([numbers,numbers[:p]])
      vertex_structure = vertex_structure[...,_]*nd+numbers
      dofshape.append(nd)
      slices.append(slices_i)

    #Cache effectivity
    log.debug('Local knot vector cache effectivity: {}'.format(100*(1.-len(cache)/float(sum(self.shape)))))

    # deduplicate stdelems and compute tensorial products `unique` with indices `index`
    # such that unique[index[i,j]] == poly_outer_product(stdelems[0][i], stdelems[1][j])
    index = numpy.array(0)
    for stdelems_i in stdelems:
      unique_i = tuple(set(stdelems_i))
      unique = unique_i if not index.ndim \
        else [numeric.poly_outer_product(a, b) for a in unique for b in unique_i]
      index = index[...,_] * len(unique_i) + tuple(map(unique_i.index, stdelems_i))

    coeffs = [unique[i] for i in index.flat]
    dofmap = [types.frozenarray(vertex_structure[S].ravel(), copy=False) for S in itertools.product(*slices)]
    return coeffs, dofmap, dofshape

  def basis_spline(self, degree, removedofs=None, **kwargs):
    'spline basis'

    if removedofs is None or isinstance(removedofs[0], int):
      removedofs = [removedofs] * self.ndims
    else:
      assert len(removedofs) == self.ndims

    coeffs, dofmap, dofshape = self._basis_spline(degree=degree, **kwargs)
    func = function.polyfunc(coeffs, dofmap, util.product(dofshape), self.transforms)
    if not any(removedofs):
      return func

    mask = numpy.ones((), dtype=bool)
    for idofs, ndofs in zip(removedofs, dofshape):
      mask = mask[...,_].repeat(ndofs, axis=-1)
      if idofs:
        mask[..., [numeric.normdim(ndofs,idof) for idof in idofs]] = False
    assert mask.shape == tuple(dofshape)
    return function.mask(func, mask.ravel())

  @staticmethod
  def _localsplinebasis (lknots, p):

    assert numeric.isarray(lknots), 'Local knot vector should be numpy array'
    assert len(lknots)==2*p, 'Expected 2*p local knots'

    #Based on Algorithm A2.2 Piegl and Tiller
    N    = [None]*(p+1)
    N[0] = numpy.poly1d([1.])

    if p > 0:

      assert numpy.less(lknots[:-1]-lknots[1:], numpy.spacing(1)).all(), 'Local knot vector should be non-decreasing'
      assert lknots[p]-lknots[p-1]>numpy.spacing(1), 'Element size should be positive'

      lknots = lknots.astype(float)

      xi = numpy.poly1d([lknots[p]-lknots[p-1],lknots[p-1]])

      left  = [None]*p
      right = [None]*p

      for i in range(p):
        left[i] = xi - lknots[p-i-1]
        right[i] = -xi + lknots[p+i]
        saved = 0.
        for r in range(i+1):
          temp = N[r]/(lknots[p+r]-lknots[p+r-i-1])
          N[r] = saved+right[r]*temp
          saved = left[i-r]*temp
        N[i+1] = saved

    assert all(Ni.order==p for Ni in N)

    return types.frozenarray([Ni.coeffs for Ni in N]).T[::-1]

  def basis_discont(self, degree):
    'discontinuous shape functions'

    ref = util.product([element.LineReference()]*self.ndims)
    coeffs = [ref.get_poly_coeffs('bernstein', degree=degree)]*len(self)
    ndofs = ref.get_ndofs(degree)
    dofs = types.frozenarray(numpy.arange(ndofs*len(self), dtype=int).reshape(len(self), ndofs), copy=False)
    return function.polyfunc(coeffs, dofs, ndofs*len(self), self.transforms)

  def basis_std(self, degree, removedofs=None, periodic=None):
    'spline from vertices'

    if periodic is None:
      periodic = self.periodic

    if numeric.isint(degree):
      degree = (degree,) * self.ndims

    if removedofs == None:
      removedofs = [None] * self.ndims
    else:
      assert len(removedofs) == self.ndims

    dofshape = []
    slices = []
    vertex_structure = numpy.array(0)
    for idim in range(self.ndims):
      periodic_i = idim in periodic
      n = self.shape[idim]
      p = degree[idim]
      nd = n * p + 1
      numbers = numpy.arange(nd)
      if periodic_i and p > 0:
        numbers[-1] = numbers[0]
        nd -= 1
      vertex_structure = vertex_structure[...,_] * nd + numbers
      dofshape.append(nd)
      slices.append([slice(p*i,p*i+p+1) for i in range(n)])

    lineref = element.LineReference()
    coeffs = [functools.reduce(numeric.poly_outer_product, (lineref.get_poly_coeffs('bernstein', degree=p) for p in degree))]*len(self)
    dofs = [types.frozenarray(vertex_structure[S].ravel(), copy=False) for S in numpy.broadcast(*numpy.ix_(*slices))]
    func = function.polyfunc(coeffs, dofs, numpy.product(dofshape), self.transforms)
    if not any(removedofs):
      return func

    mask = numpy.ones((), dtype=bool)
    for idofs, ndofs in zip(removedofs, dofshape):
      mask = mask[...,_].repeat(ndofs, axis=-1)
      if idofs:
        mask[..., [numeric.normdim(ndofs,idof) for idof in idofs]] = False
    assert mask.shape == tuple(dofshape)
    return function.mask(func, mask.ravel())

  @property
  def refined(self):
    'refine non-uniformly'

    axes = [DimAxis(i=axis.i*2,j=axis.j*2,isperiodic=axis.isperiodic) if axis.isdim
        else BndAxis(i=axis.i*2,j=axis.j*2,ibound=axis.ibound,side=axis.side) for axis in self.axes]
    return StructuredTopology(self.root, axes, self.nrefine+1, bnames=self._bnames)

  def __str__(self):
    'string representation'

    return '{}({})'.format(self.__class__.__name__, 'x'.join(str(n) for n in self.shape))

class UnstructuredTopology(Topology):
  'unstructured topology'

  __slots__ = 'references', 'transforms', 'opposites'

  @types.apply_annotations
  def __init__(self, ndims:types.strictint, references:types.tuple[element.strictreference], transforms:types.tuple[transform.stricttransform], opposites:types.tuple[transform.stricttransform]):
    self.references = references
    self.transforms = TransformsTuple(transforms, ndims)
    self.opposites = TransformsTuple(opposites, ndims)
    assert len(self.references) == len(self.transforms) == len(self.opposites)
    super().__init__(ndims)

class ConnectedTopology(UnstructuredTopology):
  'unstructured topology with connectivity'

  __slots__ = 'connectivity',

  @types.apply_annotations
  def __init__(self, ndims:types.strictint, references:types.tuple[element.strictreference], transforms:types.tuple[transform.stricttransform], opposites:types.tuple[transform.stricttransform], connectivity):
    assert len(connectivity) == len(references)
    self.connectivity = connectivity
    super().__init__(ndims, references, transforms, opposites)

class SimplexTopology(Topology):
  'simpex topology'

  __slots__ = 'simplices', 'references', 'transforms', 'opposites'
  __cache__ = 'connectivity'

  @types.apply_annotations
  def __init__(self, simplices:types.frozenarray[types.strictint], transforms:types.tuple[transform.stricttransform]):
    assert simplices.ndim == 2 and len(simplices) == len(transforms)
    self.simplices = simplices
    ndims = simplices.shape[1]-1
    self.references = (element.getsimplex(ndims),)*len(simplices)
    self.transforms = TransformsTuple(transforms, ndims)
    self.opposites = transforms
    super().__init__(ndims)

  @property
  def connectivity(self):
    connectivity = -numpy.ones((len(self.simplices), self.ndims+1), dtype=int)
    edge_vertices = numpy.arange(self.ndims+1).repeat(self.ndims).reshape(self.ndims, self.ndims+1)[:,::-1].T # nedges x nverts
    v = self.simplices.take(edge_vertices, axis=1).reshape(-1, self.ndims) # (nelems,nedges) x nverts
    o = numpy.lexsort(v.T)
    vo = v.take(o, axis=0)
    i, = numpy.equal(vo[1:], vo[:-1]).all(axis=1).nonzero()
    j = i + 1
    ielems, iedges = divmod(o[i], self.ndims+1)
    jelems, jedges = divmod(o[j], self.ndims+1)
    connectivity[ielems,iedges] = jelems
    connectivity[jelems,jedges] = ielems
    return types.frozenarray(connectivity, copy=False)

  def basis_bubble(self):
    'bubble from vertices'

    bernstein = element.getsimplex(self.ndims).get_poly_coeffs('bernstein', degree=1)
    bubble = functools.reduce(numeric.poly_mul, bernstein)
    coeffs = numpy.zeros((len(bernstein)+1,) + bubble.shape)
    coeffs[(slice(-1),)+(slice(2),)*self.ndims] = bernstein
    coeffs[-1] = bubble
    coeffs[:-1] -= bubble / (self.ndims+1)
    coeffs = types.frozenarray(coeffs, copy=False)
    nverts = self.simplices.max() + 1
    ndofs = nverts + len(self)
    nmap = [types.frozenarray(numpy.hstack([idofs, nverts+ielem]), copy=False) for ielem, idofs in enumerate(self.simplices)]
    return function.polyfunc([coeffs] * len(self), nmap, ndofs, self.transforms)

class UnionTopology(Topology):
  'grouped topology'

  __slots__ = '_topos', '_names'
  __cache__ = '_elements',

  @types.apply_annotations
  def __init__(self, topos:types.tuple[stricttopology], names:types.tuple[types.strictstr]=()):
    self._topos = topos
    self._names = tuple(names)[:len(self._topos)]
    assert len(set(self._names)) == len(self._names), 'duplicate name'
    ndims = self._topos[0].ndims
    assert all(topo.ndims == ndims for topo in self._topos)
    super().__init__(ndims)

  def getitem(self, item):
    topos = [topo if name == item else topo.getitem(item) for topo, name in itertools.zip_longest(self._topos, self._names)]
    return functools.reduce(operator.or_, topos, EmptyTopology(self.ndims))

  def __or__(self, other):
    if not isinstance(other, UnionTopology):
      return UnionTopology(self._topos + (other,), self._names)
    return UnionTopology(self._topos[:len(self._names)] + other._topos + self._topos[len(self._names):], self._names + other._names)

  @property
  def _elements(self):
    references = []
    transforms = []
    opposites =[]
    for trans, indices in util.gather((trans, (itopo, itrans)) for itopo, topo in enumerate(self._topos) for itrans, trans in enumerate(topo.transforms)):
      transforms.append(trans)
      if len(indices) == 1:
        (itopo, itrans), = indices
        references.append(self._topos[itopo].references[itrans])
        opposites.append(self._topos[itopo].opposites[itrans])
      else:
        refs = [self._topos[itopo].references[itrans] for itopo, itrans in indices]
        while len(refs) > 1: # sweep all possible unions until a single reference is left
          nrefs = len(refs)
          iref = 0
          while iref < len(refs)-1:
            for jref in range(iref+1, len(refs)):
              try:
                unionref = refs[iref] | refs[jref]
              except TypeError:
                pass
              else:
                refs[iref] = unionref
                del refs[jref]
                break
            iref += 1
          assert len(refs) < nrefs, 'incompatible elements in union'
        references.append(refs[0])
        opposite, = set(self._topos[itopo].opposites[itrans] for itopo, itrans in indices)
        opposites.append(opposite)
    return references, TransformsTuple(transforms, self.ndims), TransformsTuple(opposites, self.ndims)

  @property
  def references(self):
    return self._elements[0]

  @property
  def transforms(self):
    return self._elements[1]

  @property
  def opposites(self):
    return self._elements[2]

  @property
  def refined(self):
    return UnionTopology([topo.refined for topo in self._topos], self._names)

class SubsetTopology(Topology):
  'trimmed'

  __slots__ = 'refs', 'basetopo', 'newboundary'
  __cache__ = 'connectivity', 'references', 'transforms', 'opposites', 'boundary', 'interfaces'

  @types.apply_annotations
  def __init__(self, basetopo:stricttopology, refs:types.tuple[element.strictreference], newboundary=None):
    if newboundary is not None:
      assert isinstance(newboundary, str) or isinstance(newboundary, Topology) and newboundary.ndims == basetopo.ndims-1
    assert len(refs) == len(basetopo)
    self.refs = refs
    self.basetopo = basetopo
    self.newboundary = newboundary
    super().__init__(basetopo.ndims)

  def getitem(self, item):
    return self.basetopo.getitem(item).subset(self, strict=False)

  def __rsub__(self, other):
    if self.basetopo == other:
      refs = [baseref - ref for baseref, ref in zip(self.basetopo.references, self.refs)]
      return SubsetTopology(self.basetopo, refs, ~self.newboundary if isinstance(self.newboundary,Topology) else self.newboundary)
    return super().__rsub__(other)

  def __or__(self, other):
    if not isinstance(other, SubsetTopology) or self.basetopo != other.basetopo:
      return super().__or__(other)
    refs = [ref1 | ref2 for ref1, ref2 in zip(self.refs, other.refs)]
    if all(baseref == ref for baseref, ref in zip(self.basetopo.references, refs)):
      return self.basetopo
    return SubsetTopology(self.basetopo, refs) # TODO boundary

  @property
  def connectivity(self):
    mask = numpy.array([bool(ref) for ref in self.refs] + [False]) # trailing false serves to map -1 to -1
    renumber = numpy.cumsum(mask)-1
    renumber[~mask] = -1
    return tuple(types.frozenarray(renumber.take(ioppelems).tolist() + [-1] * (ref.nedges - len(ioppelems))) for ref, ioppelems in zip(self.refs, self.basetopo.connectivity) if ref)

  @property
  def references(self):
    return tuple(filter(None, self.refs))

  @property
  def transforms(self):
    return TransformsTuple(tuple(trans for trans, ref in zip(self.basetopo.transforms, self.refs) if ref), self.ndims)

  @property
  def opposites(self):
    return TransformsTuple(tuple(opp for opp, ref in zip(self.basetopo.opposites, self.refs) if ref), self.ndims)

  @property
  def refined(self):
    return self.basetopo.refined.subset(super().refined, self.newboundary.refined if isinstance(self.newboundary,Topology) else self.newboundary, strict=True)

  @property
  def boundary(self):
    baseboundary = self.basetopo.boundary
    superboundary = super().boundary
    brefs = [ref.empty for ref in baseboundary.references]
    trimmedreferences = []
    trimmedtransforms = []
    trimmedopposites = []
    for ref, trans, opp in zip(superboundary.references, superboundary.transforms, superboundary.opposites):
      try:
        ibelem = baseboundary.transforms.index(trans)
      except ValueError:
        trimmedreferences.append(ref)
        trimmedtransforms.append(trans)
        trimmedopposites.append(opp)
      else:
        brefs[ibelem] = ref
    origboundary = SubsetTopology(baseboundary, brefs)
    if isinstance(self.newboundary, Topology):
      trimmedbrefs = [ref.empty for ref in self.newboundary.references]
      for ref, trans in zip(trimmedreferences, trimmedtransforms):
        trimmedbrefs[self.newboundary.transforms.index(trans)] = ref
      trimboundary = SubsetTopology(self.newboundary, trimmedbrefs)
    else:
      trimboundary = OrientedGroupsTopology(self.basetopo.interfaces, trimmedreferences, trimmedtransforms, trimmedopposites)
    return UnionTopology([trimboundary, origboundary], names=[self.newboundary] if isinstance(self.newboundary,str) else [])

  @property
  def interfaces(self):
    baseinterfaces = self.basetopo.interfaces
    superinterfaces = super().interfaces
    irefs = [ref.empty for ref in baseinterfaces.references]
    for ref, trans, opp in zip(superinterfaces.references, superinterfaces.transforms, superinterfaces.opposites):
      try:
        iielem = baseinterfaces.transforms.index(trans)
      except ValueError:
        iielem = baseinterfaces.transforms.index(opp)
      irefs[iielem] = ref
    return SubsetTopology(baseinterfaces, irefs)

  @log.title
  def basis(self, name, *args, **kwargs):
    if isinstance(self.basetopo, HierarchicalTopology):
      warnings.warn('basis may be linearly dependent; a linearly indepent basis is obtained by trimming first, then creating hierarchical refinements')
    basis = self.basetopo.basis(name, *args, **kwargs)
    return self.prune_basis(basis)

class OrientedGroupsTopology(UnstructuredTopology):
  'unstructured topology with undirected semi-overlapping basetopology'

  __slots__ = 'basetopo',

  @types.apply_annotations
  def __init__(self, basetopo:stricttopology, references:types.tuple[element.strictreference], transforms:types.tuple[transform.stricttransform], opposites:types.tuple[transform.stricttransform]):
    self.basetopo = basetopo
    super().__init__(basetopo.ndims, references, transforms, opposites)

  def getitem(self, item):
    references = []
    transforms = []
    opposites = []
    topo = self.basetopo.getitem(item)
    for ref, trans1, trans2 in zip(topo.references, topo.transforms, topo.opposites):
      for trans, opp in ((trans1, trans2), (trans2, trans1)):
        try:
          ielem, tail = self.transforms.index_with_tail(trans)
        except ValueError:
          continue
        if tail:
          raise NotImplementedError
        break
      else:
        continue
      ref = self.references[ielem] & ref
      references.append(ref)
      transforms.append(trans)
      opposites.append(opp)
    return UnstructuredTopology(self.ndims, references, transforms, opposites)

class RefinedTopology(Topology):
  'refinement'

  __slots__ = 'basetopo',
  __cache__ = 'references', 'transforms', 'opposites', 'boundary', 'connectivity'

  @types.apply_annotations
  def __init__(self, basetopo:stricttopology):
    self.basetopo = basetopo
    super().__init__(basetopo.ndims)

  def getitem(self, item):
    return self.basetopo.getitem(item).refined

  @property
  def references(self):
    return tuple(child for ref in self.basetopo.references for child in ref.child_refs if child)

  @property
  def transforms(self):
    return TransformsTuple(tuple(basetrans+(childtrans,) for basetrans, baseref in zip(self.basetopo.transforms, self.basetopo.references) for childtrans, childref in baseref.children if childref), self.ndims)

  @property
  def opposites(self):
    if self.basetopo.transforms is self.basetopo.opposites:
      return self.transforms
    else:
      return TransformsTuple(tuple(basetrans+(childtrans,) for basetrans, baseref in zip(self.basetopo.opposites, self.basetopo.references) for childtrans, childref in baseref.children if childref), self.ndims)

  @property
  def boundary(self):
    return self.basetopo.boundary.refined

  @property
  def connectivity(self):
    offsets = numpy.cumsum([0] + [ref.nchildren for ref in self.basetopo.references])
    connectivity = [offset + edges for offset, ref in zip(offsets, self.basetopo.references) for edges in ref.connectivity]
    for ielem, edges in enumerate(self.basetopo.connectivity):
      for iedge, jelem in enumerate(edges):
        if jelem == -1:
          for ichild, ichildedge in self.references[ielem].edgechildren[iedge]:
            connectivity[offsets[ielem]+ichild][ichildedge] = -1
        elif jelem < ielem:
          jedge = self.basetopo.connectivity[jelem].index(ielem)
          for (ichild, ichildedge), (jchild, jchildedge) in zip(self.references[ielem].edgechildren[iedge], self.references[jelem].edgechildren[jedge]):
            connectivity[offsets[ielem]+ichild][ichildedge] = offsets[jelem]+jchild
            connectivity[offsets[jelem]+jchild][jchildedge] = offsets[ielem]+ichild
    return tuple(types.frozenarray(c, copy=False) for c in connectivity)

class HierarchicalTopology(Topology):
  'collection of nested topology elments'

  __slots__ = 'basetopo', 'references', 'transforms', 'levels', '_ilevels', '_indices', 'levels'
  __cache__ = 'refined', 'boundary', 'interfaces', 'opposites'

  @types.apply_annotations
  def __init__(self, basetopo:stricttopology, transforms:types.tuple[transform.stricttransform]):
    'constructor'

    assert not isinstance(basetopo, HierarchicalTopology)
    self.basetopo = basetopo
    self.transforms = TransformsTuple(transforms, self.basetopo.ndims)

    references = []
    levels = [self.basetopo]
    ilevels = []
    indices =[]
    basetrans = self.basetopo.transforms
    for trans in self.transforms:
      ibase, tail = basetrans.index_with_tail(trans)
      ilevel = len(tail)
      ilevels.append(ilevel)
      while ilevel >= len(levels):
        levels.append(levels[-1].refined)
      level = levels[ilevel]
      ielem = levels[ilevel].transforms.index(trans)
      indices.append(ielem)
      references.append(levels[ilevel].references[ielem])
    self.references = tuple(references)
    self.levels = tuple(levels)
    self._ilevels = tuple(ilevels)
    self._indices = tuple(indices)

    super().__init__(basetopo.ndims)

  @property
  def opposites(self):
    if self.basetopo.transforms == self.basetopo.opposites:
      return self.transforms
    else:
      return TransformsTuple(tuple(self.levels[ilevel].opposites[ielem] for ilevel, ielem in zip(self._ilevels, self._indices)), self.ndims)

  def getitem(self, item):
    itemtopo = self.basetopo.getitem(item)
    return itemtopo.hierarchical(filter(itemtopo.transforms.contains_with_tail, self.transforms))

  def hierarchical(self, elements):
    return self.basetopo.hierarchical(elements)

  @property
  def refined(self):
    transforms = tuple(trans+(ctrans,) for trans, ref in zip(self.transforms, self.references) for ctrans, cref in ref.children if cref)
    return self.basetopo.hierarchical(transforms)

  @property
  @log.title
  def boundary(self):
    'boundary elements'

    basebtopo = self.basetopo.boundary
    edgepool = ((edgeref, trans+(edgetrans,)) for ref, trans in zip(self.references, self.transforms) if self.basetopo.border_transforms.contains_with_tail(trans) for edgetrans, edgeref in ref.edges if edgeref)
    btransforms = []
    for edgeref, edgetrans in edgepool: # superset of boundary elements
      try:
        iedge, tail = basebtopo.transforms.index_with_tail(edgetrans)
      except ValueError:
        pass
      else:
        btransforms.append(edgetrans)
    return basebtopo.hierarchical(btransforms)

  @property
  @log.title
  def interfaces(self):
    'interfaces'

    references = []
    transforms = []
    opposites = []
    for ref, trans, ilevel, ielem in log.zip('elem', self.references, self.transforms, self._ilevels, self._indices):
      level = self.levels[ilevel]
      # Loop over neighbors of `trans`.
      for ielemedge, ineighbor in enumerate(level.connectivity[ielem]):
        if ineighbor < 0:
          # Not an interface.
          continue
        neighbortrans = level.transforms[ineighbor]
        # Lookup `neighbortrans` (from the same `level` as `trans`) in this topology.
        try:
          ignored, neighbortail = self.transforms.index_with_tail(neighbortrans)
        except ValueError:
          # `neighbortrans` not found, hence refinements of `neighbortrans` are
          # present.  The interface of this edge will be added when we
          # encounter the refined elements.
          continue
        # Find the edge of `neighbortrans` between `neighbortrans` and `trans`.
        ineighboredge = level.connectivity[ineighbor].index(ielem)
        if not neighbortail and (ielem, ielemedge) > (ineighbor, ineighboredge):
          # `neighbortrans` itself, not a parent of, exists in this topology
          # (`neighbortail` is empty).  To make sure we add this interface only
          # once we continue here if the current element has a higher index (in
          # `level`) than the neighbor (or a higher edge number if the elements
          # are equal, which might occur when there is only one element in a
          # periodic dimension).
          continue
        # Create and add the interface between `trans` and `neighbortrans`.
        references.append(ref.edge_refs[ielemedge])
        transforms.append(trans+(ref.edge_transforms[ielemedge],))
        opposites.append(neighbortrans+(level.references[ineighbor].edge_transforms[ineighboredge],))
    return UnstructuredTopology(self.ndims-1, references, transforms, opposites)

  @log.title
  @cache.function
  def basis(self, name, *args, truncation_tolerance=1e-15, **kwargs):
    '''Create hierarchical basis.

    A hierarchical basis is constructed from bases on different levels of
    uniform refinement. Two different types of hierarchical bases are
    supported:

    1. Classical -- Starting from the set of all basis functions originating
    from all levels of uniform refinement, only those basis functions are
    selected for which at least one supporting element is part of the
    hierarchical topology.

    2. Truncated -- Like classical, but with basis functions modified such that
    the area of support is reduced. An additional effect of this procedure is
    that it restores partition of unity. The spanned function space remains
    unchanged.

    Truncation is based on linear combinations of basis functions, where fine
    level basis functions are used to reduce the support of coarser level basis
    functions. See `Giannelli et al. 2012`_ for more information on truncated
    hierarchical refinement.

    .. _`Giannelli et al. 2012`: https://pdfs.semanticscholar.org/a858/aa68da617ad9d41de021f6807cc422002258.pdf

    Args
    ----
    name : :class:`str`
      Type of basis function as provided by the base topology, with prefix
      ``h-`` (``h-std``, ``h-spline``) for a classical hierarchical basis and
      prefix ``th-`` (``th-std``, ``th-spline``) for a truncated hierarchical
      basis. For backwards compatibility the ``h-`` prefix is optional, but
      omitting it triggers a deprecation warning as this behaviour will be
      removed in future.
    truncation_tolerance : :class:`float` (default 1e-15)
      In order to benefit from the extra sparsity resulting from truncation,
      vanishing polynomials need to be actively identified and removed from the
      basis. The ``trunctation_tolerance`` offers control over this threshold.

    Returns
    -------
    basis : :class:`nutils.function.Array`
    '''

    split = name.split('-', 1)
    if len(split) != 2 or split[0] not in ('h', 'th'):
      if name == 'discont':
        return super().basis(name, *args, **kwargs)
      warnings.deprecation('hierarchically refined bases will need to be specified using the h- or th- prefix in future')
      truncated = False
    else:
      name = split[1]
      truncated = split[0] == 'th'

    # 1. identify active (supported) and passive (unsupported) basis functions
    ubasis_dofscoeffs = []
    ubasis_active = []
    ubasis_passive = []
    for ltopo in self.levels:
      ubasis = ltopo.basis(name, *args, **kwargs)
      ((ubasis_dofmap,), ubasis_func), = function.blocks(ubasis)
      ubasis_dofscoeffs.append(function.Tuple((ubasis_dofmap, ubasis_func.coeffs)))
      on_current, on_coarser = on_ = numpy.zeros((2, len(ubasis)), dtype=bool)
      for trans in ltopo.transforms:
        try:
          ielem, tail = self.transforms.index_with_tail(trans)
        except ValueError:
          continue
        ubasis_idofs, = ubasis_dofmap.eval(_transforms=(trans,))
        on_[1 if tail else 0, ubasis_idofs] = True
      ubasis_active.append((on_current & ~on_coarser))
      ubasis_passive.append(on_coarser)

    # 2. create consecutive numbering for all active basis functions
    ndofs = 0
    dof_renumber = []
    for myactive in ubasis_active:
      r = myactive.cumsum() + (ndofs-1)
      dof_renumber.append(r)
      ndofs = r[-1]+1

    # 3. construct hierarchical polynomials
    hbasis_dofs = []
    hbasis_coeffs = []
    projectcache = {}

    for hbasis_trans in self.transforms:

      ielem, tail = self.basetopo.transforms.index_with_tail(hbasis_trans) # len(tail) == level of the hierarchical element
      trans_dofs = []
      trans_coeffs = []

      if not truncated: # classical hierarchical basis

        for h in range(len(tail)+1): # loop from coarse to fine
          (mydofs,), (mypoly,) = ubasis_dofscoeffs[h].eval(_transforms=(hbasis_trans,))

          myactive = ubasis_active[h][mydofs]
          if myactive.any():
            trans_dofs.append(dof_renumber[h][mydofs[myactive]])
            trans_coeffs.append(mypoly[myactive])

          if h < len(tail):
            trans_coeffs = [tail[h].transform_poly(c) for c in trans_coeffs]

      else: # truncated hierarchical basis

        for h in reversed(range(len(tail)+1)): # loop from fine to coarse
          (mydofs,), (mypoly,) = ubasis_dofscoeffs[h].eval(_transforms=(hbasis_trans,))

          truncpoly = mypoly if h == len(tail) \
            else numpy.tensordot(numpy.tensordot(tail[h].transform_poly(mypoly), project[...,mypassive], self.ndims), truncpoly[mypassive], 1)

          myactive = ubasis_active[h][mydofs] & numpy.greater(abs(truncpoly), truncation_tolerance).any(axis=tuple(range(1,truncpoly.ndim)))
          if myactive.any():
            trans_dofs.append(dof_renumber[h][mydofs[myactive]])
            trans_coeffs.append(truncpoly[myactive])

          mypassive = ubasis_passive[h][mydofs]
          if not mypassive.any():
            break

          try: # construct least-squares projection matrix
            project = projectcache[mypoly]
          except KeyError:
            P = mypoly.reshape(len(mypoly), -1)
            U, S, V = numpy.linalg.svd(P) # (U * S).dot(V[:len(S)]) == P
            project = (V.T[:,:len(S)] / S).dot(U.T).reshape(mypoly.shape[1:]+mypoly.shape[:1])
            projectcache[mypoly] = project

      # add the dofs and coefficients to the hierarchical basis
      hbasis_dofs.append(numpy.concatenate(trans_dofs))
      hbasis_coeffs.append(numeric.poly_concatenate(trans_coeffs))

    return function.polyfunc(hbasis_coeffs, hbasis_dofs, ndofs, self.transforms)

class ProductTopology(Topology):
  'product topology'

  __slots__ = 'topo1', 'topo2'
  __cache__ = 'references', 'transforms', 'opposites', 'boundary', 'interfaces'

  @types.apply_annotations
  def __init__(self, topo1:stricttopology, topo2:stricttopology):
    assert not isinstance(topo1, ProductTopology)
    self.topo1 = topo1
    self.topo2 = topo2
    super().__init__(topo1.ndims+topo2.ndims)

  def __len__(self):
    return len(self.topo1) * len(self.topo2)

  def __mul__(self, other):
    return ProductTopology(self.topo1, self.topo2 * other)

  @property
  def references(self):
    return tuple(ref1 * ref2 for ref1 in self.topo1.references for ref2 in self.topo2.references)

  @property
  def transforms(self):
    return TransformsTuple(tuple((transform.Bifurcate(trans1, trans2),) for trans1 in self.topo1.transforms for trans2 in self.topo2.transforms), self.ndims)

  @property
  def opposites(self):
    return TransformsTuple(tuple((transform.Bifurcate(opp1, opp2),) if (trans1 != opp1) != (trans2 != opp2) else (transform.Bifurcate(trans1, trans2),) for trans1, opp1 in zip(self.topo1.transforms, self.topo1.opposites) for trans2, opp2 in zip(self.topo2.transforms, self.topo2.opposites)), self.ndims)

  @property
  def refined(self):
    return self.topo1.refined * self.topo2.refined

  def refine(self, n):
    if numpy.iterable(n):
      assert len(n) == self.ndims
    else:
      n = (n,)*self.ndims
    return self.topo1.refine(n[:self.topo1.ndims]) * self.topo2.refine(n[self.topo1.ndims:])

  def getitem(self, item):
    return self.topo1.getitem(item) * self.topo2 | self.topo1 * self.topo2.getitem(item) if isinstance(item, str) \
      else self.topo1[item[:self.topo1.ndims]] * self.topo2[item[self.topo1.ndims:]]

  def basis(self, name, *args, **kwargs):
    def _split(arg):
      if not numpy.iterable(arg):
        return arg, arg
      assert len(arg) == self.ndims
      return tuple(a[0] if all(ai == a[0] for ai in a[1:]) else a for a in (arg[:self.topo1.ndims], arg[self.topo1.ndims:]))
    splitargs = [_split(arg) for arg in args]
    splitkwargs = [(name,)+_split(arg) for name, arg in kwargs.items()]
    basis1, basis2 = function.bifurcate(
      self.topo1.basis(name, *[arg1 for arg1, arg2 in splitargs], **{name: arg1 for name, arg1, arg2 in splitkwargs}),
      self.topo2.basis(name, *[arg2 for arg1, arg2 in splitargs], **{name: arg2 for name, arg1, arg2 in splitkwargs}))
    return function.ravel(function.outer(basis1,basis2), axis=0)

  @property
  def boundary(self):
    return self.topo1 * self.topo2.boundary + self.topo1.boundary * self.topo2

  @property
  def interfaces(self):
    return self.topo1 * self.topo2.interfaces + self.topo1.interfaces * self.topo2

class RevolutionTopology(Topology):
  'topology consisting of a single revolution element'

  __slots__ = 'references', 'transforms', 'opposites', 'boundary'

  def __init__(self):
    self.references = element.RevolutionReference(),
    self.opposites = self.transforms = TransformsTuple([(transform.Identifier(1, 'angle'),)], 1)
    self.boundary = EmptyTopology(ndims=0)
    super().__init__(ndims=1)

  def basis(self, name, *args, **kwargs):
    return function.asarray([1])

class PatchBoundary(types.Singleton):

  __slots__ = 'id', 'dim', 'side', 'reverse', 'transpose'

  @types.apply_annotations
  def __init__(self, id:types.tuple[types.strictint], dim, side, reverse:types.tuple[bool], transpose:types.tuple[types.strictint]):
    super().__init__()
    self.id = id
    self.dim = dim
    self.side = side
    self.reverse = reverse
    self.transpose = transpose

  def apply_transform(self, array):
    return array[tuple(slice(None, None, -1) if i else slice(None) for i in self.reverse)].transpose(self.transpose)

class Patch(types.Singleton):

  __slots__ = 'topo', 'verts', 'boundaries'

  @types.apply_annotations
  def __init__(self, topo:stricttopology, verts:types.frozenarray, boundaries:types.tuple[types.strict[PatchBoundary]]):
    super().__init__()
    self.topo = topo
    self.verts = verts
    self.boundaries = boundaries

class MultipatchTopology(Topology):
  'multipatch topology'

  __slots__ = 'patches',
  __cache__ = '_patchinterfaces', 'references', 'transforms', 'opposites', 'boundary', 'interfaces', 'refined'

  @staticmethod
  def build_boundarydata(connectivity):
    'build boundary data based on connectivity'

    boundarydata = []
    for patch in connectivity:
      ndims = len(patch.shape)
      patchboundarydata = []
      for dim, side in itertools.product(range(ndims), [-1, 0]):
        # ignore vertices at opposite face
        verts = numpy.array(patch)
        opposite = tuple({0:-1, -1:0}[side] if i == dim else slice(None) for i in range(ndims))
        verts[opposite] = verts.max()+1
        if len(set(verts.flat)) != 2**(ndims-1)+1:
          raise NotImplementedError('Cannot compute canonical boundary if vertices are used more than once.')
        # reverse axes such that lowest vertex index is at first position
        reverse = tuple(map(bool, numpy.unravel_index(verts.argmin(), verts.shape)))
        verts = verts[tuple(slice(None, None, -1) if i else slice(None) for i in reverse)]
        # transpose such that second lowest vertex connects to lowest vertex in first dimension, third in second dimension, et cetera
        k = [verts[tuple(1 if i == j else 0 for j in range(ndims))] for i in range(ndims)]
        transpose = tuple(sorted(range(ndims), key=k.__getitem__))
        verts = verts.transpose(transpose)
        # boundarid
        boundaryid = tuple(verts[...,0].flat)
        patchboundarydata.append(PatchBoundary(boundaryid,dim,side,reverse,transpose))
      boundarydata.append(tuple(patchboundarydata))

    # TODO: boundary sanity checks

    return boundarydata

  @types.apply_annotations
  def __init__(self, patches:types.tuple[types.strict[Patch]]):
    'constructor'

    self.patches = patches

    super().__init__(self.patches[0].topo.ndims)

  @property
  def _patchinterfaces(self):
    patchinterfaces = {}
    for patch in self.patches:
      for boundary in patch.boundaries:
        patchinterfaces.setdefault(boundary.id, []).append((patch.topo, boundary))
    return {
      boundaryid: tuple(data)
      for boundaryid, data in patchinterfaces.items()
      if len(data) > 1
    }

  @property
  def references(self):
    return tuple(itertools.chain.from_iterable(patch.topo.references for patch in self.patches))

  @property
  def transforms(self):
    return TransformsTuple(tuple(itertools.chain.from_iterable(patch.topo.transforms for patch in self.patches)), self.ndims)

  @property
  def opposites(self):
    return TransformsTuple(tuple(itertools.chain.from_iterable(patch.topo.opposites for patch in self.patches)), self.ndims)

  def getitem(self, key):
    for i in range(len(self.patches)):
      if key == 'patch{}'.format(i):
        return self.patches[i].topo
    else:
      return UnionTopology(patch.topo.getitem(key) for patch in self.patches)

  def basis_spline(self, degree, patchcontinuous=True, knotvalues=None, knotmultiplicities=None):
    '''spline from vertices

    Create a spline basis with degree ``degree`` per patch.  If
    ``patchcontinuous``` is true the basis is $C^0$-continuous at patch
    interfaces.
    '''

    if knotvalues is None:
      knotvalues = {None: None}
    else:
      knotvalues, _knotvalues = {}, knotvalues
      for edge, k in _knotvalues.items():
        if k is None:
          rk = None
        else:
          k = tuple(k)
          rk = k[::-1]
        if edge is None:
          knotvalues[edge] = k
        else:
          l, r = edge
          assert (l,r) not in knotvalues
          assert (r,l) not in knotvalues
          knotvalues[(l,r)] = k
          knotvalues[(r,l)] = rk

    if knotmultiplicities is None:
      knotmultiplicities = {None: None}
    else:
      knotmultiplicities, _knotmultiplicities = {}, knotmultiplicities
      for edge, k in _knotmultiplicities.items():
        if k is None:
          rk = None
        else:
          k = tuple(k)
          rk = k[::-1]
        if edge is None:
          knotmultiplicities[edge] = k
        else:
          l, r = edge
          assert (l,r) not in knotmultiplicities
          assert (r,l) not in knotmultiplicities
          knotmultiplicities[(l,r)] = k
          knotmultiplicities[(r,l)] = rk

    missing = object()

    coeffs = []
    dofmap = []
    dofcount = 0
    commonboundarydofs = {}
    for ipatch, patch in enumerate(self.patches):
      # build structured spline basis on patch `patch.topo`
      patchknotvalues = []
      patchknotmultiplicities = []
      for idim in range(self.ndims):
        left = tuple(0 if j == idim else slice(None) for j in range(self.ndims))
        right = tuple(1 if j == idim else slice(None) for j in range(self.ndims))
        dimknotvalues = set()
        dimknotmultiplicities = set()
        for edge in zip(patch.verts[left].flat, patch.verts[right].flat):
          v = knotvalues.get(edge, knotvalues.get(None, missing))
          m = knotmultiplicities.get(edge, knotmultiplicities.get(None, missing))
          if v is missing:
            raise 'missing edge'
          dimknotvalues.add(v)
          if m is missing:
            raise 'missing edge'
          dimknotmultiplicities.add(m)
        if len(dimknotvalues) != 1:
          raise 'ambiguous knot values for patch {}, dimension {}'.format(ipatch, idim)
        if len(dimknotmultiplicities) != 1:
          raise 'ambiguous knot multiplicities for patch {}, dimension {}'.format(ipatch, idim)
        patchknotvalues.append(next(iter(dimknotvalues)))
        patchknotmultiplicities.append(next(iter(dimknotmultiplicities)))
      patchcoeffs, patchdofmap, patchdofcount = patch.topo._basis_spline(degree, knotvalues=patchknotvalues, knotmultiplicities=patchknotmultiplicities)
      coeffs.extend(patchcoeffs)
      dofmap.extend(types.frozenarray(dofs+dofcount, copy=False) for dofs in patchdofmap)
      if patchcontinuous:
        # reconstruct multidimensional dof structure
        dofs = dofcount + numpy.arange(numpy.prod(patchdofcount), dtype=int).reshape(patchdofcount)
        for boundary in patch.boundaries:
          # get patch boundary dofs and reorder to canonical form
          boundarydofs = boundary.apply_transform(dofs)[...,0].ravel()
          # append boundary dofs to list (in increasing order, automatic by outer loop and dof increment)
          commonboundarydofs.setdefault(boundary.id, []).append(boundarydofs)
      dofcount += numpy.prod(patchdofcount)

    if patchcontinuous:
      # build merge mapping: merge common boundary dofs (from low to high)
      pairs = itertools.chain(*(zip(*dofs) for dofs in commonboundarydofs.values() if len(dofs) > 1))
      merge = {}
      for dofs in sorted(pairs):
        dst = merge.get(dofs[0], dofs[0])
        for src in dofs[1:]:
          merge[src] = dst
      # build renumber mapping: renumber remaining dofs consecutively, starting at 0
      remainder = set(merge.get(dof, dof) for dof in range(dofcount))
      renumber = dict(zip(sorted(remainder), range(len(remainder))))
      # apply mappings
      dofmap = tuple(types.frozenarray(tuple(renumber[merge.get(dof, dof)] for dof in v.flat), dtype=int).reshape(v.shape) for v in dofmap)
      dofcount = len(remainder)

    return function.polyfunc(coeffs, dofmap, dofcount, self.transforms)

  def basis_discont(self, degree):
    'discontinuous shape functions'

    bases = [patch.topo.basis('discont', degree=degree) for patch in self.patches]
    coeffs = []
    dofs = []
    ndofs = 0
    for patch in self.patches:
      basis = patch.topo.basis('discont', degree=degree)
      (axes,func), = function.blocks(basis)
      patch_dofmap, = axes
      if isinstance(func, function.Polyval):
        patch_coeffs = func.coeffs
        assert patch_coeffs.ndim == 1+self.ndims
      elif func.isconstant:
        assert func.ndim == 1
        patch_coeffs = func[(slice(None),*(_,)*self.ndims)]
      else:
        raise ValueError
      patch_coeffs_dofs = function.Tuple((patch_coeffs, patch_dofmap))
      for trans in patch.topo.transforms:
        (elem_coeffs,), (elem_dofs,) = patch_coeffs_dofs.eval(_transforms=(trans,))
        coeffs.append(elem_coeffs)
        dofs.append(types.frozenarray(ndofs+elem_dofs, copy=False))
      ndofs += len(basis)
    return function.polyfunc(coeffs, dofs, ndofs, self.transforms)

  def basis_patch(self):
    'degree zero patchwise discontinuous basis'

    npatches = len(self.patches)
    coeffs = [types.frozenarray(1, dtype=int).reshape(1, *(1,)*self.ndims)]*npatches
    dofs = types.frozenarray(range(npatches), dtype=int)[:,_]
    return function.polyfunc(coeffs, dofs, npatches, ((patch.topo.root,) for patch in self.patches))

  @property
  def boundary(self):
    'boundary'

    subtopos = []
    subnames = []
    for i, patch in enumerate(self.patches):
      names = dict(zip(itertools.product(range(self.ndims), [0,-1]), patch.topo._bnames))
      for boundary in patch.boundaries:
        if boundary.id in self._patchinterfaces:
          continue
        subtopos.append(patch.topo.boundary[names[boundary.dim,boundary.side]])
        subnames.append('patch{}-{}'.format(i, names[boundary.dim,boundary.side]))
    if len(subtopos) == 0:
      return EmptyTopology(self.ndims-1)
    else:
      return UnionTopology(subtopos, subnames)

  @property
  def interfaces(self):
    '''interfaces

    Return a topology with all element interfaces.  The patch interfaces are
    accessible via the group ``'interpatch'`` and the interfaces *inside* a
    patch via ``'intrapatch'``.
    '''

    intrapatchtopo = EmptyTopology(self.ndims-1) if not self.patches else \
      UnionTopology(patch.topo.interfaces for patch in self.patches)

    btopos = []
    bconnectivity = []
    for boundaryid, patchdata in self._patchinterfaces.items():
      if len(patchdata) > 2:
        raise ValueError('Cannot create interfaces of multipatch topologies with more than two interface connections.')
      pairs = []
      references = None
      for topo, boundary in patchdata:
        names = dict(zip(itertools.product(range(self.ndims), [0,-1]), topo._bnames))
        btopo = topo.boundary[names[boundary.dim, boundary.side]]
        if references is None:
          references = numeric.asobjvector(btopo.references).reshape(btopo.shape)
          references = references[tuple(_ if i == boundary.dim else slice(None) for i in range(self.ndims))]
          references = boundary.apply_transform(references)[..., 0]
          references = tuple(references.flat)
        transforms = numeric.asobjvector(btopo.transforms).reshape(btopo.shape)
        transforms = transforms[tuple(_ if i == boundary.dim else slice(None) for i in range(self.ndims))]
        transforms = boundary.apply_transform(transforms)[..., 0]
        pairs.append(tuple(transforms.flat))
      # create structured topology of joined element pairs
      transforms, opposites = pairs
      btopos.append(UnstructuredTopology(self.ndims-1, references, transforms, opposites))
      bconnectivity.append(numpy.array(boundaryid).reshape((2,)*(self.ndims-1)))
    # create multipatch topology of interpatch boundaries
    interpatchtopo = MultipatchTopology(tuple(map(Patch, btopos, bconnectivity, self.build_boundarydata(bconnectivity))))

    return UnionTopology((intrapatchtopo, interpatchtopo), ('intrapatch', 'interpatch'))

  @property
  def refined(self):
    'refine'

    return MultipatchTopology(Patch(patch.topo.refined, patch.verts, patch.boundaries) for patch in self.patches)

# UTILITY FUNCTIONS

def common_refine(topo1, topo2):
  warnings.deprecation('common_refine(a, b) will be removed in future; use a & b instead')
  return topo1 & topo2

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
