# -*- coding: utf8 -*-
#
# Module ELEMENT
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The transform module.
"""

from __future__ import print_function, division
from . import cache, rational, numeric, core
import numpy


class TransformChain( tuple ):

  __slots__ = ()

  def slicefrom( self, i ):
    return TransformChain( self[i:] )

  def sliceto( self, j ):
    return TransformChain( self[:j] )

  @property
  def todims( self ):
    return self[0].todims

  @property
  def fromdims( self ):
    return self[-1].fromdims

  @property
  def isflipped( self ):
    return sum( trans.isflipped for trans in self ) % 2 == 1

  def orientation( self, ndims ):
    # returns orientation (+1/-1) for ndims->ndims+1 transformation
    index = core.index( trans.fromdims == ndims for trans in self )
    updim = self[index]
    assert updim.todims == ndims+1
    return -1 if updim.isflipped else 1

  def lookup( self, transforms ):
    # to be replaced by bisection soon
    headtrans = self
    while headtrans:
      if headtrans in transforms:
        return headtrans
      headtrans = headtrans.sliceto(-1)
    return None

  def split( self, ndims ):
    if self[-1].todims != ndims:
      assert self[-1].fromdims == ndims
      return self, identity
    i = core.index( trans.todims == ndims for trans in self )
    return self.sliceto(i), self.slicefrom(i)

  def __lshift__( self, other ):
    # self << other
    assert isinstance( other, TransformChain )
    if not self:
      return other
    if not other:
      return self
    assert self.fromdims == other.todims
    return TransformChain( self + other )

  def __rshift__( self, other ):
    # self >> other
    assert isinstance( other, TransformChain )
    return other << self

  @property
  def flipped( self ):
    assert len(self) == 1
    return TransformChain( trans.flipped for trans in self )

  @property
  def det( self ):
    det = 1
    for trans in self:
      det *= trans.det
    return det

  @property
  def offset( self ):
    offset = self[-1].offset
    for trans in self[-2::-1]:
      offset = trans.apply( offset )
    return offset

  @property
  def linear( self ):
    linear = rational.unit
    for trans in self:
      linear = rational.dot( linear, trans.linear ) if linear.ndim and trans.linear.ndim \
          else linear * trans.linear
    return linear

  @property
  def invlinear( self ):
    invlinear = rational.unit
    for trans in self:
      invlinear = rational.dot( trans.invlinear, invlinear ) if invlinear.ndim and trans.linear.ndim \
             else trans.invlinear * invlinear
    return invlinear

  def apply( self, points ):
    for trans in reversed(self):
      points = trans.apply( points )
    return points

  def __str__( self ):
    return ' << '.join( str(trans) for trans in self ) if self else '='

  def __repr__( self ):
    return 'TransformChain( %s )' % (self,)

  @property
  def flat( self ):
    return self if len(self) == 1 \
      else affine( self.linear, self.offset, isflipped=self.isflipped )

  @property
  def canonical( self ):
    # Keep at lowest ndims possible. The reason is that in this form we can do
    # lookups of embedded elements.
    items = list( self )
    for i in range(len(items)-1)[::-1]:
      trans1, trans2 = items[i:i+2]
      if isinstance( trans1, Scale ) and trans2.todims == trans2.fromdims + 1:
        try:
          newscale = solve( TransformChain([trans2]), TransformChain([trans1,trans2]) )
        except numpy.linalg.LinAlgError:
          pass
        else:
          items[i:i+2] = trans2, newscale
    return TransformChain( items )

  def promote( self, ndims ):
    if ndims == self.fromdims:
      return self
    index = core.index( trans.fromdims == self.fromdims for trans in self )
    body = list( self[:index] )
    uptrans = self[index]
    assert uptrans.todims == self.fromdims+1
    for i in range( index+1, len(self) ):
      scale = self[i]
      if not isinstance( scale, Scale ):
        break
      newscale = Scale( scale.linear, uptrans.apply(scale.offset) - scale.linear * uptrans.offset )
      body.append( newscale )
    else:
      i = len(self)+1
    assert equivalent( body[index:]+[uptrans], self[index:i] )
    return TransformChain( TransformChain(body).promote(ndims)+(uptrans,)+self[i:] )


## TRANSFORM ITEMS

class TransformItem( cache.Immutable ):

  def __init__( self, todims, fromdims, isflipped ):
    self.todims = todims
    self.fromdims = fromdims
    self.isflipped = isflipped

  __lt__ = lambda self, other: id(self) <  id(other)
  __gt__ = lambda self, other: id(self) >  id(other)
  __le__ = lambda self, other: id(self) <= id(other)
  __ge__ = lambda self, other: id(self) >= id(other)

  def __repr__( self ):
    return '%s( %s )' % ( self.__class__.__name__, self )

class Shift( TransformItem ):

  def __init__( self, offset ):
    self.linear = self.invlinear = self.det = rational.unit
    self.offset = offset
    assert offset.ndim == 1
    TransformItem.__init__( self, offset.shape[0], offset.shape[0], False )

  @property
  def flipped( self ):
    return self

  def apply( self, points ):
    return points + self.offset

  def __str__( self ):
    return 'x + %s' % self.offset

class Scale( TransformItem ):

  def __init__( self, linear, offset ):
    assert linear.ndim == 0 and offset.ndim == 1
    self.linear = linear
    self.offset = offset
    TransformItem.__init__( self, offset.shape[0], offset.shape[0], False )

  @property
  def flipped( self ):
    return self

  def apply( self, points ):
    return self.linear * points + self.offset

  @property
  def det( self ):
    return self.linear**self.todims

  @property
  def invlinear( self ):
    return 1 / self.linear

  def __str__( self ):
    return '%s x + %s' % ( self.linear, self.offset )

class Matrix( TransformItem ):

  def __init__( self, linear, offset, isflipped ):
    self.linear = linear
    self.offset = offset
    assert linear.ndim == 2 and offset.shape == linear.shape[:1]
    TransformItem.__init__( self, linear.shape[0], linear.shape[1], isflipped )

  @property
  def flipped( self ):
    return self if self.fromdims == self.todims \
      else Matrix( self.linear, self.offset, not self.isflipped )

  def apply( self, points ):
    assert points.shape[-1] == self.fromdims
    return rational.dot( points, self.linear.T ) + self.offset

  @cache.property
  def det( self ):
    return rational.det( self.linear )

  @cache.property
  def invlinear( self ):
    return rational.invdet( self.linear ) / self.det

  def __str__( self ):
    return '%s x + %s' % ( self.linear, self.offset )

class VertexTransform( TransformItem ):

  def __init__( self, fromdims ):
    TransformItem.__init__( self, None, fromdims, False )

class MapTrans( VertexTransform ):

  def __init__( self, coords, vertices ):
    self.coords = numpy.asarray(coords)
    assert numeric.isintarray( self.coords )
    nverts, ndims = coords.shape
    assert len(vertices) == nverts
    self.vertices = tuple(vertices)
    VertexTransform.__init__( self, ndims )

  def apply( self, coords ):
    assert coords.ndim == 2
    coords = rational.asarray( coords )
    assert coords.denom == 1 # for now
    indices = map( self.coords.tolist().index, coords.numer.tolist() )
    return [ self.vertices[n] for n in indices ]

  def __str__( self ):
    return ','.join( str(v) for v in self.vertices )

class RootTrans( VertexTransform ):

  def __init__( self, name, shape ):
    VertexTransform.__init__( self, len(shape) )
    self.I, = numpy.where( shape )
    self.w = numpy.take( shape, self.I )
    self.fmt = name+'{}'

  def apply( self, coords ):
    assert coords.ndim == 2
    coords = rational.asarray( coords )
    if self.I.size:
      ci = coords.numer.copy()
      wi = self.w * coords.denom
      ci[:,self.I] = ci[:,self.I] % wi
      coords = rational.Rational( ci, coords.denom )
    return [ self.fmt.format(c) for c in coords ]

  def __str__( self ):
    return repr( self.fmt.format('*') )

class RootTransEdges( VertexTransform ):

  def __init__( self, name, shape ):
    VertexTransform.__init__( self, len(shape) )
    self.shape = shape
    assert isinstance( name, numpy.ndarray )
    assert name.shape == (3,)*len(shape)
    self.name = name.copy()

  def apply( self, coords ):
    assert coords.ndim == 2
    labels = []
    for coord in coords.T.frac.T:
      right = (coord[:,1]==1) & (coord[:,0]==self.shape)
      left = coord[:,0]==0
      where = (1+right)-left
      s = self.name[tuple(where)] + '[%s]' % ','.join( str(n) if d == 1 else '%d/%d' % (n,d) for n, d in coord[where==1] )
      labels.append( s )
    return labels

  def __str__( self ):
    return repr( ','.join(self.name.flat)+'*' )


## CONSTRUCTORS

def affine( linear=None, offset=None, numer=1, isflipped=False ):
  r_offset = rational.asarray( offset ) / numer if offset is not None else rational.zeros( len(linear) )
  n, = r_offset.shape
  if linear is None:
    r_linear = rational.unit
  else:
    r_linear = rational.asarray( linear ) / numer
    if r_linear.ndim == 2:
      assert r_linear.shape[0] == n
      if r_linear.shape[1] == n:
        if n == 0:
          r_linear = rational.unit
        elif n == 1 or r_linear.numer[0,-1] == 0 and numpy.all( r_linear.numer == r_linear.numer[0,0] * numpy.eye(n) ):
          r_linear = r_linear[0,0]
    else:
      assert r_linear.ndim == 0
  return TransformChain((
         Matrix( r_linear, r_offset, isflipped ) if r_linear.ndim
    else Scale( r_linear, r_offset ) if r_linear != rational.unit
    else Shift( r_offset ), ))

def simplex( coords, isflipped=False ):
  offset = coords[0]
  return affine( (coords[1:]-offset).T, offset, isflipped=isflipped )

def roottrans( name, shape ):
  return TransformChain(( RootTrans( name, shape ), ))

def roottransedges( name, shape ):
  return TransformChain(( RootTransEdges( name, shape ), ))

def maptrans( coords, vertices ):
  return TransformChain(( MapTrans( coords, vertices ), ))

def equivalent( trans1, trans2 ):
  trans1 = TransformChain( trans1 )
  trans2 = TransformChain( trans2 )
  return numpy.all( trans1.linear == trans2.linear ) and numpy.all( trans1.offset == trans2.offset )


## UTILITY FUNCTIONS

identity = TransformChain()

def solve( transA, transB ): # A << X = B
  assert transA.isflipped == transB.isflipped
  assert transA.fromdims == transB.fromdims and transA.todims == transB.todims
  X, x = rational.solve( transA.linear, transB.linear, transB.offset - transA.offset )
  transX = affine( X, x, isflipped=False )
  assert ( transA << transX ).flat == transB.flat
  return transX

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
