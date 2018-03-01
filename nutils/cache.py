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
The cache module.
"""

from . import config, log, types
import os, numpy, functools, inspect, builtins, pathlib, pickle, hashlib

def property(f):
  _self = object()
  _temp = object()
  _name = f.__name__
  def property_getter(self):
    try:
      dictvalue = self.__dict__[_name]
    except KeyError:
      self.__dict__[_name] = _temp # placeholder for detection of cyclic dependencies
      value = f(self)
      self.__dict__[_name] = value if value is not self else _self
    else:
      assert dictvalue is not _temp, 'attribute {!r} requested during construction'.format(_name)
      value = dictvalue if dictvalue is not _self else self
    return value
  def property_setter(self, value):
    assert _name not in self.__dict__, 'attempting to set attribute {!r} twice'.format(_name)
    self.__dict__[_name] = value if value is not self else _self
  return builtins.property(fget=property_getter, fset=property_setter)

class Wrapper:
  'function decorator that caches results by arguments'

  def __init__(self, func):
    self.func = func
    self.cache = {}
    self.count = 0
    self.signature = inspect.signature(func)

  def __call__(self, *args, **kwargs):
    self.count += 1
    bound = self.signature.bind(*args, **kwargs)
    bound.apply_defaults()
    args = bound.args
    assert not bound.kwargs
    try:
      value = self.cache[args]
    except KeyError:
      value = self.func(*args)
      self.cache[args] = value
    return value

  @builtins.property
  def hits(self):
    return self.count - len(self.cache)

class WrapperCache:
  'maintains a cache for Wrapper instances'

  def __init__(self):
    self.cache = {}

  def __getitem__(self, func):
    try:
      wrapper = self.cache[func]
    except KeyError:
      wrapper = Wrapper(func)
      self.cache[func] = wrapper
    return wrapper

  @builtins.property
  def stats(self):
    hits = count = 0
    for wrapper in self.cache.values():
      hits += wrapper.hits
      count += wrapper.count
    return 'not used' if not count \
      else 'effectivity {}% (hit {}/{} calls over {} functions)'.format(100*hits/count, hits, count, len(self.cache))

  @property
  def __nutils_hash__(self):
    return hashlib.sha1(b'nutils.cache.WrapperCache\0').digest()

class WrapperDummyCache:
  'placeholder object'

  stats = 'caching disabled'

  def __getitem__(self, func):
    return func

  @property
  def __nutils_hash__(self):
    # This hash is intentionally the same as `WrapperCache`: Both can be
    # interchanged without affecting results.
    return hashlib.sha1(b'nutils.cache.WrapperCache\0').digest()

class FileCache:
  'cache'

  def __init__(self, *args):
    'constructor'

    import os, numpy, hashlib, pickle
    serial = pickle.dumps(args, -1)
    self.myhash = hash(serial)
    hexhash = hashlib.md5(serial).hexdigest()
    cachedir = config.cachedir
    if not os.path.exists(cachedir):
      os.makedirs(cachedir)
    path = os.path.join(cachedir, hexhash)
    if not os.path.isfile(path) or config.recache:
      log.info('starting new cache:', hexhash)
      data = open(path, 'wb+')
      data.write(serial)
      data.flush()
    else:
      log.info('continuing from cache:', hexhash)
      data = open(path, 'ab+')
      data.seek(0)
      recovered_args = pickle.load(data)
      assert recovered_args == args, 'hash clash'
    self.data = data

  def __call__(self, func, *args, **kwargs):
    'call'

    try:
      import cPickle as pickle
    except ImportError:
      import pickle
    name = func.__name__ + ''.join(' {}'.format(arg) for arg in args) + ''.join(' {}={}'.format(*item) for item in kwargs.items())
    pos = self.data.tell()
    try:
      data = pickle.load(self.data)
    except EOFError:
      data = func(*args, **kwargs)
      self.data.seek(pos)
      pickle.dump(data, self.data, -1)
      self.data.flush()
      msg = 'written to'
    else:
      msg = 'loaded from'
    log.info(msg, 'cache:', name, '[{}b]'.format(self.data.tell()-pos))
    return data

  def truncate(self):
    log.info('truncating cache')
    self.data.truncate()

  def __hash__(self):
    return self.myhash

def replace(func):
  '''decorator for deep object replacement

  Generates a deep replacement method for Immutable objects based on a callable
  that is applied (recursively) on individual constructor arguments.

  Args
  ----
  func : callable which maps (obj, ...) onto replaced_obj

  Returns
  -------
  callable
      The method that searches the object to perform the replacements.
  '''

  @functools.wraps(func)
  def wrapped(target, *funcargs, **funckwargs):
    cache = {}
    def op(obj):
      try:
        replaced = cache[obj]
      except TypeError: # unhashable
        replaced = obj
      except KeyError:
        replaced = func(obj, *funcargs, **funckwargs)
        if replaced is None:
          replaced = obj.edit(op) if isinstance(obj, types.Immutable) else obj
        cache[obj] = replaced
      return replaced
    retval = op(target)
    del op
    return retval

  return wrapped

# Define platform-dependent `_lock_file` function.
def _lock_file_fallback(f): pass

try:
  import fcntl
except ImportError:
  _lock_file_fcntl = None
else:
  # On Linux and BSD (including macOS) we use `flock`, interfaced by Python via
  # `fcntl.flock`.  The lock is exclusive, tied to the file descriptor (and not
  # to the process as is `lockf`) and is released automatically when the file
  # descriptor is closed.
  def _lock_file_fcntl(f):
    fcntl.flock(f, fcntl.LOCK_EX)

try:
  import msvcrt
except ImportError:
  _lock_file_msvcrt = None
else:
  # On Windows we use `msvcrt.locking`.  We lock the first byte at the current
  # position of the file.  Like `fcntl.flock` the lock is exclusive, tied to
  # the file descriptor and released automatically when the file descriptor is
  # closed.  `msvcrt.locking` tries to lock the file descriptor ten times with
  # an interval of a second, and raises `OSError` if unsuccessfull.  Hence the
  # `while: try ... except OSError: pass` construction.
  def _lock_file_msvcrt(f):
    while True:
      try:
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
      except OSError:
        pass
      else:
        return

_lock_file = next(filter(None, [_lock_file_fcntl, _lock_file_msvcrt, _lock_file_fallback]))


def function(func=None, *, version=0):
  '''
  Decorator to wrap a function ``func`` with a memoizing callable.  It is
  assumed that ``func`` computes its return value based strictly on the
  arguments.  In other words: calling ``func`` with the same arguments
  repeatedly, should produce the same return value.  All arguments passed to
  the decorator should be hashable (by :func:`nutils.types.nutils_hash`).

  Memoization is controlled by the ``nutils.config.cache`` variable.  If
  ``True``, memoization is enabled: The first time the decorator is called with
  a unique set of arguments, the decorator calls ``func`` and stores the result
  on disk in the directory specified by ``nutils.config.cachedir``; when the
  decorator is called with the same arguments, the result is retrieved from the
  cache.  If ``False``, the decorator calls ``func`` directly, bypassing the
  cache.  Note that ``nutils.config.cachedir`` defaults to ``False``.

  Parameters
  ----------
  func : :any:`callable`
      The function to be memoized.
  version : :class:`int`
      Optional version number of ``func``.  Increment this if the behavior of
      ``func`` is changed.  The decorator can be applied as follows:

      >>> @function(version=1)
      ... def f(x):
      ...   return x

  Returns
  -------
  :any:`callable`
      A memoized version of ``func``.
  '''

  if not isinstance(version, int):
    raise ValueError("'version' should be of type 'int' but got {!r}".format(version))
  if func is None:
    return functools.partial(function, version=version)

  # Hash of the full function name (closest thing to a unique representation of
  # `func`).
  func_key = hashlib.sha1('{}.{}:{}'.format(func.__module__, func.__qualname__, version).encode()).digest()
  canonicalize = types.argument_canonicalizer(inspect.signature(func))

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    if not config.cache:
      return func(*args, **kwargs)
    args, kwargs = canonicalize(*args, **kwargs)
    # Hash the function key and the canonicalized arguments and compute the
    # hexdigest.  This is used to identify cache file `cachefile`.
    h = hashlib.sha1(func_key)
    for arg in args:
      h.update(types.nutils_hash(arg))
    for hkv in sorted(hashlib.sha1(k.encode()).digest()+types.nutils_hash(v) for k, v in kwargs.items()):
      h.update(hkv)
    hkey = h.hexdigest()
    cachefile = pathlib.Path(config.cachedir)/hkey
    # Open and lock `cachefile`.  Try to read it and, if successful, unlock
    # the file (implicitly by closing the file) and return the value.  If
    # reading fails, e.g. because the file did not exist, call `func`, store
    # the result, unlock and return.  While not necessary per se, we lock the
    # file immediately to avoid checking twice if there is a cached value: once
    # before locking the file, and once after locking, at which point another
    # party may have written something to the cache already.
    cachefile.parent.mkdir(parents=True, exist_ok=True)
    cachefile.touch()
    with cachefile.open('r+b') as f:
      log.debug('[cache.function {}] acquiring lock'.format(hkey))
      _lock_file(f)
      log.debug('[cache.function {}] lock acquired'.format(hkey))
      try:
        value, log_ = pickle.load(f)
      except (EOFError, pickle.UnpicklingError):
        log.debug('[cache.function {}] failed to load, cache will be rewritten'.format(hkey))
        pass
      else:
        log.debug('[cache.function {}] load'.format(hkey))
        log_.replay()
        return value
      # Seek back to the beginning, because pickle might have read garbage.
      f.seek(0)
      # Disable the cache temporarily to prevent caching subresults *in* `func`.
      with config(cache=False), log.RecordLog() as log_:
        value = func(*args, **kwargs)
      pickle.dump((value, log_), f)
      log.debug('[cache.function {}] store'.format(hkey))
      return value

  return wrapper

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
