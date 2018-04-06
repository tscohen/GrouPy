import numpy as np

from groupy.garray.garray import GArray


class Z3Array(GArray):
    parameterizations = ['int']
    _left_actions = {}
    _reparameterizations = {}
    _g_shapes = {'int': (3,)}
    _group_name = 'Z3'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int
        self._left_actions[Z3Array] = self.__class__.z3_composition
        super(Z3Array, self).__init__(data, p)

    def z3_composition(self, other):
        return Z3Array(self.data + other.data)

    def inv(self):
        return Z3Array(-self.data)

    def __repr__(self):
        return 'Z3\n' + self.data.__repr__()

    def reparameterize(self, p):
        assert p == 'int'
        return self


def identity(shape=()):
    '''
    Returns the identity element: an array of 3 zeros.
    '''
    e = Z3Array(np.zeros(shape + (3,), dtype=np.int), 'int')
    return e


def rand(minu, maxu, minv, maxv, minw, maxw, size=()):
    '''
    Returns an Z3Array of shape size, with randomly chosen elements in int parameterization.
    '''
    data = np.zeros(size + (3,), dtype=np.int64)
    data[..., 0] = np.random.randint(minu, maxu, size)
    data[..., 1] = np.random.randint(minv, maxv, size)
    data[..., 2] = np.random.randint(minw, maxw, size)
    return Z3Array(data=data, p='int')


def meshgrid(minu=-1, maxu=2, minv=-1, maxv=2, minw=-1, maxw=2):
    '''
    Creates a meshgrid of all elements of the group, within the given
    translation parameters.
    '''
    li = [[u, v, w] for u in xrange(minu, maxu) for v in xrange(minv, maxv) for
          w in xrange(minw, maxw)]
    return Z3Array(li, p='int')
