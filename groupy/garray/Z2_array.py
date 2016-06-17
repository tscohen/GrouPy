
import numpy as np

from groupy.garray.garray import GArray


class Z2Array(GArray):

    parameterizations = ['int']
    _left_actions = {}
    _reparameterizations = {}
    _g_shapes = {'int': (2,)}
    _group_name = 'Z2'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int
        self._left_actions[Z2Array] = self.__class__.z2_composition
        super(Z2Array, self).__init__(data, p)

    def z2_composition(self, other):
        return Z2Array(self.data + other.data)

    def inv(self):
        return Z2Array(-self.data)

    def __repr__(self):
        return 'Z2\n' + self.data.__repr__()

    def reparameterize(self, p):
        assert p == 'int'
        return self


def identity(shape=()):
    e = Z2Array(np.zeros(shape + (2,), dtype=np.int), 'int')
    return e


def rand(minu, maxu, minv, maxv, size=()):
    data = np.zeros(size + (2,), dtype=np.int64)
    data[..., 0] = np.random.randint(minu, maxu, size)
    data[..., 1] = np.random.randint(minv, maxv, size)
    return Z2Array(data=data, p='int')


def u_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 2), dtype=np.int)
    m[:, 0] = np.arange(start, stop, step)
    return Z2Array(m)


def v_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 2), dtype=np.int)
    m[:, 1] = np.arange(start, stop, step)
    return Z2Array(m)


def meshgrid(u=u_range(), v=v_range()):
    u = Z2Array(u.data[:, None, ...], p=u.p)
    v = Z2Array(v.data[None, :, ...], p=v.p)
    return u * v


# def gmeshgrid(*args):
#    out = identity()
#    for i in range(len(args)):
#        slices = [None if j != i else slice(None) for j in range(len(args))] + [Ellipsis]
#        d = args[i].data[slices]
#        print i, slices, d.shape
#        out *= P4MArray(d, p=args[i].p)
#
#    return out
