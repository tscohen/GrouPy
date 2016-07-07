
import numpy as np
from groupy.garray.matrix_garray import MatrixGArray
from groupy.garray.Z2_array import Z2Array

# A transformation in p4 can be coded using three integers:
# r in {0, 1, 2, 3}, the rotation index
# u, translation along the first spatial axis
# v, translation along the second spatial axis
# We will always store these in the order (r, u, v).
# This is called the 'int' parameterization of p4.

# A matrix representation of this group is given by
# T(u, v) R(r)
# where
# T = [[ 1, 0, u],
#      [ 0, 1, v],
#      [ 0, 0, 1]]
# R = [[ cos(r pi / 2), -sin(r pi /2), 0],
#      [ sin(r pi / 2), cos(r pi / 2), 0],
#      [ 0,             0,             1]]
# This is called the 'hmat' (homogeneous matrix) parameterization of p4.

# The matrix representation is easier to work with when multiplying and inverting group elements,
# while the integer parameterization is required when indexing gfunc on p4.


class P4Array(MatrixGArray):

    parameterizations = ['int', 'hmat']
    _g_shapes = {'int': (3,), 'hmat': (3, 3)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'p4'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int
        self._left_actions[P4Array] = self.__class__.left_action_hmat
        self._left_actions[Z2Array] = self.__class__.left_action_hvec
        super(P4Array, self).__init__(data, p)

    def int2hmat(self, int_data):
        r = int_data[..., 0]
        u = int_data[..., 1]
        v = int_data[..., 2]
        out = np.zeros(int_data.shape[:-1] + (3, 3), dtype=np.int)
        out[..., 0, 0] = np.cos(0.5 * np.pi * r)
        out[..., 0, 1] = -np.sin(0.5 * np.pi * r)
        out[..., 0, 2] = u
        out[..., 1, 0] = np.sin(0.5 * np.pi * r)
        out[..., 1, 1] = np.cos(0.5 * np.pi * r)
        out[..., 1, 2] = v
        out[..., 2, 2] = 1.
        return out

    def hmat2int(self, mat_data):
        s = mat_data[..., 1, 0]
        c = mat_data[..., 1, 1]
        u = mat_data[..., 0, 2]
        v = mat_data[..., 1, 2]
        r = ((np.arctan2(s, c) / np.pi * 2) % 4).astype(np.int)

        out = np.zeros(mat_data.shape[:-2] + (3,), dtype=np.int)
        out[..., 0] = r
        out[..., 1] = u
        out[..., 2] = v
        return out


# Generators
r = P4Array(data=np.array([1, 0, 0]), p='int')
u = P4Array(data=np.array([0, 1, 0]), p='int')
v = P4Array(data=np.array([0, 0, 1]), p='int')


def identity(shape=(), p='int'):
    e = P4Array(np.zeros(shape + (3,), dtype=np.int), 'int')
    return e.reparameterize(p)


def rand(minu, maxu, minv, maxv, size=()):
    data = np.zeros(size + (3,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 4, size)
    data[..., 1] = np.random.randint(minu, maxu, size)
    data[..., 2] = np.random.randint(minv, maxv, size)
    return P4Array(data=data, p='int')


def rotation(r, center=(0, 0)):
    r = np.asarray(r)
    center = np.asarray(center)

    rdata = np.zeros(r.shape + (3,), dtype=np.int)
    rdata[..., 0] = r
    r0 = P4Array(rdata)

    tdata = np.zeros(center.shape[:-1] + (3,), dtype=np.int)
    tdata[..., 1:] = center
    t = P4Array(tdata)

    return t * r0 * t.inv()


def translation(t):
    t = np.asarray(t)
    tdata = np.zeros(t.shape[:-1] + (3,), dtype=np.int)
    tdata[..., 1:] = t
    return P4Array(tdata)


def r_range(start=0, stop=4, step=1):
    assert stop > 0
    assert stop <= 4
    assert start >= 0
    assert start < 4
    assert start < stop
    m = np.zeros((stop - start, 3), dtype=np.int)
    m[:, 0] = np.arange(start, stop, step)
    return P4Array(m)


def u_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 3), dtype=np.int)
    m[:, 1] = np.arange(start, stop, step)
    return P4Array(m)


def v_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 3), dtype=np.int)
    m[:, 2] = np.arange(start, stop, step)
    return P4Array(m)


def meshgrid(r=r_range(), u=u_range(), v=v_range()):
    r = P4Array(r.data[:, None, None, ...], p=r.p)
    u = P4Array(u.data[None, :, None, ...], p=u.p)
    v = P4Array(v.data[None, None, :, ...], p=v.p)
    return u * v * r


# When rotating even-sized filters, rotating around the origin would not map the filter onto itself.
# For example, take a 2x2 filter
# [[a, b],
#  [c, d]]
# To rotate this filter, we want to rotate about its center, which is not a point in the grid Z^2.
# The following subgroup contains all 4 rotations around the point (-0.5, -0.5), which we can take as the center of
# the filter.
# C4_halfshift = P4Array(data=np.array([[0, 0, 0],
#                                      [1, 1, 0],
#                                      [2, 1, 1],
#                                      [3, 0, 1]]), p='int')
C4_halfshift = P4Array(data=np.array([[0, 0, 0],
                                      [1, -1, 0],
                                      [2, -1, -1],
                                      [3, 0, -1]]), p='int')

# def gmeshgrid(*args):
#    out = identity()
#    for i in range(len(args)):
#        slices = [None if j != i else slice(None) for j in range(len(args))] + [Ellipsis]
#        d = args[i].data[slices]
#        print i, slices, d.shape
#        out *= P4MArray(d, p=args[i].p)
#
#    return out
