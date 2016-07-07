
import numpy as np
from groupy.garray.matrix_garray import MatrixGArray
from groupy.garray.Z2_array import Z2Array

# A transformation in p4m can be coded using four integers:
# m in {0, 1}, mirror reflection in the second translation axis or not
# r in {0, 1, 2, 3}, the rotation index
# u, translation along the first spatial axis
# v, translation along the second spatial axis
# We will always store these in the order (m, r, u, v).
# This is called the 'int' parameterization of p4m.

# A matrix representation of this group is given by
# T(u, v) M(m) R(r)
# where
# T = [[ 1, 0, u],
#      [ 0, 1, v],
#      [ 0, 0, 1]]
# M = [[ (-1) ** m, 0, 0],
#      [ 0,         1, 0],
#      [ 0,         0, 1]]
# R = [[ cos(r pi / 2), -sin(r pi /2), 0],
#      [ sin(r pi / 2), cos(r pi / 2), 0],
#      [ 0,             0,             1]]
# This is called the 'hmat' (homogeneous matrix) parameterization of p4m.

# The matrix representation is easier to work with when multiplying and inverting group elements,
# while the integer parameterization is required when indexing gfunc on p4m.


class P4MArray(MatrixGArray):

    parameterizations = ['int', 'hmat']
    _g_shapes = {'int': (4,), 'hmat': (3, 3)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'p4m'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int
        assert (p == 'int' and data.shape[-1] == 4) or (p == 'hmat' and data.shape[-2:] == (3, 3))

        self._left_actions[P4MArray] = self.__class__.left_action_hmat
        self._left_actions[Z2Array] = self.__class__.left_action_hvec

        super(P4MArray, self).__init__(data, p)

    def int2hmat(self, int_data):
        m = int_data[..., 0]
        r = int_data[..., 1]
        u = int_data[..., 2]
        v = int_data[..., 3]
        out = np.zeros(int_data.shape[:-1] + (3, 3), dtype=np.int)
        out[..., 0, 0] = np.cos(0.5 * np.pi * r) * (-1) ** m
        out[..., 0, 1] = -np.sin(0.5 * np.pi * r) * (-1) ** m
        out[..., 0, 2] = u
        out[..., 1, 0] = np.sin(0.5 * np.pi * r)
        out[..., 1, 1] = np.cos(0.5 * np.pi * r)
        out[..., 1, 2] = v
        out[..., 2, 2] = 1.
        return out

    def hmat2int(self, hmat_data):
        neg_det_r = hmat_data[..., 1, 0] * hmat_data[..., 0, 1] - hmat_data[..., 0, 0] * hmat_data[..., 1, 1]
        s = hmat_data[..., 1, 0]
        c = hmat_data[..., 1, 1]
        u = hmat_data[..., 0, 2]
        v = hmat_data[..., 1, 2]
        m = (neg_det_r + 1) // 2
        r = ((np.arctan2(s, c) / np.pi * 2) % 4).astype(np.int)

        out = np.zeros(hmat_data.shape[:-2] + (4,), dtype=np.int)
        out[..., 0] = m
        out[..., 1] = r
        out[..., 2] = u
        out[..., 3] = v
        return out


def identity(shape=(), p='int'):
    e = P4MArray(np.zeros(shape + (4,), dtype=np.int), 'int')
    return e.reparameterize(p)


def rand(minu, maxu, minv, maxv, size=()):
    data = np.zeros(size + (4,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 2, size)
    data[..., 1] = np.random.randint(0, 4, size)
    data[..., 2] = np.random.randint(minu, maxu, size)
    data[..., 3] = np.random.randint(minv, maxv, size)
    return P4MArray(data=data, p='int')


def rotation(r, center=(0, 0)):
    r = np.asarray(r)
    center = np.asarray(center)

    rdata = np.zeros(r.shape + (4,), dtype=np.int)
    rdata[..., 1] = r
    r0 = P4MArray(rdata)

    tdata = np.zeros(center.shape[:-1] + (4,), dtype=np.int)
    tdata[..., 2:] = center
    t = P4MArray(tdata)

    return t * r0 * t.inv()


def mirror_u(shape=None):
    shape = shape if shape is not None else ()
    mdata = np.zeros(shape + (4,), dtype=np.int)
    mdata[0] = 1
    return P4MArray(mdata)


def mirror_v(shape=None):
    hm = mirror_u(shape)
    r = rotation(1)
    return r * hm * r.inv()


def m_range(start=0, stop=2):
    assert stop > 0
    assert stop <= 2
    assert start >= 0
    assert start < 2
    assert start < stop
    m = np.zeros((stop - start, 4), dtype=np.int)
    m[:, 0] = np.arange(start, stop)
    return P4MArray(m)


def r_range(start=0, stop=4, step=1):
    assert stop > 0
    assert stop <= 4
    assert start >= 0
    assert start < 4
    assert start < stop
    m = np.zeros((stop - start, 4), dtype=np.int)
    m[:, 1] = np.arange(start, stop, step)
    return P4MArray(m)


def u_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 4), dtype=np.int)
    m[:, 2] = np.arange(start, stop, step)
    return P4MArray(m)


def v_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 4), dtype=np.int)
    m[:, 3] = np.arange(start, stop, step)
    return P4MArray(m)


def meshgrid(m=m_range(), r=r_range(), u=u_range(), v=v_range()):
    m = P4MArray(m.data[:, None, None, None, ...], p=m.p)
    r = P4MArray(r.data[None, :, None, None, ...], p=r.p)
    u = P4MArray(u.data[None, None, :, None, ...], p=u.p)
    v = P4MArray(v.data[None, None, None, :, ...], p=v.p)
    return u * v * m * r


# def gmeshgrid(*args):
#    out = identity()
#    for i in range(len(args)):
#        slices = [None if j != i else slice(None) for j in range(len(args))] + [Ellipsis]
#        d = args[i].data[slices]
#        print i, slices, d.shape
#        out *= P4MArray(d, p=args[i].p)
#
#    return out
