import numpy as np
from groupy.garray.garray import GArray
from groupy.garray.finitegroup import FiniteGroup
from groupy.garray.p4m_array import P4MArray
from groupy.garray.Z2_array import Z2Array

from groupy.garray.matrix_garray import MatrixGArray


class D4Array(MatrixGArray):

    parameterizations = ['int', 'mat', 'hmat']
    _g_shapes = {'int': (2,), 'mat': (2, 2), 'hmat': (3, 3)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'D4'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int

        self._left_actions[D4Array] = self.__class__.left_action_mat
        self._left_actions[P4MArray] = self.__class__.left_action_hmat
        self._left_actions[Z2Array] = self.__class__.left_action_vec

        super(D4Array, self).__init__(data, p)

    def int2mat(self, int_data):
        m = int_data[..., 0]
        r = int_data[..., 1]
        out = np.zeros(int_data.shape[:-1] + self._g_shapes['mat'], dtype=np.int)
        out[..., 0, 0] = np.cos(0.5 * np.pi * r) * (-1) ** m
        out[..., 0, 1] = -np.sin(0.5 * np.pi * r) * (-1) ** m
        out[..., 1, 0] = np.sin(0.5 * np.pi * r)
        out[..., 1, 1] = np.cos(0.5 * np.pi * r)
        return out

    def mat2int(self, mat_data):
        neg_det_r = mat_data[..., 1, 0] * mat_data[..., 0, 1] - mat_data[..., 0, 0] * mat_data[..., 1, 1]
        s = mat_data[..., 1, 0]
        c = mat_data[..., 1, 1]
        m = (neg_det_r + 1) // 2
        r = ((np.arctan2(s, c) / np.pi * 2) % 4).astype(np.int)

        out = np.zeros(mat_data.shape[:-2] + self._g_shapes['int'], dtype=np.int)
        out[..., 0] = m
        out[..., 1] = r
        return out


class D4Group(FiniteGroup, D4Array):

    def __init__(self):
        D4Array.__init__(
            self,
            data=np.array([[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3]]),
            p='int'
        )
        FiniteGroup.__init__(self, D4Array)

    def factory(self, *args, **kwargs):
        return D4Array(*args, **kwargs)


D4 = D4Group()

# Generators & special elements
r = D4Array(data=np.array([0, 1]), p='int')
m = D4Array(data=np.array([1, 0]), p='int')
e = D4Array(data=np.array([0, 0]), p='int')


def identity(shape=(), p='int'):
    e = D4Array(np.zeros(shape + (2,), dtype=np.int), 'int')
    return e.reparameterize(p)


def rand(size=()):
    data = np.zeros(size + (2,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 2, size)
    data[..., 1] = np.random.randint(0, 4, size)
    return D4Array(data=data, p='int')
