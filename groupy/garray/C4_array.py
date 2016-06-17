import numpy as np
from groupy.garray.matrix_garray import MatrixGArray
from groupy.garray.finitegroup import FiniteGroup
from groupy.garray.p4_array import P4Array
from groupy.garray.Z2_array import Z2Array


class C4Array(MatrixGArray):

    parameterizations = ['int', 'mat', 'hmat']
    _g_shapes = {'int': (1,), 'mat': (2, 2), 'hmat': (3, 3)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'C4'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int

        self._left_actions[C4Array] = self.__class__.left_action_mat
        self._left_actions[P4Array] = self.__class__.left_action_hmat
        self._left_actions[Z2Array] = self.__class__.left_action_vec

        super(C4Array, self).__init__(data, p)

    def int2mat(self, int_data):
        r = int_data[..., 0]
        out = np.zeros(int_data.shape[:-1] + (2, 2), dtype=np.int)
        out[..., 0, 0] = np.cos(0.5 * np.pi * r)
        out[..., 0, 1] = -np.sin(0.5 * np.pi * r)
        out[..., 1, 0] = np.sin(0.5 * np.pi * r)
        out[..., 1, 1] = np.cos(0.5 * np.pi * r)
        return out

    def mat2int(self, mat_data):
        s = mat_data[..., 1, 0]
        c = mat_data[..., 1, 1]
        r = ((np.arctan2(s, c) / np.pi * 2) % 4).astype(np.int)
        out = np.zeros(mat_data.shape[:-2] + (1,), dtype=np.int)
        out[..., 0] = r
        return out


class C4Group(FiniteGroup, C4Array):

    def __init__(self):
        C4Array.__init__(
            self,
            data=np.arange(4)[:, None],
            p='int'
        )
        FiniteGroup.__init__(self, C4Array)

    def factory(self, *args, **kwargs):
        return C4Array(*args, **kwargs)


C4 = C4Group()

# Generators & special elements
r = C4Array(data=np.array([1]), p='int')
e = C4Array(data=np.array([0]), p='int')


def identity(shape=(), p='int'):
    e = C4Array(np.zeros(shape + (1,), dtype=np.int), 'int')
    return e.reparameterize(p)


def rand(size=()):
    data = np.zeros(size + (1,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 4, size)
    return C4Array(data=data, p='int')
