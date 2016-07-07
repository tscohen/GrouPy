
import numpy as np

from groupy.garray.garray import GArray


class MatrixGArray(GArray):
    """
    Base class for matrix group GArrays.
    Composition, inversion and the action on vectors is implemented as
    matrix multiplication, matrix inversion and matrix-vector multiplication, respectively.
    """

    def __init__(self, data, p='int'):
        data = np.asarray(data)

        if p == 'int' and data.dtype != np.int:
            raise ValueError('data.dtype must be int when integer parameterization is used.')

        if 'mat' not in self.parameterizations and 'hmat' not in self.parameterizations:
            raise AssertionError('Subclasses of MatrixGArray should always have a "mat" and/or "hmat" parameterization')

        if 'mat' in self.parameterizations:
            self._reparameterizations[('int', 'mat')] = self.int2mat
            self._reparameterizations[('mat', 'int')] = self.mat2int

        if 'hmat' in self.parameterizations:
            self._reparameterizations[('int', 'hmat')] = self.int2hmat
            self._reparameterizations[('hmat', 'int')] = self.hmat2int

        if 'mat' in self.parameterizations and 'hmat' in self.parameterizations:
            self._reparameterizations[('hmat', 'mat')] = self.hmat2mat
            self._reparameterizations[('mat', 'hmat')] = self.mat2hmat

        super(MatrixGArray, self).__init__(data, p)

    def inv(self):
        mat_p = 'mat' if 'mat' in self.parameterizations else 'hmat'
        self_mat = self.reparameterize(mat_p).data
        self_mat_inv = np.linalg.inv(self_mat)
        self_mat_inv = np.round(self_mat_inv, 0).astype(self_mat.dtype)
        return self.factory(data=self_mat_inv, p=mat_p).reparameterize(self.p)

    def left_action_mat(self, other):
        self_mat = self.reparameterize('mat').data
        other_mat = other.reparameterize('mat').data
        c_mat = np.einsum('...ij,...jk->...ik', self_mat, other_mat)
        return other.factory(data=c_mat, p='mat').reparameterize(other.p)

    def left_action_hmat(self, other):
        self_hmat = self.reparameterize('hmat').data
        other_hmat = other.reparameterize('hmat').data
        c_hmat = np.einsum('...ij,...jk->...ik', self_hmat, other_hmat)
        return other.factory(data=c_hmat, p='hmat').reparameterize(other.p)

    def left_action_vec(self, other):
        self_mat = self.reparameterize('mat').data
        assert other.p == 'int'  # TODO
        out = np.einsum('...ij,...j->...i', self_mat, other.data)
        return other.factory(data=out, p=other.p)

    def left_action_hvec(self, other):
        self_hmat = self.reparameterize('hmat').data
        assert other.p == 'int'  # TODO
        self_mat = self_hmat[..., :-1, :-1]
        out = np.einsum('...ij,...j->...i', self_mat, other.data) + self_hmat[..., :-1, -1]
        return other.factory(data=out, p=other.p)

    def int2mat(self, int_data):
        raise NotImplementedError()

    def mat2int(self, mat_data):
        raise NotImplementedError()

    def mat2hmat(self, mat_data):
        n, m = self._g_shapes['mat']
        out = np.zeros(mat_data.shape[:-2] + (n + 1, m + 1), dtype=mat_data.dtype)
        out[..., :n, :m] = mat_data
        return out

    def hmat2mat(self, hmat_data):
        return hmat_data[..., :-1, :-1]

    def int2hmat(self, int_data):
        return self.mat2hmat(self.int2mat(int_data))

    def hmat2int(self, hmat_data):
        return self.mat2int(self.hmat2mat(hmat_data))
