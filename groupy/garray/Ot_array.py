import random

import numpy as np
from groupy.garray.matrix_garray import MatrixGArray

'''
Implementation of the space group O that allows translations.
It has no official name, and is therefore now referred to as Ot.

Implementation is similar to that of group O. However, to represent
the translations in a 3D space, the int parameterization is now
in the form of (i, u, v, w) representing the index in the element list,
and the translation in the x, y and z direction respectively.

To accurately represent the translation, we use 4x4 homogeneous matrices
(hmat) instead of the 3x3 matrix.

Note: self.base_elements are 3x3 matrices.
'''


class OtArray(MatrixGArray):
    '''
    Implementation of space group Ot.
    '''
    parameterizations = ['int', 'hmat']
    _g_shapes = {'int': (4,), 'hmat': (4, 4)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'Ot'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int
        self._left_actions[OtArray] = self.__class__.left_action_hmat
        super(OtArray, self).__init__(data, p)
        self.base_elements = self.get_base_elements()

    def hmat2int(self, hmat_data):
        '''
        Transforms 4x4 matrix representation to int representation.
        To handle any size and shape of hmat_data, the original hmat_data
        is reshaped to a long list of 4x4 matrices, converted to a list of
        int representations, and reshaped back to the original mat_data shape.
        '''

        input = hmat_data.reshape((-1, 4, 4))
        data = np.zeros((input.shape[0], 4), dtype=np.int)
        for i in xrange(input.shape[0]):
            hmat = input[i]
            mat = [elem[0:3] for elem in hmat.tolist()][0:3]
            index = self.base_elements.index(mat)
            u, v, w, _ = hmat[:, 3]
            data[i, 0] = index
            data[i, 1] = u
            data[i, 2] = v
            data[i, 3] = w
        data = data.reshape(hmat_data.shape[:-2] + (4,))
        return data

    def int2hmat(self, int_data):
        '''
        Transforms integer representation to 3x3 matrix representation.
        Original int_data is flattened and later reshaped back to its original
        shape to handle any size and shape of input.
        '''
        i = int_data[..., 0].flatten()
        u = int_data[..., 1].flatten()
        v = int_data[..., 2].flatten()
        w = int_data[..., 3].flatten()
        data = np.zeros((len(i),) + (4, 4), dtype=np.int)

        for j in xrange(len(i)):
            mat = self.base_elements[i[j]]
            data[j, 0:3, 0:3] = mat
            data[j, 0, 3] = u[j]
            data[j, 1, 3] = v[j]
            data[j, 2, 3] = w[j]
            data[j, 3, 3] = 1

        data = data.reshape(int_data.shape[:-1] + (4, 4))
        return data

    def get_base_elements(self):
        '''
        Function to generate a list containing all 24 elements of
        group O. This list is necessary for the integer representation:
        the index of an element in this list represents the group element.

        Elements are stored as lists rather than numpy arrays to enable
        lookup through self.elements.index(x) and sorting.

        Basic principle to find all elements in the list:
            specify the generators (rotations over x and y axis)
            while not all elements have been found, repeat:
                choose random element from current list of elements as multiplier
                multiply this element with the last found element
                if this element is new, add it to the list of elements.

        These are the base elements in 3x3 matrix notation without translations.
        '''
        g1 = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # 90o degree rotation over x
        g2 = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # 90o degree rotation over y
        element_list = [g1, g2]
        current = g1
        while len(element_list) < 24:
            multiplier = random.choice(element_list)
            current = np.dot(np.array(current), np.array(multiplier)).tolist()
            if current not in element_list:
                element_list.append(current)
        element_list.sort()
        return element_list


def identity(shape=(), p='int'):
    '''
    Returns the identity element: a matrix with 1's on the diagonal.
    '''
    li = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    e = OtArray(data=np.array(li, dtype=np.int), p='hmat')
    return e.reparameterize(p)


def rand(minu, maxu, minv, maxv, minw, maxw, size=()):
    '''
    Returns an OtArray of shape size, with randomly chosen elements in int parameterization.
    '''
    data = np.zeros(size + (4,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 24, size)
    data[..., 1] = np.random.randint(minu, maxu, size)
    data[..., 2] = np.random.randint(minv, maxv, size)
    data[..., 3] = np.random.randint(minw, maxw, size)
    return OtArray(data=data, p='int')


def meshgrid(minu=-1, maxu=2, minv=-1, maxv=2, minw=-1, maxw=2):
    li = [[i, u, v, w] for i in xrange(24) for u in xrange(minu, maxu) for v in xrange(minv, maxv) for
          w in xrange(minw, maxw)]
    return OtArray(li, p='int')
