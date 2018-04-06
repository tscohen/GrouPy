import random
import numpy as np
from groupy.garray.finitegroup import FiniteGroup
from groupy.garray.matrix_garray import MatrixGArray
from groupy.garray.C4ht_array import C4htArray
from groupy.garray.Z3_array import Z3Array

'''
Implementation of finite group C4h, the group of beam symmetry. No official name is known as of yet, 
but the group exists of 180 degree rotations over the y-axis and 90 degree rotations over the z-axis -- 8 elements
in total. 

Int parameterization is in the form of (y, z) where y represents the number of 180 degree rotations over the y axis
(0, 1) and z represents the number of 180 degree rotations over the z axis (0, 1, 2, 3).
'''
class C4hArray(MatrixGArray):
    parameterizations = ['int', 'mat', 'hmat']
    _g_shapes = {'int': (2,), 'mat': (3, 3), 'hmat': (4, 4)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'C4h'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int

        # classes C4hArray can be multiplied with
        self._left_actions[C4hArray] = self.__class__.left_action_hmat
        self._left_actions[C4htArray] = self.__class__.left_action_hmat
        self._left_actions[Z3Array] = self.__class__.left_action_vec

        super(C4hArray, self).__init__(data, p)
        self.elements = self.get_elements()

    def mat2int(self, mat_data):
        '''
        Transforms 3x3 matrix representation to int representation.
        To handle any size and shape of mat_data, the original mat_data
        is reshaped to a long list of 3x3 matrices, converted to a list of
        int representations, and reshaped back to the original mat_data shape.

        mat-2-int is achieved by taking the matrix, looking up the index in the
        element list, and converting that index to two numbers: y and z. The index
        is the result of (y * 4) + z.
        '''

        input = mat_data.reshape((-1, 3, 3))
        data = np.zeros((input.shape[0], 2), dtype=np.int)
        for i in xrange(input.shape[0]):
            index = self.elements.index(input[i].tolist())
            z = int(index % 4)
            y = int((index - z) / 4)
            data[i, 0] = y
            data[i, 1] = z
        data = data.reshape(mat_data.shape[:-2] + (2,))
        return data

    def int2mat(self, int_data):
        '''
        Transforms integer representation to 3x3 matrix representation.
        Original int_data is flattened and later reshaped back to its original
        shape to handle any size and shape of input.
        '''
        y = int_data[..., 0].flatten()
        z = int_data[..., 1].flatten()
        data = np.zeros((len(y),) + (3, 3), dtype=np.int)

        for j in xrange(len(y)):
            index = (y[j] * 4) + z[j]
            mat = self.elements[index]
            data[j, 0:3, 0:3] = mat

        data = data.reshape(int_data.shape[:-1] + (3, 3))
        return data

    def _multiply(self, element, generator, times):
        element = np.array(element)
        for i in range(times):
            element = np.dot(element, np.array(generator))
        return element

    def get_elements(self):
        '''
        Function to generate a list containing  elements of group C4hrt,
        similar to get_elements() of C4hArray.

        Elements are stored as lists rather than numpy arrays to enable
        lookup through self.elements.index(x) and sorting.

        All elements are found by multiplying the identity matrix with all
        possible combinations of the generators, i.e. 0 or 1 rotations over y
        and 0, 1, 2, or 3 rotations over z.
        '''
        # specify generators
        mode = 'zyx'
        if mode == 'xyz':
            g1 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])  # 180 degrees over y
            g2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90 degrees over z
        elif mode == 'zyx':
            g1 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])  # 180 degrees over y
            g2 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # 90 degrees over z


        element_list = []
        element = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for i in range(0, 2):
            element = self._multiply(element, g1, i)
            for j in range(0, 4):
                element = self._multiply(element, g2, j)
                element_list.append(element.tolist())
        return element_list


class C4hGroup(FiniteGroup, C4hArray):
    def __init__(self):
        C4hArray.__init__(
            self,
            data=np.array([[i, j] for i in xrange(2) for j in xrange(4)]),
            p='int'
        )
        FiniteGroup.__init__(self, C4hArray)

    def factory(self, *args, **kwargs):
        return C4hArray(*args, **kwargs)

C4h = C4hGroup()

def rand(size=()):
    '''
    Returns an C4hArray of shape size, with randomly chosen elements in int parameterization.
    '''
    data = np.zeros(size + (2,), dtype=np.int)
    data[..., 0] = np.random.randint(0, 2, size)
    data[..., 1] = np.random.randint(0, 4, size)
    return C4hArray(data=data, p='int')

def identity(p='int'):
    '''
    Returns the identity element: a matrix with 1's on the diagonal.
    '''
    li = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    e = C4hArray(data=np.array(li, dtype=np.int), p='mat')
    return e.reparameterize(p)
