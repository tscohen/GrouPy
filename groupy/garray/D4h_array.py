import numpy as np
from groupy.garray.finitegroup import FiniteGroup
from groupy.garray.matrix_garray import MatrixGArray
from groupy.garray.D4ht_array import D4htArray
from groupy.garray.Z3_array import Z3Array

'''
Implementation of finite group D4h, the group of beam symmetry with reflections. No official name is known as of yet, 
but the group exists of 180 degree rotations over the y-axis and 90 degree rotations over the z-axis,
combined with reflections -- 16 elements in total. 

Int parameterization is in the form of (y, z, m) where y represents the number of 180 degree rotations over the y axis
(0, 1), z represents the number of 180 degree rotations over the z axis (0, 1, 2, 3) and m (for mirror) represents the
reflection (0, 1).
'''

class D4hArray(MatrixGArray):
    parameterizations = ['int', 'mat', 'hmat']
    _g_shapes = {'int': (3,), 'mat': (3, 3), 'hmat': (4, 4)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'D4h'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int

        # classes OArray can be multiplied with
        self._left_actions[D4hArray] = self.__class__.left_action_hmat
        self._left_actions[D4htArray] = self.__class__.left_action_hmat
        self._left_actions[Z3Array] = self.__class__.left_action_vec

        super(D4hArray, self).__init__(data, p)
        self.elements = self.get_elements()


    def mat2int(self, mat_data):
        '''
        Transforms 3x3 matrix representation to int representation.
        To handle any size and shape of mat_data, the original mat_data
        is reshaped to a long list of 3x3 matrices, converted to a list of
        int representations, and reshaped back to the original mat_data shape.

        mat-2-int is achieved by taking the matrix, and looking up whether it
        exists in the element list. If not, the matrix should be multiplied with -1
        to retrieve the reflection. The resulting matrix can be looked up in the
        element list, and that index can be converted to y and z.
        '''

        input = mat_data.reshape((-1, 3, 3))
        data = np.zeros((input.shape[0], 3), dtype=np.int)
        for i in xrange(input.shape[0]):
            mat = input[i]
            # check for reflection
            if mat.tolist() not in self.elements:
                mat = np.array(mat) * -1
                data[i, 2] = 1

            # determine z and y
            index = self.elements.index(mat.tolist())
            z = int(index % 4)
            y = int((index - z) / 4)
            data[i, 0] = y
            data[i, 1] = z
        data = data.reshape(mat_data.shape[:-2] + (3,))
        return data

    def int2mat(self, int_data):
        '''
        Transforms integer representation to 3x3 matrix representation.
        Original int_data is flattened and later reshaped back to its original
        shape to handle any size and shape of input.
        '''
        # rotations over y, z and reflection
        y = int_data[..., 0].flatten()
        z = int_data[..., 1].flatten()
        m = int_data[..., 2].flatten()
        data = np.zeros((len(y),) + (3, 3), dtype=np.int)

        for j in xrange(len(y)):
            index = (y[j] * 4) + z[j]
            mat = self.elements[index]
            mat = np.array(mat) * ((-1) ** m[j])    # mirror if reflection
            data[j, 0:3, 0:3] = mat.tolist()

        data = data.reshape(int_data.shape[:-1] + (3, 3))
        return data

    def _multiply(self, element, generator, times):
        '''
        Helper function to multiply an _element_ with a _generator_
        _times_ number of times.
        '''
        element = np.array(element)
        for i in range(times):
            element = np.dot(element, np.array(generator))
        return element

    def get_elements(self):
        '''
        Function to generate a list containing  elements of group D4h,
        similar to get_elements() of BArray.

        Elements are stored as lists rather than numpy arrays to enable
        lookup through self.elements.index(x).
        '''
        # specify generators
        g1 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])  # 180 degrees over y
        g2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90 degrees over z

        element_list = []
        element = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])   # starting point = identity matrix
        for i in range(0, 2):
            element = self._multiply(element, g1, i)
            for j in range(0, 4):
                element = self._multiply(element, g2, j)
                element_list.append(element.tolist())
        return element_list


class D4hGroup(FiniteGroup, D4hArray):
    def __init__(self):
        D4hArray.__init__(
            self,
            data=np.array([[i, j, m] for i in xrange(2) for j in xrange(4) for m in xrange(2)]),
            p='int'
        )
        FiniteGroup.__init__(self, D4hArray)

    def factory(self, *args, **kwargs):
        return D4hArray(*args, **kwargs)

D4h = D4hGroup()

def rand(size=()):
    '''
    Returns an D4hArray of shape size, with randomly chosen elements in int parameterization.
    '''
    data = np.zeros(size + (3,), dtype=np.int)
    data[..., 0] = np.random.randint(0, 2, size)
    data[..., 1] = np.random.randint(0, 4, size)
    data[..., 2] = np.random.randint(0, 2, size)
    return D4hArray(data=data, p='int')

def identity(p='int'):
    '''
    Returns the identity element: a matrix with 1's on the diagonal.
    '''
    li = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    e = D4hArray(data=np.array(li, dtype=np.int), p='mat')
    return e.reparameterize(p)
