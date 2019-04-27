import numpy as np
from groupy.garray.matrix_garray import MatrixGArray

""" 
Implementation of the non-orientation perserving variant of group C4h -- C4h with translations. 
The int parameterisation is similar to that of C4h, but with the added 3D translation (u, v, w) to indicate
translation in Z3. 

4x4 homogeneous matrices (hmat) are used to represent the transformation in matrix format. 
"""

class C4htArray(MatrixGArray):
    parameterizations = ['int', 'hmat']
    _g_shapes = {'int': (5,), 'hmat': (4, 4)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'C4ht'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int
        self._left_actions[C4htArray] = self.__class__.left_action_hmat
        super(C4htArray, self).__init__(data, p)
        self.elements = self.get_elements()

    def hmat2int(self, hmat_data):
        '''
        Transforms 4x4 matrix representation to int representation.
        To handle any size and shape of hmat_data, the original hmat_data
        is reshaped to a long list of 4x4 matrices, converted to a list of
        int representations, and reshaped back to the original mat_data shape.

        hmat-2-int is achieved by taking the matrix, looking up the index in the
        element list, and converting that index to two numbers: y and z. The index
        is the result of (y * 4) + z. u, v, w are retrieved by looking at the last
        column of the hmat.
        '''

        input = hmat_data.reshape((-1, 4, 4))
        data = np.zeros((input.shape[0], 5), dtype=np.int)
        for i in range(input.shape[0]):
            hmat = input[i]
            mat = [elem[0:3] for elem in hmat.tolist()][0:3]
            index = self.elements.index(mat)
            z = int(index % 4)
            y = int((index - z) / 4)
            u, v, w, _ = hmat[:, 3]
            data[i, 0] = y
            data[i, 1] = z
            data[i, 2] = u
            data[i, 3] = v
            data[i, 4] = w
        data = data.reshape(hmat_data.shape[:-2] + (5,))
        return data

    def int2hmat(self, int_data):
        '''
        Transforms integer representation to 4x4 matrix representation.
        Original int_data is flattened and later reshaped back to its original
        shape to handle any size and shape of input.
        '''
        # rotations over y and z
        y = int_data[..., 0].flatten()
        z = int_data[..., 1].flatten()

        # translations
        u = int_data[..., 2].flatten()
        v = int_data[..., 3].flatten()
        w = int_data[..., 4].flatten()
        data = np.zeros((len(y),) + (4, 4), dtype=np.int)

        for j in range(len(y)):
            index = (y[j] * 4) + z[j]
            mat = self.elements[index]

            data[j, 0:3, 0:3] = mat
            data[j, 0, 3] = u[j]
            data[j, 1, 3] = v[j]
            data[j, 2, 3] = w[j]
            data[j, 3, 3] = 1

        data = data.reshape(int_data.shape[:-1] + (4, 4))
        return data

    def _multiply(self, element, generator, times):
        '''
        Helper function to multiply an _element_ with a _generator_
        _times_ number of times. Used in self.get_elements()
        '''
        element = np.array(element)
        for i in range(times):
            element = np.dot(element, np.array(generator))
        return element

    def get_elements(self):
        '''
        Function to generate a list containing elements of group C4ht,
        similar to get_elements() of BArray.

        These are the base elements in 3x3 matrix notation without translations.
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


def rand(minu=0, maxu=5, minv=0, maxv=5, minw=0, maxw=5, size=()):
    '''
    Returns an C4htArray of shape size, with randomly chosen elements in int parameterization.
    '''
    data = np.zeros(size + (5,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 2, size)    # rotations over y
    data[..., 1] = np.random.randint(0, 4, size)    # rotations over x
    data[..., 2] = np.random.randint(minu, maxu, size)  # translation on x
    data[..., 3] = np.random.randint(minv, maxv, size)  # translation on y
    data[..., 4] = np.random.randint(minw, maxw, size)  # translation on z
    return C4htArray(data=data, p='int')


def identity(p='int'):
    '''
    Returns the identity element: a matrix with 1's on the diagonal.
    '''
    li = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    e = C4htArray(data=np.array(li, dtype=np.int), p='hmat')
    return e.reparameterize(p)


def meshgrid(minu=-1, maxu=2, minv=-1, maxv=2, minw=-1, maxw=2):
    '''
    Creates a meshgrid of all elements of the group, within the given
    translation parameters.
    '''
    li = [[i, m, u, v, w] for i in range(2) for m in range(4) for u in range(minu, maxu) for v in range(minv, maxv)
          for
          w in range(minw, maxw)]
    return C4htArray(li, p='int')
