import numpy as np
from groupy.garray.matrix_garray import MatrixGArray

'''
Implementation of space group D4h that allows translations. It has no official name, and is therefor referred to as D4ht.
Implementation is similar to that of group D4h. However, to represent the translations in 3D space, 
the int parameterization is now in the form of (y, z, m, u, v, w)

Implementation of the space group Oh that allows translations.
It has no official name, and is therefore now referred to as Oht.

Implementation is similar to that of group Oh. However, to represent
the translations in a 3D space, the int parameterization is now
in the form of (i, m, u, v, w) representing the index of the unmirrored
element in the element list, the mirror (1 or -1)  and the translation
in the x, y and z direction respectively.

To accurately represent the translation, we use 4x4 homogeneous matrices
(hmat) instead of the 3x3 matrix.

Note: self.base_elements are 3x3 matrices.
'''


class D4htArray(MatrixGArray):
    parameterizations = ['int', 'hmat']
    _g_shapes = {'int': (6,), 'hmat': (4, 4)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'D4ht'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int
        self._left_actions[D4htArray] = self.__class__.left_action_hmat
        super(D4htArray, self).__init__(data, p)
        self.elements = self.get_elements()

    def hmat2int(self, hmat_data):
        '''
        Transforms 4x4 matrix representation to int representation.
        To handle any size and shape of hmat_data, the original hmat_data
        is reshaped to a long list of 4x4 matrices, converted to a list of
        int representations, and reshaped back to the original mat_data shape.


        hmat-2-int is achieved by taking the matrix, and looking up whether it
        exists in the element list. If not, the matrix should be multiplied with -1
        to retrieve the reflection. The resulting matrix can be looked up in the
        element list, and that index can be converted to y and z. u, v, and w
        are retrieved by looking at the last column in the matrix.
        '''

        input = hmat_data.reshape((-1, 4, 4))
        data = np.zeros((input.shape[0], 6), dtype=np.int)
        for i in xrange(input.shape[0]):
            hmat = input[i]
            mat = [elem[0:3] for elem in hmat.tolist()][0:3]
            # check for reflection
            if mat not in self.elements:
                mat = (np.array(mat) * -1).tolist()
                data[i, 2] = 1  # reflection

            # retrieve values
            index = self.elements.index(mat)
            z = int(index % 4)
            y = int((index - z) / 4)
            u, v, w, _ = hmat[:, 3]

            # rotations over y and z
            data[i, 0] = y
            data[i, 1] = z

            # translation
            data[i, 3] = u
            data[i, 4] = v
            data[i, 5] = w

        data = data.reshape(hmat_data.shape[:-2] + (6,))
        return data

    def int2hmat(self, int_data):
        '''
        Transforms integer representation to 3x3 matrix representation.
        Original int_data is flattened and later reshaped back to its original
        shape to handle any size and shape of input.
        '''
        # rotatiins over y, z and reflection
        y = int_data[..., 0].flatten()
        z = int_data[..., 1].flatten()
        m = int_data[..., 2].flatten()

        # translations
        u = int_data[..., 3].flatten()
        v = int_data[..., 4].flatten()
        w = int_data[..., 5].flatten()

        data = np.zeros((len(y),) + (4, 4), dtype=np.int)

        for j in xrange(len(y)):
            index = (y[j] * 4) + z[j]
            mat = self.elements[index]
            mat = np.array(mat) * ((-1) ** m[j])  # mirror if reflection

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
        _times_ number of times.
        '''
        element = np.array(element)
        for i in range(times):
            element = np.dot(element, np.array(generator))
        return element

    def get_elements(self):
        '''
        Function to generate a list containing  elements of group D4ht,
        similar to get_elements() of BArray.

        These are the base elements in 3x3 matrix notation without translations.
        '''
        # specify generators
        g1 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])  # 180 degrees over y
        g2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90 degrees over z

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
    Returns an D4htArray of shape size, with randomly chosen elements in int parameterization.
    '''
    data = np.zeros(size + (6,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 2, size)  # 180 degree rotation over y
    data[..., 1] = np.random.randint(0, 4, size)  # 90 degree rotation over z
    data[..., 2] = np.random.randint(0, 2, size)  # reflection or not
    data[..., 3] = np.random.randint(minu, maxu, size)  # translation on x
    data[..., 4] = np.random.randint(minv, maxv, size)  # translation on y
    data[..., 5] = np.random.randint(minw, maxw, size)  # translation on z
    return D4htArray(data=data, p='int')


def identity(p='int'):
    '''
    Returns the identity element: a matrix with 1's on the diagonal.
    '''
    li = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    e = D4htArray(data=np.array(li, dtype=np.int), p='hmat')
    return e.reparameterize(p)


def meshgrid(minu=-1, maxu=2, minv=-1, maxv=2, minw=-1, maxw=2):
    '''
    Creates a meshgrid of all elements of the group, within the given
    translation parameters.
    '''
    li = [[y, z, m, u, v, w] for y in xrange(2) for z in xrange(4) for m in xrange(2) for u in xrange(minu, maxu) for v
          in xrange(minv, maxv) for w in xrange(minw, maxw)]
    return D4htArray(li, p='int')
