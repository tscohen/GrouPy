
import copy
import numpy as np

# TODO: add checks in constructor to make sure data argument is well formed (for the given parameterization).
# TODO: for example, for a finite group, when p=='int', we want data >= 0 and data <= order_of_G


class GArray(object):
    """
    GArray is a wrapper of numpy.ndarray that can store group elements instead of numbers.
    Subclasses of GArray implement the needed functionality for specific groups G.

    A GArray has a shape (how many group elements are in the array),
    and a g_shape, which is the shape used to store group element itself (e.g. (3, 3) for a 3x3 matrix).
    The user of a GArray usually doesn't need to know the g_shape, or even the group G.
    GArrays should be fully gufunc compatible; i.e. they support broadcasting according to the rules of numpy.
    A GArray of a given shape broadcasts just like a numpy array of that shape, regardless of the g_shape.

    A group may have multiple parameterizations, each with its own g_shape.
    Group elements can be composed and compared (using the * and == operators) irrespective of their parameterization.
    """

    # To be set in subclass
    parameterizations = []
    _g_shapes = {}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'GArray Base Class'

    def __init__(self, data, p):

        if not isinstance(data, np.ndarray):
            raise TypeError('data should be of type np.ndarray, got ' + str(type(data)) + ' instead.')

        if p not in self.parameterizations:
            raise ValueError('Unknown parameterization: ' + str(p))

        self.data = data
        self.p = p
        self.g_shape = self._g_shapes[p]
        self.shape = data.shape[:data.ndim - self.g_ndim]

        if self.data.shape[self.ndim:] != self.g_shape:
            raise ValueError('Invalid data shape. Expected shape ' + str(self.g_shape) +
                             ' for parameterization ' + str(p) +
                             '. Got data shape ' + str(self.data.shape[self.ndim:]) + ' instead.')

    def inv(self):
        """
        Compute the inverse of the group elements

        :return: GArray of the same shape as self, containing inverses of each element in self.
        """
        raise NotImplementedError()

    def reparameterize(self, p):
        """
        Return a GArray containing the same group elements in the requested parameterization p.
        If p is the same as the current parameterization, this function returns self.

        :param p: the requested parameterization. Must be an element of self.parameterizations
        :return: GArray subclass with reparameterized elements.
        """
        if p == self.p:
            return self

        if p not in self.parameterizations:
            raise ValueError('Unknown parameterization:' + str(p))

        if not (self.p, p) in self._reparameterizations:
            return ValueError('No reparameterization implemented for ' + self.p + ' -> ' + str(p))

        new_data = self._reparameterizations[(self.p, p)](self.data)
        return self.factory(data=new_data, p=p)

    def reshape(self, *shape):
        shape = shape[0] if isinstance(shape[0], tuple) else shape
        full_shape = shape + self.g_shape
        new = copy.copy(self)
        new.data = self.data.reshape(full_shape)
        new.shape = shape
        return new

    def flatten(self):
        return self.reshape(np.prod(self.shape))

    def __mul__(self, other):
        """
        Act on another GArray from the left.

        If the arrays do not have the same shape for the loop dimensions, they are broadcast together.

        The left action is chosen from self.left_actions depending on the type of other;
        this way, a GArray subclass can act on various other compatible GArray subclasses.

        This function will still work if self and other have a different parameterization.
        The output is always returned in the other's parameterization.

        :param other:
        :return:
        """
        for garray_type in self._left_actions:
            if isinstance(other, garray_type):
                return self._left_actions[garray_type](self, other)
        return NotImplemented

    def __eq__(self, other):
        """
        Elementwise equality test of GArrays.
        Group elements are considered equal if, after reparameterization, they are numerically identical.

        :param other: GArray to be compared to
        :return: a boolean numpy.ndarray of shape self.shape
        """
        if isinstance(other, self.__class__) or isinstance(self, other.__class__):
            return (self.data == other.reparameterize(self.p).data).all(axis=-1)
        else:
            return NotImplemented

    def __ne__(self, other):
        """
        Elementwise inequality test of GArrays.
        Group elements are considered equal if, after reparameterization, they are numerically identical.

        :param other: GArray to be compared to
        :return: a boolean numpy.ndarray of shape self.shape
        """
        if isinstance(other, self.__class__) or isinstance(self, other.__class__):
            return (self.data != other.reparameterize(self.p).data).any(axis=-1)
        else:
            return NotImplemented

    def __len__(self):
        if len(self.shape) > 0:
            return self.shape[0]
        else:
            return 1
    
    def __getitem__(self, key):
        # We return a factory here instead of self.__class__(..) so that a subclass
        # can decide what type the result should have.
        # For instance, a FiniteGroup may wish to return an instance of a different GArray instead of a FiniteGroup.
        return self.factory(data=self.data[key], p=self.p)

    # def __setitem__(self, key, value):
    #    raise NotImplementedError()  # TODO

    def __delitem__(self, key):
        # Raise an error to mimic the behaviour of numpy.ndarray
        raise ValueError('cannot delete garray elements')

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    def __contains__(self, item):
        return (self == item).any()

    # Factory is used to create new instances from a given instance, e.g. when using __getitem__ or inv()
    # In some cases (e.g. FiniteGroup), we may wish to instantiate a superclass instead of self.__class__
    # Example: D4Group instantiates a D4Array when an element is selected.
    def factory(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)

    @property
    def size(self):
        # Usually, np.prod(self.shape) returns an int because self.shape contains ints.
        # However, if self.shape == (), np.prod(self.shape) returns the float 1.0,
        # so we convert to int.
        return int(np.prod(self.shape))

    @property
    def g_ndim(self):
        """
        The shape of each group element in this GArray, for the current parameterization.

        :return:
        """
        return len(self.g_shape)

    @property
    def ndim(self):
        return len(self.shape)

    def __repr__(self):
        return self._group_name + self.data.__repr__()