
import copy
import numpy as np
from groupy.garray.garray import GArray


class GFuncArray(object):

    def __init__(self, v, i2g):
        """
        A GFunc is a discretely sampled function on a group or homogeneous space G.
        The GFuncArray stores an array of GFuncs,
        together with a map from G to an index set I (the set of sampling points) and the inverse of this map.

        The ndarray v can be thought of as a map
         v : J x I -> R
        from an index set J x I to real numbers.
        The index set J may have arbitrary shape, and each index in j identifies a GFunc.
        The index set I is the set of valid indices to the ndarray v.
        From here on, consider a single GFunc v : I -> R

        The GArray i2g can be thought of as a map
          i2g: I -> G
        that takes indices from I and produces a group element g in G.

        The map i2g is required to be invertible, and its inverse
         g2i : G -> I
        is implemented in the function g2i of a subclass.

        So we have the following diagram:
              i2g
          I <-----> G
          |   g2i
        v |
          |
          V
          R

        So v implicitly defines a function v' on G:
        v'(g) = v(g2i(g))

        If we have a map T: G - > G (e.g. left multiplication by g^-1), that we want to precompose with v',
         w'(g) = v'(T(g))

        we can get the corresponding map v by composing maps like this:
        I ---> G ---> G ---> I ---> R
          i2g     T     g2i     v
        to obtain the transformed function w : I -> R.
        This class knows how to produce such a w as an ndarray that directly maps indices to numbers,
        (and such that the indices correspond to group elements by the same maps i2g and g2i)

        :param i2g: a GArray of sample points. The sample points are elements of G or H
        :param v: a numpy.ndarray of values corresponding to the sample points.
        """

        if not isinstance(i2g, GArray):
            raise TypeError('i2g must be of type GArray, got' + str(type(i2g)) + ' instead.')

        if not isinstance(v, np.ndarray):
            raise TypeError('v must be of type np.ndarray, got ' + str(type(v)) + ' instead.')

        if i2g.shape != v.shape[-i2g.ndim:]:  # TODO: allow vector-valued gfunc, or leave this to Section?
            raise ValueError('The trailing axes of v must match the shape of i2g. Got ' +
                             str(i2g.shape) + ' and ' + str(v.shape) + '.')

        self.i2g = i2g
        self.v = v

    def __call__(self, sample_points):
        """
        Evaluate the G-func at the sample points
        """
        if not isinstance(sample_points, type(self.i2g)):
            raise TypeError('Invalid type ' + str(type(sample_points)) + ' expected ' + str(type(self.i2g)))

        si = self.g2i(sample_points)
        inds = [Ellipsis] + [si[..., i] for i in range(si.shape[-1])]
        vi = self.v[inds]
        ret = copy.copy(self)
        ret.v = vi
        return ret

    def __getitem__(self, item):
        """
        Get an element from the array of G-funcs
        """
        # TODO bounds / dim checking
        ret = copy.copy(self)
        ret.v = self.v[item]
        return ret

    def __mul__(self, other):
        # Compute self * other
        if isinstance(other, GArray):
            gp = self.right_translation_points(other)
            return self(gp)
        else:
            # Python assumes we *return* NotImplemented instead of raising NotImplementedError,
            # when we dont know how to left multiply the given type of object by self.
            return NotImplemented

    def __rmul__(self, other):
        # Compute other * self
        if isinstance(other, GArray):
            gp = self.left_translation_points(other)
            return self(gp)
        else:
            # Python assumes we *return* NotImplemented instead of raising NotImplementedError,
            # when we dont know how to left multiply the given type of object by self.
            return NotImplemented

    def g2i(self, g):
        raise NotImplementedError()

    def left_translation_points(self, g):
        return g.inv() * self.i2g

    def right_translation_points(self, g):
        return self.i2g * g

    def left_translation_indices(self, g):
        ginv_s = self.left_translation_points(g)
        ginv_s_inds = self.g2i(ginv_s)
        return ginv_s_inds

    def right_translation_indices(self, g):
        sg = self.right_translation_points(g)
        sg_inds = self.g2i(sg)
        return sg_inds

    @property
    def ndim(self):
        return self.v.ndim - self.i2g.ndim

    @property
    def shape(self):
        return self.v.shape[:self.ndim]

    @property
    def f_shape(self):
        return self.i2g.shape

    @property
    def f_ndim(self):
        return self.i2g.ndim
