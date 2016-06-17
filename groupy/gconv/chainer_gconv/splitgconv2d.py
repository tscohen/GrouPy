import math

import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable
from chainer.utils import type_check

from groupy.gconv.chainer_gconv.transform_filter import TransformGFilter

# Implementation note:
# The standard operation computed by chainer's Convolution2D is the correlation with filter psi on the right:
# output(x) = psi \corr f(t) = sum_T psi(T) f(t + T) = sum_T f(T) psi(T - t)
# This operation is equivariant: psi \corr [L_t f] = L_t [psi \corr f]
# What we want to compute is the following:
# o(r, t) = int_T f(T) [L_tr psi](T) dT
#         = int_T f(T) [L_r psi](T - t) dT
# This is exactly a Convolution2D correlation of f with the rotated filter [L_r psi].


class SplitGConv2D(chainer.Link):
    """
    Group convolution base class for split plane groups.

    A plane group (aka wallpaper group) is a group of distance-preserving transformations that includes two independent
    discrete translations.

    A group is called split (or symmorphic) if every element in this group can be written as the composition of an
    element from the "stabilizer of the origin" and a translation. The stabilizer of the origin consists of those
    transformations in the group that leave the origin fixed. For example, the stabilizer in the rotation-translation
    group p4 is the set of rotations around the origin, which is (isomorphic to) the group C4.

    Most plane groups are split, but some include glide-reflection generators; such groups are not split.
    For split groups G, the G-conv can be split into a "filter transform" and "translational convolution" part.

    Different subclasses of this class implement the filter transform for various groups, while this class implements
    the common functionality.
    """

    # To be set in subclass; the size of the stabilizer for the input and output space.
    # For example: for Z2, this is 1, for P4, this is 4, for P4M, this is 8.
    input_stabilizer_size = None
    output_stabilizer_size = None

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 filter_mask=None,
                 flat_channels=False,
                 stride=1,
                 pad=0,
                 wscale=1,
                 nobias=False,
                 use_cudnn=True,
                 initialW=None,
                 initial_bias=None,
                 dtype=np.float32):
        """
        :param in_channels:
        :param out_channels:
        :param ksize:
        :param filter_mask:
        :param stride:
        :param pad:
        :param wscale:
        :param nobias:
        :param use_cudnn:
        :param initialW:
        :param initial_bias:
        :param dtype:
        :return:
        """
        super(SplitGConv2D, self).__init__()

        self.dtype = np.dtype(dtype)
        if self.dtype != np.float32 and use_cudnn:
            raise FloatingPointError('float64 cudnn convolutions are buggy, see chainer issue #519')

        if not isinstance(ksize, int):
            raise TypeError('ksize must be an integer (only square filters are supported).')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride if hasattr(stride, '__getitem__') else (stride, stride)
        self.pad = pad if hasattr(pad, '__getitem__') else (pad, pad)
        self.use_cudnn = use_cudnn
        self.flat_channels = flat_channels

        w_shape = (self.out_channels, self.in_channels, self.input_stabilizer_size, self.ksize, self.ksize)
        self.add_param(name='W', shape=w_shape, dtype=self.dtype)

        if initialW is not None:
            assert initialW.shape == w_shape
            assert isinstance(initialW, self.xp.ndarray)
            self.W.data[:] = initialW.astype(self.dtype)
        else:
            self.W.data[:] = self.xp.random.normal(
                0, wscale * math.sqrt(1. / (self.input_stabilizer_size * self.ksize ** 2 * self.in_channels)),
                w_shape
            ).astype(self.dtype)

        self.usebias = not nobias
        if self.usebias:
            self.add_param(
                name='b',
                shape=self.out_channels,
                dtype=self.dtype
            )

            if initial_bias is not None:  # Todo: update in accordance with outcome of #525
                assert initial_bias.shape == (self.out_channels,)
                assert isinstance(initial_bias, self.xp.ndarray)
                self.b.data[:] = initial_bias.astype(self.dtype)
            elif not nobias:
                self.b.data[:] = self.xp.repeat(self.dtype.type(0.), self.out_channels)

        if filter_mask is not None:
            if not filter_mask.shape == (self.out_channels, self.in_channels, self.input_stabilizer_size):
                raise ValueError('Invalid filter_mask shape. Got: ' + str(filter_mask.shape) +
                                 '. Expected: ' + str((self.out_channels, self.in_channels, self.input_stabilizer_size)))

            filter_mask = filter_mask[..., None, None].astype(dtype)

            self.add_persistent('filter_mask', filter_mask)
        else:
            self.filter_mask = None

        self.add_persistent(name='inds', value=self.make_transformation_indices(ksize=self.ksize))

    def make_transformation_indices(self, ksize):
        raise NotImplementedError()

    def __call__(self, x):

        # Apply a mask to the filters (optional)
        if self.filter_mask is not None:
            w, m = F.broadcast(self.W, Variable(self.filter_mask))
            w = w * m
            # w = self.W * Variable(self.filter_mask)
        else:
            w = self.W

        # Transform the filters
        # w.shape  == (out_channels, in_channels, input_stabilizer_size, ksize, ksize)
        # tw.shape == (out_channels, output_stabilizer_size, in_channels, input_stabilizer_size, ksize, ksize)
        tw = TransformGFilter(self.inds)(w)

        # Fold the transformed filters
        tw_shape = (self.out_channels * self.output_stabilizer_size,
                    self.in_channels * self.input_stabilizer_size,
                    self.ksize, self.ksize)
        tw = F.Reshape(tw_shape)(tw)

        # If flat_channels is False, we need to flatten the input feature maps to have a single 1d feature dimension.
        if not self.flat_channels:
            batch_size = x.data.shape[0]
            in_ny, in_nx = x.data.shape[-2:]
            x = F.reshape(x, (batch_size, self.in_channels * self.input_stabilizer_size, in_ny, in_nx))

        # Perform the 2D convolution
        y = F.convolution_2d(x, tw, b=None, stride=self.stride, pad=self.pad, use_cudnn=self.use_cudnn)

        # Unfold the output feature maps
        # We do this even if flat_channels is True, because we need to add the same bias to each G-feature map
        batch_size, _, ny_out, nx_out = y.data.shape
        y = F.reshape(y, (batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out))

        # Add a bias to each G-feature map
        if self.usebias:
            bb = F.Reshape((1, self.out_channels, 1, 1, 1))(self.b)
            y, b = F.broadcast(y, bb)
            y = y + b

        # Flatten feature channels if needed
        if self.flat_channels:
            n, nc, ng, nx, ny = y.data.shape
            y = F.reshape(y, (n, nc * ng, nx, ny))

        return y
