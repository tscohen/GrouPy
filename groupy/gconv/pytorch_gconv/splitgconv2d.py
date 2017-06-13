import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nninit
from torch.autograd import Variable


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    else:
        return (x, x)


class SplitGConv2D(nn.Module):
    """
    Group convolution base class for split plane groups.

    A plane group (aka wallpaper group) is a group of distance-preserving
    transformations that includes two independent discrete translations.

    A group is called split (or symmorphic) if every element in this group can
    be written as the composition of an element from the "stabilizer of the
    origin" and a translation. The stabilizer of the origin consists of those
    transformations in the group that leave the origin fixed. For example, the
    stabilizer in the rotation-translation group p4 is the set of rotations
    around the origin, which is (isomorphic to) the group C4.

    Most plane groups are split, but some include glide-reflection generators;
    such groups are not split.  For split groups G, the G-conv can be split
    into a "filter transform" and "translational convolution" part.

    Different subclasses of this class implement the filter transform for
    various groups, while this class implements the common functionality.

    This PyTorch implementation mimicks the original Chainer implementation.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 flat_channels=False,
                 stride=1,
                 pad=0,
                 bias=True,
                 *args, **kwargs):
        """
        :param in_channels:
        :param out_channels:
        :param ksize:
        :param flat_channels
        :param stride:
        :param pad:
        :param bias:
        :return:
        """

        super(SplitGConv2D, self).__init__(*args, **kwargs)

        if not isinstance(ksize, int):
            raise TypeError('ksize must be an integer (only square filters '
                            'are supported).')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.flat_channels = flat_channels
        self.use_bias = bias

        self.weight = nn.Parameter(torch.Tensor(self.out_channels,
                                                self.in_channels,
                                                self.input_stabilizer_size,
                                                self.ksize,
                                                self.ksize))
        nninit.xavier_normal(self.weight)

        if self.use_bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_channels))

        # Shorthands
        ni, no = in_channels, out_channels
        nti, nto = self.input_stabilizer_size, self.output_stabilizer_size
        n = self.ksize

        self.expand_shape = (no, nto, ni, nti * n * n)
        self.weight_shape = (no * nto, ni * nti, n, n)
        self.weight_flat_shape = (no, 1, ni, nti * n * n)

        transform_indices = self._create_indices(self.expand_shape)
        self.register_buffer('transform_indices', transform_indices)

    def _create_indices(self, expand_shape):
        no, nto, ni, r = expand_shape
        transform_indices = self.make_transformation_indices(ksize=self.ksize)
        transform_indices = transform_indices.astype(np.int64)
        transform_indices = transform_indices.reshape(1, nto, 1, r)
        transform_indices = torch.from_numpy(transform_indices)
        transform_indices = transform_indices.expand(*expand_shape)
        return transform_indices

    @property
    def input_stabilizer_size():
        raise NotImplementedError()

    @property
    def output_stabilizer_size():
        raise NotImplementedError()

    def make_transformation_indices(self, ksize):
        raise NotImplementedError()

    def forward(self, x):
        # Transform the filters
        w_flat_ = self.weight.view(self.weight_flat_shape)
        w_flat = w_flat_.expand(*self.expand_shape)
        w = torch.gather(w_flat, 3, Variable(self.transform_indices)) \
                 .view(self.weight_shape)

        # If flat_channels is False, we need to flatten the input feature maps
        # to have a single 1d feature dimension.
        if not self.flat_channels:
            batch_size = x.size(0)
            in_ny, in_nx = x.size()[-2:]
            x = x.view(batch_size,
                       self.in_channels * self.input_stabilizer_size,
                       in_ny,
                       in_nx)

        # Perform the 2D convolution
        y = F.conv2d(x, w, stride=self.stride, padding=self.pad)

        # Unfold the output feature maps
        # We do this even if flat_channels is True, because we need to add the
        # same bias to each G-feature map
        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.output_stabilizer_size,
                   ny_out, nx_out)

        # Add a bias to each G-feature map
        if self.use_bias:
            b = self.bias.view(1, self.out_channels, 1, 1, 1)
            b = b.expand_as(y)
            y = y + b

        # Flatten feature channels if needed
        if self.flat_channels:
            n, nc, ng, nx, ny = y.size()
            y = y.view(n, nc * ng, nx, ny)

        return y
