import tensorflow as tf

from groupy.gconv.make_gconv_indices import make_o_z3_indices, make_o_ot_indices, make_c4h_z3_indices, \
    make_c4h_c4ht_indices, make_d4h_z3_indices, make_d4h_d4ht_indices, make_oh_z3_indices, make_oh_oht_indices, \
    flatten_indices_3d
from groupy.gconv.tensorflow_gconv.transform_filter import transform_filter_3d_nhwc


def gconv3d(input, filter, strides, padding, gconv_indices, gconv_shape_info,
            use_cudnn_on_gpu=None, data_format='NHWC', name=None):
    """
    Implements the g-convolution. Similar interface as the standard 3D convolution in tensorflow, with gconv_indices
    and gconv_shape_info as additional parameters. These can be obtained using gconv3d_util.
    Args:
        input: tensor with (b, z, y, x, c) axes
        filter: tensor with (ksize, ksize, ksize, in channels * transformations, out_channels) axes
        strides: A list of ints. 1-D of length 5. The stride of the sliding window for each dimension of input.
        padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
        gconv_indices:  indices used in the filter transformation step of the G-Conv.
        gconv_shape_info: a tuple containing information for the gconv: in/out channels, in/out transformations, ksize
        data_format: only nhwc supported
        name: a name for the operation

    Returns:
        conv: tensor with (batch, z, y, x, c) axes

    """
    if data_format != 'NHWC':
        raise NotImplemented('Currently only NHWC data_format is supported. Got:' + str(data_format))

    # Transform the filters
    transformed_filter = transform_filter_3d_nhwc(w=filter, flat_indices=gconv_indices, shape_info=gconv_shape_info)

    # Convolve input with transformed filters
    conv = tf.nn.conv3d(input=input, filter=transformed_filter, strides=strides, padding=padding, name=name)

    return conv


def gconv3d_util(h_input, h_output, in_channels, out_channels, ksize):
    """
    Convenience function for setting up static data required for the G-Conv. The number of 3D channels will be
    1, 8, 16, 24 or 48 times larger depending on the value of h_input and h_output.

    Args:
        h_input: Z3, C4H, D4H, O, OH -- use one. Z3 for first layer.
        h_output: Z3, C4H, D4H, O, OH -- use one.
        in_channels: number of input channels of the 3D channels on the group.
        out_channels: number of output channels of the 3D channels on the group.
        ksize: the spatial size of filter kernels, typicall 3, 5 or 7. Only uneven ksize is supported.

    Returns:
        gconv_indices: an array of indices used in the filter transformation step of gconv3d
        w_shape: the shape of the filter tensor to be allocated and passed to gconv3d
        gconv_shape_info: shape information required by gconv3d
                          -- (nr. out channels, nr. out transformations, nr. in channels, nr. in tranformations, ksize)
    """

    # uppercase for consistency
    h_input = h_input.upper()
    h_output = h_output.upper()

    # get number of transformations in and out
    mapping = {'Z3': 1, 'C4': 4, 'D4': 8, 'O': 24, 'C4H': 8, 'D4H': 16, 'OH': 48}
    nti = mapping[h_input]
    nto = mapping[h_output]

    # get gconv_indices
    if h_input == 'Z3' and h_output == 'O':
        gconv_indices = make_o_z3_indices(ksize=ksize)
    elif h_input == 'O' and h_output == 'O':
        gconv_indices = make_o_ot_indices(ksize=ksize)
    elif h_input == 'Z3' and h_output == 'C4H':
        gconv_indices = make_c4h_z3_indices(ksize=ksize)
    elif h_input == 'C4H' and h_output == 'C4H':
        gconv_indices = make_c4h_c4ht_indices(ksize=ksize)
    elif h_input == 'Z3' and h_output == 'D4H':
        gconv_indices = make_d4h_z3_indices(ksize=ksize)
    elif h_input == 'D4H' and h_output == 'D4H':
        gconv_indices = make_d4h_d4ht_indices(ksize=ksize)
    elif h_input == 'Z3' and h_output == 'OH':
        gconv_indices = make_oh_z3_indices(ksize=ksize)
    elif h_input == 'OH' and h_output == 'OH':
        gconv_indices = make_oh_oht_indices(ksize=ksize)
    else:
        raise ValueError('Unknown (h_input, h_output) pair:' + str((h_input, h_output)))

    # flatten and get shape information and filter tensor shape
    gconv_indices = flatten_indices_3d(gconv_indices)
    w_shape = (ksize, ksize, ksize, in_channels * nti, out_channels)
    gconv_shape_info = (out_channels, nto, in_channels, nti, ksize)

    return gconv_indices, gconv_shape_info, w_shape


def gconv2d_addbias(input, bias, nti=8):
    """
    In a G-CNN, the feature maps are interpreted as functions on a group G instead of functions on the plane Z^2.
    Just like how we use a single scalar bias per 2D feature map, in a G-CNN we should use a single scalar bias per
    G-feature map. Failing to do this breaks the equivariance and typically hurts performance.
    A G-feature map usually consists of a number (e.g. 4 or 8) adjacent channels.
    This function will add a single bias vector to a stack of feature maps that has e.g. 4 or 8 times more 2D channels
    than G-channels, by replicating the bias across adjacent groups of 2D channels.

    :param input: tensor of shape (n, h, w, ni * nti), where n is the batch dimension, (h, w) are the height and width,
     ni is the number of input G-channels, and nti is the number of transformations in H.
    :param bias: tensor of shape (ni,)
    :param nti: number of transformations, e.g. 4 for C4/p4 or 8 for D4/p4m.
    :return: input with bias added
    """
    # input = tf.reshape(input, ())
    pass  # TODO
