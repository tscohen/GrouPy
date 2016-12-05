
import tensorflow as tf

from groupy.gconv.make_gconv_indices import make_c4_z2_indices, make_c4_p4_indices,\
    make_d4_z2_indices, make_d4_p4m_indices, flatten_indices
from groupy.gconv.tensorflow_gconv.transform_filter import transform_filter_2d_nchw, transform_filter_2d_nhwc


def gconv2d(input, filter, strides, padding, gconv_indices, gconv_shape_info,
            use_cudnn_on_gpu=None, data_format='NHWC', name=None):
    """
    Tensorflow implementation of the group convolution.
    This function has the same interface as the standard convolution nn.conv2d, except for two new parameters,
    gconv_indices and gconv_shape_info. These can be obtained from gconv2d_util(), and are described below

    :param input: a tensor with (batch, height, width, in channels) axes.
    :param filter: a tensor with (ksize, ksize, in channels * in transformations, out channels) axes.
      The shape for filter can be obtained from gconv2d_util().
    :param strides: A list of ints. 1-D of length 4. The stride of the sliding window for each dimension of input.
     Must be in the same order as the dimension specified with format.
    :param padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
    :param gconv_indices: indices used in the filter transformation step of the G-Conv.
      Can be obtained from gconv2d_util() or using a command like flatten_indices(make_d4_p4m_indices(ksize=3)).
    :param gconv_shape_info: a tuple containing
     (num output channels, num output transformations, num input channels, num input transformations, kernel size)
     Can be obtained from gconv2d_util()
    :param use_cudnn_on_gpu: an optional bool. Defaults to True.
    :param data_format: the order of axes. Currently only NCHW is supported
    :param name: a name for the operation (optional)
    :return: tensor with (batch, out channels, height, width) axes.
    """

    if data_format != 'NHWC':
        raise NotImplemented('Currently only NHWC data_format is supported. Got:' + str(data_format))

    # Transform the filters
    transformed_filter = transform_filter_2d_nhwc(w=filter, flat_indices=gconv_indices, shape_info=gconv_shape_info)

    # Convolve input with transformed filters
    conv = tf.nn.conv2d(input=input, filter=transformed_filter, strides=strides, padding=padding,
                        use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format, name=name)

    return conv


def gconv2d_util(h_input, h_output, in_channels, out_channels, ksize):
    """
    Convenience function for setting up static data required for the G-Conv.
     This function returns:
      1) an array of indices used in the filter transformation step of gconv2d
      2) shape information required by gconv2d
      5) the shape of the filter tensor to be allocated and passed to gconv2d

    :param h_input: one of ('Z2', 'C4', 'D4'). Use 'Z2' for the first layer. Use 'C4' or 'D4' for later layers.
    :param h_output: one of ('C4', 'D4'). What kind of transformations to use (rotations or roto-reflections).
      The choice of h_output of one layer should equal h_input of the next layer.
    :param in_channels: the number of input channels. Note: this refers to the number of (3D) channels on the group.
    The number of 2D channels will be 1, 4, or 8 times larger, depending the value of h_input.
    :param out_channels: the number of output channels. Note: this refers to the number of (3D) channels on the group.
    The number of 2D channels will be 1, 4, or 8 times larger, depending on the value of h_output.
    :param ksize: the spatial size of the filter kernels (typically 3, 5, or 7).
    :return: gconv_indices
    """

    if h_input == 'Z2' and h_output == 'C4':
        gconv_indices = flatten_indices(make_c4_z2_indices(ksize=ksize))
        nti = 1
        nto = 4
    elif h_input == 'C4' and h_output == 'C4':
        gconv_indices = flatten_indices(make_c4_p4_indices(ksize=ksize))
        nti = 4
        nto = 4
    elif h_input == 'Z2' and h_output == 'D4':
        gconv_indices = flatten_indices(make_d4_z2_indices(ksize=ksize))
        nti = 1
        nto = 8
    elif h_input == 'D4' and h_output == 'D4':
        gconv_indices = flatten_indices(make_d4_p4m_indices(ksize=ksize))
        nti = 8
        nto = 8
    else:
        raise ValueError('Unknown (h_input, h_output) pair:' + str((h_input, h_output)))

    w_shape = (ksize, ksize, in_channels * nti, out_channels)
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
    pass # TODO
