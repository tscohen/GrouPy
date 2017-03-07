
import numpy as np
import tensorflow as tf

from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d_util, gconv2d
from groupy.gfunc.z2func_array import Z2FuncArray
from groupy.gfunc.p4func_array import P4FuncArray
from groupy.gfunc.p4mfunc_array import P4MFuncArray
import groupy.garray.C4_array as C4a
import groupy.garray.D4_array as D4a

# NOTE: it seems like loading tensorflow and Chainer in the same session is likely to result in problems.
# I've disabled these tests for now (renamed to check_... instead of test_... so they are ignored by nose)
# They should still work if you run these in a separate session


def check_c4_z2_conv_equivariance():
    im = np.random.randn(2, 5, 5, 1)
    x, y = make_graph('Z2', 'C4')
    check_equivariance(im, x, y, Z2FuncArray, P4FuncArray, C4a)


def check_c4_c4_conv_equivariance():
    im = np.random.randn(2, 5, 5, 4)
    x, y = make_graph('C4', 'C4')
    check_equivariance(im, x, y, P4FuncArray, P4FuncArray, C4a)


def check_d4_z2_conv_equivariance():
    im = np.random.randn(2, 5, 5, 1)
    x, y = make_graph('Z2', 'D4')
    check_equivariance(im, x, y, Z2FuncArray, P4MFuncArray, D4a)


def check_d4_d4_conv_equivariance():
    im = np.random.randn(2, 5, 5, 8)
    x, y = make_graph('D4', 'D4')
    check_equivariance(im, x, y, P4MFuncArray, P4MFuncArray, D4a)


def make_graph(h_input, h_output):
    gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
        h_input=h_input, h_output=h_output, in_channels=1, out_channels=1, ksize=3)
    nti = gconv_shape_info[-2]
    x = tf.placeholder(tf.float32, [None, 5, 5, 1 * nti])
    w = tf.Variable(tf.truncated_normal(w_shape, stddev=1.))
    y = gconv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME',
                gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)
    return x, y


def check_equivariance(im, input, output, input_array, output_array, point_group):

    # Transform the image
    f = input_array(im.transpose((0, 3, 1, 2)))
    g = point_group.rand()
    gf = g * f
    im1 = gf.v.transpose((0, 2, 3, 1))

    # Compute
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    yx = sess.run(output, feed_dict={input: im})
    yrx = sess.run(output, feed_dict={input: im1})
    sess.close()

    # Transform the computed feature maps
    fmap1_garray = output_array(yrx.transpose((0, 3, 1, 2)))
    r_fmap1_data = (g.inv() * fmap1_garray).v.transpose((0, 2, 3, 1))

    print (np.abs(yx - r_fmap1_data).sum())
    assert np.allclose(yx, r_fmap1_data, rtol=1e-5, atol=1e-3)
