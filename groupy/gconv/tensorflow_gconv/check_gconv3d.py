import groupy.garray.O_array as O
import groupy.garray.Oh_array as Oh
import groupy.garray.C4h_array as C4h
import groupy.garray.D4h_array as D4h
import numpy as np
import tensorflow as tf
from groupy.gconv.tensorflow_gconv.splitgconv3d import gconv3d_util, gconv3d
from groupy.gfunc.otfunc_array import OtFuncArray
from groupy.gfunc.c4htfunc_array import C4htFuncArray
from groupy.gfunc.d4htfunc_array import D4htFuncArray
from groupy.gfunc.ohtfunc_array import OhtFuncArray
from groupy.gfunc.z3func_array import Z3FuncArray


def check_o_z3_conv_equivariance():
    ksize = 3
    im = np.random.randn(2, ksize, ksize, ksize, 1)
    x, y = make_graph('Z3', 'O', ksize)
    check_equivariance(im, x, y, Z3FuncArray, OtFuncArray, O)


def check_o_o_conv_equivariance():
    ksize = 3
    im = np.random.randn(2, ksize, ksize, ksize, 24)
    x, y = make_graph('O', 'O', ksize)
    check_equivariance(im, x, y, OtFuncArray, OtFuncArray, O)


def check_oh_z3_conv_equivariance():
    ksize = 3
    im = np.random.randn(2, ksize, ksize, ksize, 1)
    x, y = make_graph('Z3', 'OH', ksize)
    check_equivariance(im, x, y, Z3FuncArray, OhtFuncArray, Oh)


def check_oh_oh_conv_equivariance():
    ksize = 3
    im = np.random.randn(2, ksize, ksize, ksize, 48)
    x, y = make_graph('OH', 'OH', ksize)
    check_equivariance(im, x, y, OhtFuncArray, OhtFuncArray, Oh)


def check_c4h_z3_conv_equivariance():
    ksize = 3
    im = np.random.randn(2, ksize, ksize, ksize, 1)
    x, y = make_graph('Z3', 'C4H', ksize)
    check_equivariance(im, x, y, Z3FuncArray, C4htFuncArray, C4h)


def check_c4h_c4h_conv_equivariance():
    ksize = 3
    im = np.random.randn(2, ksize, ksize, ksize, 8)
    x, y = make_graph('C4H', 'C4H', ksize)
    check_equivariance(im, x, y, C4htFuncArray, C4htFuncArray, C4h)


def check_d4h_z3_conv_equivariance():
    ksize = 3
    im = np.random.randn(2, ksize, ksize, ksize, 1)
    x, y = make_graph('Z3', 'D4H', ksize)
    check_equivariance(im, x, y, Z3FuncArray, D4htFuncArray, D4h)


def check_d4h_d4h_conv_equivariance():
    ksize = 3
    im = np.random.randn(2, ksize, ksize, ksize, 16)
    x, y = make_graph('D4H', 'D4H', ksize)
    check_equivariance(im, x, y, Z3FuncArray, D4htFuncArray, D4h)


def make_graph(h_input, h_output, ksize):
    gconv_indices, gconv_shape_info, w_shape = gconv3d_util(
        h_input=h_input, h_output=h_output, in_channels=1, out_channels=1, ksize=ksize)
    nti = gconv_shape_info[-2]

    x = tf.placeholder(tf.float32, [None, ksize, ksize, ksize, 1 * nti])
    w = tf.Variable(tf.truncated_normal(w_shape, stddev=1.))

    y = gconv3d(input=x, filter=w, strides=[1, 1, 1, 1, 1], padding='SAME',
                gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)
    return x, y


def check_equivariance(im, input, output, input_array, output_array, point_group):
    # Transform the image
    f = input_array(im.transpose((0, 4, 1, 2, 3)))
    g = point_group.rand()
    gf = g * f
    im1 = gf.v.transpose((0, 2, 3, 4, 1))

    # Compute
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    yx = sess.run(output, feed_dict={input: im})
    yrx = sess.run(output, feed_dict={input: im1})
    sess.close()

    # Transform the computed feature maps
    fmap1_garray = output_array(yrx.transpose((0, 4, 1, 2, 3)))
    r_fmap1_data = (g.inv() * fmap1_garray).v.transpose((0, 2, 3, 4, 1))

    assert np.allclose(yx, r_fmap1_data, rtol=1e-5, atol=1e-3)

