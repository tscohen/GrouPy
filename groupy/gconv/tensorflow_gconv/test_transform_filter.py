
import numpy as np
import tensorflow as tf

from groupy.gconv.make_gconv_indices import make_c4_z2_indices, make_c4_p4_indices
from groupy.gconv.tensorflow_gconv.transform_filter import transform_filter_2d_nchw, transform_filter_2d_nhwc

from groupy.gconv.make_gconv_indices import make_c4_z2_indices, make_c4_p4_indices,\
    make_d4_z2_indices, make_d4_p4m_indices, flatten_indices

# NOTE: it seems like loading tensorflow and Chainer in the same session is likely to result in problems.
# I've disabled these tests for now (renamed to check_... instead of test_... so they are ignored by nose)
# They should still work if you run these in a separate session


def check_c4_z2():
    inds = make_c4_z2_indices(ksize=3)
    w = np.random.randn(6, 7, 1, 3, 3)

    rt = tf_trans_filter(w, inds)
    rc = ch_trans_filter(w, inds)

    diff = np.abs(rt - rc).sum()
    print ('>>>>> DIFFERENCE:', diff)
    assert diff == 0


def check_c4_p4():
    inds = make_c4_p4_indices(ksize=3)
    w = np.random.randn(6, 7, 4, 3, 3)

    rt = tf_trans_filter(w, inds)
    rc = ch_trans_filter(w, inds)

    diff = np.abs(rt - rc).sum()
    print ('>>>>> DIFFERENCE:', diff)
    assert diff == 0


def check_d4_z2():
    inds = make_d4_z2_indices(ksize=3)
    w = np.random.randn(6, 7, 1, 3, 3)

    rt = tf_trans_filter(w, inds)
    rc = ch_trans_filter(w, inds)

    diff = np.abs(rt - rc).sum()
    print ('>>>>> DIFFERENCE:', diff)
    assert diff == 0


def check_d4_p4m():
    inds = make_d4_p4m_indices(ksize=3)
    w = np.random.randn(6, 7, 8, 3, 3)

    rt = tf_trans_filter(w, inds)
    rc = ch_trans_filter(w, inds)

    diff = np.abs(rt - rc).sum()
    print ('>>>>> DIFFERENCE:', diff)
    assert diff == 0


def tf_trans_filter(w, inds):

    flat_inds = flatten_indices(inds)
    no, ni, nti, n, _ = w.shape
    shape_info = (no, inds.shape[0], ni, nti, n)

    w = w.transpose((3, 4, 2, 1, 0)).reshape((n, n, nti * ni, no))

    wt = tf.constant(w)
    rwt = transform_filter_2d_nhwc(wt, flat_inds, shape_info)

    sess = tf.Session()
    rwt = sess.run(rwt)
    sess.close()

    nto = inds.shape[0]
    rwt = rwt.transpose(3, 2, 0, 1).reshape(no, nto, ni, nti, n, n)
    return rwt

def tf_trans_filter2(w, inds):

    flat_inds = flatten_indices(inds)
    no, ni, nti, n, _ = w.shape
    shape_info = (no, inds.shape[0], ni, nti, n)

    w = w.reshape(no, ni * nti, n, n)

    wt = tf.constant(w)
    rwt = transform_filter_2d_nchw(wt, flat_inds, shape_info)

    sess = tf.Session()
    rwt = sess.run(rwt)
    sess.close()

    nto = inds.shape[0]
    rwt = rwt.reshape(no, nto, ni, nti, n, n)
    return rwt

def ch_trans_filter(w, inds):
    from chainer import cuda, Variable
    from groupy.gconv.chainer_gconv.transform_filter import TransformGFilter

    w_gpu = cuda.to_gpu(w)
    inds_gpu = cuda.to_gpu(inds)

    wv = Variable(w_gpu)
    rwv = TransformGFilter(inds_gpu)(wv)

    return cuda.to_cpu(rwv.data)
