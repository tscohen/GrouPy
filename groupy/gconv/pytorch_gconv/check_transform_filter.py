import numpy as np
import tensorflow as tf
import torch

from groupy.gconv.tensorflow_gconv.transform_filter import transform_filter_2d_nchw, transform_filter_2d_nhwc
from groupy.gconv.make_gconv_indices import make_c4_z2_indices, make_c4_p4_indices,\
    make_d4_z2_indices, make_d4_p4m_indices, flatten_indices
from groupy.gconv.pytorch_gconv.splitgconv2d import trans_filter as pytorch_trans_filter_

# Comparing tensorflow and pytorch filter transformation


def check_c4_z2():
    inds = make_c4_z2_indices(ksize=3)
    w = np.random.randn(6, 7, 1, 3, 3)

    rt = tf_trans_filter(w, inds)
    rp = pytorch_trans_filter(w, inds)
    diff = np.abs(rt - rp).sum()
    print ('>>>>> DIFFERENCE:', diff)
    assert diff == 0


def check_c4_p4():
    inds = make_c4_p4_indices(ksize=3)
    w = np.random.randn(6, 7, 4, 3, 3)

    rt = tf_trans_filter(w, inds)
    rp = pytorch_trans_filter(w, inds)

    diff = np.abs(rt - rp).sum()
    print ('>>>>> DIFFERENCE:', diff)
    assert diff == 0


def check_d4_z2():
    inds = make_d4_z2_indices(ksize=3)
    w = np.random.randn(6, 7, 1, 3, 3)

    rt = tf_trans_filter(w, inds)
    rp = pytorch_trans_filter(w, inds)

    diff = np.abs(rt - rp).sum()
    print ('>>>>> DIFFERENCE:', diff)
    assert diff == 0


def check_d4_p4m():
    inds = make_d4_p4m_indices(ksize=3)
    w = np.random.randn(6, 7, 8, 3, 3)

    rt = tf_trans_filter(w, inds)
    rp = pytorch_trans_filter(w, inds)

    diff = np.abs(rt - rp).sum()
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


def pytorch_trans_filter(w, inds):
    w = torch.DoubleTensor(w)
    rp = pytorch_trans_filter_(w, inds)
    rp = rp.numpy()
    return rp
