
import numpy as np
import tensorflow as tf

from groupy.gconv.make_gconv_indices import make_c4_z2_indices, make_c4_p4_indices
from groupy.gconv.tensorflow_gconv.transform_filter import transform_filter_2d

# NOTE: these tests are not yet included in the tests to be run by nosetests (because we use the work 'check' instead
# of 'test' which nose looks for).
# This is because running tensorflow and chainer in the same session leads to unstable behaviour.
# But the code does seem to work, as can be checked by calling check_c4_z2 and check_c4_p4
# (but not both in the same session)
# A cleaner solution would be to test not against the chainer implementation, but against the expected behaviour /
# mathematical properties (associativity, invertibility, etc.)


def check_c4_z2():
    inds = make_c4_z2_indices(ksize=5)
    w = np.random.randn(6, 7, 1, 5, 5)

    rt = tf_trans_filter(w, inds)
    rc = ch_trans_filter(w, inds)

    diff = np.abs(rt - rc).sum()
    print '>>>>> DIFFERENCE:', diff
    assert diff == 0


def check_c4_p4():
    inds = make_c4_p4_indices(ksize=5)
    w = np.random.randn(6, 7, 4, 5, 5)

    rt = tf_trans_filter(w, inds)
    rc = ch_trans_filter(w, inds)

    diff = np.abs(rt - rc).sum()
    print '>>>>> DIFFERENCE:', diff
    assert diff == 0


def tf_trans_filter(w, inds):

    wt = tf.constant(w)
    rwt = transform_filter_2d(wt, inds, w.shape, inds.shape)

    sess = tf.Session()
    rwt = sess.run(rwt)
    sess.close()

    return rwt


def ch_trans_filter(w, inds):
    from chainer import cuda, Variable
    from groupy.gconv.chainer_gconv.transform_filter import TransformGFilter

    w_gpu = cuda.to_gpu(w)
    inds_gpu = cuda.to_gpu(inds)

    wv = Variable(w_gpu)
    rwv = TransformGFilter(inds_gpu)(wv)

    return cuda.to_cpu(rwv.data)
