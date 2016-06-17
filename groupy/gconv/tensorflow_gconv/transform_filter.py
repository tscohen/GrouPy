
import tensorflow as tf


def transform_filter_2d(w, inds, w_shape, inds_shape):
    """
    Transform a set of filters defined on a split (symmorphic) plane group G.

    The input filterbank w has shape (no, ni, nti, n, n), where:
    no: the number of output channels
    ni: the number of input channels
    nti: the number of transformations in the stabilizer of the origin in the input space
    For example, nti == 1 for images / functions on Z2, since only the identity translation leaves the origin invariant.
    Similarly, nti == 4 for the group p4, because there are 4 transformations in p4 (namely, the four rotations around
    the origin) that leave the origin in p4 (i.e. the identity transformation) fixed.
    n: the filter width and height

    The index array has shape (nto, nti, n, n, 3)

    The output filterbank rotated_w has shape (no, nto, ni, nti, n, n), where a length-nto axis is added.
    The filter at rotated_w[o, t, i] is the filter w[o, i] transformed by t.
    """

    no, ni, nti, n, n2 = w_shape
    nto, _nti, _n, _n2, _nc = inds_shape

    assert nti == _nti
    assert n == n2
    assert n == _n
    assert n == _n2
    assert _nc == 3

    T = inds[..., 0]  # shape (nto, nti, n, n)
    U = inds[..., 1]  # shape (nto, nti, n, n)
    V = inds[..., 2]  # shape (nto, nti, n, n)
    inds_flat = T * n * n + U * n + V

    w_flat = tf.transpose(tf.reshape(w, [no * ni, nti * n * n]))  # shape (nti * n * n, no * ni)

    rotated_w_flat = tf.gather(w_flat, inds_flat)  # shape (nto * nti * n * n, no * ni)

    rotated_w = tf.reshape(rotated_w_flat, [nto, nti, n, n, no, ni])  # shape (nto, nti, n, n, no, ni)
    rotated_w = tf.transpose(rotated_w, [4, 0, 5, 1, 2, 3])           # shape (no, nto, ni, nti, n, n)

    return rotated_w
