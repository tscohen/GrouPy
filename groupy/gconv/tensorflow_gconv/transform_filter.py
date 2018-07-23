import tensorflow as tf


def transform_filter_3d_nhwc(w, flat_indices, shape_info, validate_indices=True):
    no, nto, ni, nti, n = shape_info
    w_flat = tf.reshape(w, [n * n * n * nti, ni, no])  # shape (n * n * n * nti, ni, no)

    # Do the transformation / indexing operation.
    transformed_w = tf.gather(w_flat, flat_indices,
                              validate_indices=validate_indices)  # shape (nto, nti, n, n, n, ni, no)

    # Put the axes in the right order, and collapse them to get a standard shape filter bank
    transformed_w = tf.transpose(transformed_w, [2, 3, 4, 5, 1, 6, 0])  # shape (n, n, n, ni, nti, no, nto)

    transformed_w = tf.reshape(transformed_w, [n, n, n, ni * nti, no * nto])  # shape (n, n, n, ni * nti, no * nto)

    return transformed_w


def transform_filter_2d_nhwc(w, flat_indices, shape_info, validate_indices=True):
    """
    Transform a set of filters defined on a split plane group G.
    This is the first step of the G-Conv. The user will typically not have to call this function directly.

    The input filter bank w has shape (n, n, nti * ni, no), where:
    n: the filter width and height
    ni: the number of input channels (note: the input feature map is assumed to have ni * nti number of channels)
    nti: the number of transformations in H (the stabilizer of the origin in the input space)
    For example, nti == 1 for images / functions on Z2, since only the identity translation leaves the origin invariant.
    Similarly, nti == 4 for the group p4, because there are 4 transformations in p4 (namely, the four rotations around
    the origin) that leave the origin in p4 (i.e. the identity transformation) fixed.
    no: the number of output channels (note: the G-Conv will actually create no * nto number of channels, see below.

    The index array has shape (nto, nti, n, n)
    Index arrays for various groups can be created with functions in groupy.gconv.make_gconv_indices.
    For example: flat_inds = flatten_indices(make_d4_z2_indices(ksize=3))

    The output filter bank transformed_w has shape (no * nto, ni * nti, n, n),
    (so there are nto times as many filters in the output as we had in the input w)
    """

    # The indexing is done using tf.gather. This function can only do integer indexing along the first axis.
    # We want to index the spatial and transformation axes of our filter, so we must flatten them into one axis.
    no, nto, ni, nti, n = shape_info
    w_flat = tf.reshape(w, [n * n * nti, ni, no])  # shape (n * n * nti, ni, no)

    # Do the transformation / indexing operation.
    transformed_w = tf.gather(w_flat, flat_indices,
                              validate_indices=validate_indices)  # shape (nto, nti, n, n, ni, no)

    # Put the axes in the right order, and collapse them to get a standard shape filter bank
    transformed_w = tf.transpose(transformed_w, [2, 3, 4, 1, 5, 0])  # shape (n, n, ni, nti, no, nto)
    transformed_w = tf.reshape(transformed_w, [n, n, ni * nti, no * nto])  # shape (n, n, ni * nti, no * nto)

    return transformed_w


def transform_filter_2d_nchw(w, flat_indices, shape_info, validate_indices=True):
    """
    Transform a set of filters defined on a split plane group G.
    This is the first step of the G-Conv. The user will typically not have to call this function directly.

    The input filter bank w has shape (no, ni * nti, n, n), where:
    no: the number of output channels (note: the G-Conv will actually create no * nto number of channels, see below.
    ni: the number of input channels (note: the input feature map is assumed to have ni * nti number of channels)
    nti: the number of transformations in H (the stabilizer of the origin in the input space)
    For example, nti == 1 for images / functions on Z2, since only the identity translation leaves the origin invariant.
    Similarly, nti == 4 for the group p4, because there are 4 transformations in p4 (namely, the four rotations around
    the origin) that leave the origin in p4 (i.e. the identity transformation) fixed.
    n: the filter width and height

    The index array has shape (nto, nti, n, n)
    Index arrays for various groups can be created with functions in groupy.gconv.make_gconv_indices.
    For example: flat_inds = flatten_indices(make_d4_z2_indices(ksize=3))

    The output filter bank transformed_w has shape (no * nto, ni * nti, n, n),
    (so there are nto times as many filters in the output as we had in the input w)
    """

    # The indexing is done using tf.gather. This function can only do integer indexing along the first axis.
    # We want to index the spatial and transformation axes of our filter, so we must flatten them into one axis,
    # and bring them to the first axis
    no, nto, ni, nti, n = shape_info
    w_flat = tf.transpose(tf.reshape(w, [no, ni, nti * n * n]), [2, 0, 1])  # shape (nti * n * n, no, ni)

    # Do the transformation / indexing operation.
    transformed_w = tf.gather(w_flat, flat_indices,
                              validate_indices=validate_indices)  # shape (nto, nti, n, n, no, ni)

    # Put the axes in the right order, and collapse them to get a standard-shape filter bank
    transformed_w = tf.transpose(transformed_w, [4, 0, 5, 1, 2, 3])  # shape (no, nto, ni, nti, n, n)
    transformed_w = tf.reshape(transformed_w, (no * nto, ni * nti, n, n))  # shape (no * nto, ni * nti, n, n)

    return transformed_w
