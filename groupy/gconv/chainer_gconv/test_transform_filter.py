import cupy as cp
import numpy as np
from chainer import Variable
from chainer import cuda

# TODO: check that sequential transforms match the application of a composition of transforms: g (h f) = (gh) f
# TODO: check that applying a transformation and its inverse leaves the signal invariant g^-1 (g f) = f

from groupy.gconv.make_gconv_indices import make_c4_z2_indices, make_c4_p4_indices,\
    make_d4_z2_indices, make_d4_p4m_indices
from groupy.gconv.chainer_gconv.transform_filter import TransformGFilter


def test_transform_grad():
    for dtype, toll in [('float32', 1e-3), ('float64', 1e-10)]:
        check_transform_c4_z2_grad(dtype, toll)
        check_transform_c4_p4_grad(dtype, toll)
        check_transform_d4_p4m_grad(dtype, toll)
        check_transform_d4_z2_grad(dtype, toll)


def check_transform_c4_z2_grad(dtype='float64', toll=1e-10):
    inds = make_c4_z2_indices(ksize=5)
    w = cp.random.randn(3, 2, 1, 5, 5)
    check_transform_grad(inds, w, TransformGFilter, dtype, toll)


def check_transform_c4_p4_grad(dtype='float64', toll=1e-10):
    inds = make_c4_p4_indices(ksize=3)
    w = cp.random.randn(1, 2, 4, 3, 3)
    check_transform_grad(inds, w, TransformGFilter, dtype, toll)


def check_transform_d4_z2_grad(dtype='float64', toll=1e-10):
    inds = make_d4_z2_indices(ksize=5)
    w = cp.random.randn(3, 2, 1, 5, 5)
    check_transform_grad(inds, w, TransformGFilter, dtype, toll)


def check_transform_d4_p4m_grad(dtype='float64', toll=1e-10):
    inds = make_d4_p4m_indices(ksize=3)
    w = cp.random.randn(1, 2, 8, 3, 3)
    check_transform_grad(inds, w, TransformGFilter, dtype, toll)


def check_transform_grad(inds, w, transformer, dtype, toll):
    from chainer import gradient_check

    inds = cuda.to_gpu(inds)

    W = Variable(w.astype(dtype))
    R = transformer(inds)

    RW = R(W)

    RW.grad = cp.random.randn(*RW.data.shape).astype(dtype)
    RW.backward(retain_grad=True)

    func = RW.creator
    fn = lambda: func.forward((W.data,))
    gW, = gradient_check.numerical_grad(fn, (W.data,), (RW.grad,))

    gan = cuda.to_cpu(gW)
    gat = cuda.to_cpu(W.grad)

    relerr = np.max(np.abs(gan - gat) / np.maximum(np.abs(gan), np.abs(gat)))

    print dtype, toll, relerr
    assert relerr < toll
