
# Chainer Functions for rotating filters or feature maps

from chainer import cuda
from chainer import function
from chainer.utils import type_check

from groupy.gconv.chainer_gconv.kernels.integer_indexing_cuda_kernel import grad_index_group_func_kernel
from groupy.gconv.chainer_gconv.kernels.integer_indexing_cuda_kernel import index_group_func_kernel


class TransformGFilter(function.Function):
    """
    Transform a set of filters defined on a split (symmorphic) plane group G.

    The input filterbank w has shape (no, ni, nt, n, n), where:
     no: the number of output channels
     ni: the number of input channels
     nt: the number of transformations in the stabilizer of the origin in G
     n: the filter width and height

    The output filterbank rotated_w has shape (no, nt, ni, nt, n, n), where a length-nt axis is added.
    The filter at rotated_w[o, t, i] is the filter w[o, i] transformed by t.
    """

    def __init__(self, inds):
        assert inds.dtype == 'int32'
        assert inds.ndim == 5
        self.T = inds[..., 0]
        self.U = inds[..., 1]
        self.V = inds[..., 2]

    def check_type_forward(self, in_types):
        w_type, = in_types
        type_check.expect(w_type.ndim == 5)
        # TODO: check x_type is float or double

    def forward_gpu(self, inputs):

        w, = inputs
        xp = cuda.get_array_module(w)
        och, ich, _, ny, nx = w.shape

        nto, nti = self.T.shape[:2]
        rotated_w = xp.empty((och, nto, ich, nti, ny, nx), dtype=w.dtype)

        index_group_func_kernel(
            input=w,
            T=self.T,
            U=self.U,
            V=self.V,
            output=rotated_w
        )

        return rotated_w,

    def backward_gpu(self, inputs, grad_output):

        w, = inputs
        grad_rotated_w, = grad_output
        xp = cuda.get_array_module(w)

        # Gradient must be initialized with zeros,
        # because the kernel accumulates the gradient instead of overwriting it
        grad_w = xp.zeros_like(w)

        grad_index_group_func_kernel(
            grad_output=grad_rotated_w,
            T=self.T,
            U=self.U,
            V=self.V,
            grad_input=grad_w
        )

        return grad_w,
