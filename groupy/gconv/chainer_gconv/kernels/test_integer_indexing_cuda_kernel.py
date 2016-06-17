
from groupy.gconv.chainer_gconv.kernels.integer_indexing_cuda_kernel import index_group_func_kernel


def test_index_group_func():
    import numpy as np
    import cupy as cp
    from chainer import cuda
    input = np.random.randn(2, 3, 4, 5, 6)
    I = np.random.randint(0, 4, (7, 8, 9, 10))
    J = np.random.randint(0, 5, (7, 8, 9, 10))
    K = np.random.randint(0, 6, (7, 8, 9, 10))

    output = input[..., I, J, K].swapaxes(1, 2)

    cpoutput = cp.zeros(output.shape)
    cpinput = cuda.to_gpu(input)
    cpI = cuda.to_gpu(I)
    cpJ = cuda.to_gpu(J)
    cpK = cuda.to_gpu(K)

    index_group_func_kernel(cpinput, cpI, cpJ, cpK, cpoutput)

    cpoutput = cuda.to_cpu(cpoutput)

    error = np.abs(cpoutput - output).sum()
    print error
    assert np.isclose(error, 0.)

