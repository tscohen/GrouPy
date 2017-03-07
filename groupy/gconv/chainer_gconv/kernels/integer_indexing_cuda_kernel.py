# Chainer implementation of the indexing kernels used in the G-conv

# These kernels take an input array containing filters, as well as an array of indices,
# and produce a set of transformed filters.

# The shapes are as follows
# Filter shape: (output_channels, input_channels, input_transforms, nu, nv)
# Index shape (one per coordinate t, u, v): (output_transforms, input_transforms, nu, nv)
# Result shape: (output_channels, output_transforms, input_channels, input_transforms, nu, nv)
# Note that there is one index array per group coordinate (t, u, v).

# A Z2 filter is viewed as a function on G that is right-invariant to the stabilizer of the origin H
# For example, for the P4 (rotation-translation) conv, the input image is a function on Z2,
# which we may think of as a function on P4 that is right-invariant to rotation.
# A right-rotation-invariant P4 function has the same value at (r, u, v) as it has at (r', u, v).
# Naturally, we don't store this invariant P4 function, but we store an array with a length-1 axis for the rotation
# coordinate.
# This is consistent with the numpy convention that lenght-1 axes get broadcast automatically.
# So for Z2 filters, we get the following shapes:
# Filter shape: (output_channels, input_channels, 1, nu, nv)
# Index shape (one per coordinate t, u, v): (output_transforms, 1, nu, nv)
# Result shape: (output_channels, output_transforms, input_channels, 1, nu, nv)


import cupy
from cupy.core.core import compile_with_cache

x = cupy.arange(2, dtype='f')  # WORKAROUND - currently, cupy compile_with_cache fails if no cupy code is executed first

# This computes input[..., T, U, V].swapaxes(1, 2)
_index_group_func_str = \
    """
    extern "C" __global__ void indexing_kernel(
        CArray<{0}, 5> input,
        CArray<int, 4> T,
        CArray<int, 4> U,
        CArray<int, 4> V,
        CArray<{0}, 6> output)
    {{
        CUPY_FOR(i, output.size()) {{

            const int* oshape = output.shape();
            const int* ostrides = output.strides();

            // The flat index i corresponds to the following multi-index in the output array:
            // (output_channel, output_transform, input_channel, input_transform, u, v)
            const int output_channel =   (sizeof({0}) * i / ostrides[0]) % oshape[0];
            const int output_transform = (sizeof({0}) * i / ostrides[1]) % oshape[1];
            const int input_channel =    (sizeof({0}) * i / ostrides[2]) % oshape[2];
            const int input_transform =  (sizeof({0}) * i / ostrides[3]) % oshape[3];
            const int u =                (sizeof({0}) * i / ostrides[4]) % oshape[4];
            const int v =                (sizeof({0}) * i / ostrides[5]) % oshape[5];

            int indexTUV[4] = {{output_transform, input_transform, u, v}};
            int index[5] = {{output_channel, input_channel, T[indexTUV], U[indexTUV], V[indexTUV]}};
            output[i] = input[index];
        }}
    }}
    """

_index_group_func_kernel32 = compile_with_cache(_index_group_func_str.format('float')).get_function('indexing_kernel')
_index_group_func_kernel64 = compile_with_cache(_index_group_func_str.format('double')).get_function('indexing_kernel')


def index_group_func_kernel(input, T, U, V, output):
    if input.dtype == 'float32':
        _index_group_func_kernel32.linear_launch(
            size=output.size,
            args=(input, T, U, V, output)
        )
    elif input.dtype == 'float64':
        _index_group_func_kernel64.linear_launch(
            size=output.size,
            args=(input, T, U, V, output)
        )
    else:
        raise ValueError()


_grad_index_group_func_str_double = \
    """
    // atomicAdd for doubles is not implemented in cuda, so have to add it here
    __device__ double my_atomicAdd(double* address, double val)
    {{
        unsigned long long int* address_as_ull =
                                  (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;

        do {{
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                                   __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        }} while (assumed != old);

        return __longlong_as_double(old);
    }}

    extern "C" __global__ void grad_indexing_kernel(
        CArray<{0}, 6> grad_output,
        CArray<int, 4> T,
        CArray<int, 4> U,
        CArray<int, 4> V,
        CArray<{0}, 5> grad_input)
    {{
        CUPY_FOR(i, grad_output.size()) {{

            const int* oshape = grad_output.shape();
            const int* ostrides = grad_output.strides();

            // The flat index i corresponds to the following multi-index in the output array:
            // (output_channel, output_transform, input_channel, input_transform, u, v)
            const int output_channel =   (sizeof({0}) * i / ostrides[0]) % oshape[0];
            const int output_transform = (sizeof({0}) * i / ostrides[1]) % oshape[1];
            const int input_channel =    (sizeof({0}) * i / ostrides[2]) % oshape[2];
            const int input_transform =  (sizeof({0}) * i / ostrides[3]) % oshape[3];
            const int u =                (sizeof({0}) * i / ostrides[4]) % oshape[4];
            const int v =                (sizeof({0}) * i / ostrides[5]) % oshape[5];

            int indexTUV[4] = {{output_transform, input_transform, u, v}};
            int index[5] = {{output_channel, input_channel, T[indexTUV], U[indexTUV], V[indexTUV]}};
            my_atomicAdd(&grad_input[index], grad_output[i]);
        }}
    }}
    """

_grad_index_group_func_str_float = \
    """
    extern "C" __global__ void grad_indexing_kernel(
        CArray<{0}, 6> grad_output,
        CArray<int, 4> T,
        CArray<int, 4> U,
        CArray<int, 4> V,
        CArray<{0}, 5> grad_input)
    {{
        CUPY_FOR(i, grad_output.size()) {{

            const int* oshape = grad_output.shape();
            const int* ostrides = grad_output.strides();

            // The flat index i corresponds to the following multi-index in the output array:
            // (output_channel, output_transform, input_channel, input_transform, u, v)
            const int output_channel =   (sizeof({0}) * i / ostrides[0]) % oshape[0];
            const int output_transform = (sizeof({0}) * i / ostrides[1]) % oshape[1];
            const int input_channel =    (sizeof({0}) * i / ostrides[2]) % oshape[2];
            const int input_transform =  (sizeof({0}) * i / ostrides[3]) % oshape[3];
            const int u =                (sizeof({0}) * i / ostrides[4]) % oshape[4];
            const int v =                (sizeof({0}) * i / ostrides[5]) % oshape[5];

            int indexTUV[4] = {{output_transform, input_transform, u, v}};
            int index[5] = {{output_channel, input_channel, T[indexTUV], U[indexTUV], V[indexTUV]}};
            atomicAdd(&grad_input[index], grad_output[i]);
        }}
    }}
    """

_grad_index_group_func_kernel32 = compile_with_cache(
    #_grad_index_group_func_str.format('float')
    _grad_index_group_func_str_float.format('float')
).get_function('grad_indexing_kernel')
_grad_index_group_func_kernel64 = compile_with_cache(
    #_grad_index_group_func_str.format('double')
    _grad_index_group_func_str_double.format('double')
).get_function('grad_indexing_kernel')


def grad_index_group_func_kernel(grad_output, T, U, V, grad_input):
    if grad_output.dtype == 'float32':
        _grad_index_group_func_kernel32.linear_launch(
            size=grad_output.size,
            args=(grad_output, T, U, V, grad_input)
        )
    elif grad_output.dtype == 'float64':
        _grad_index_group_func_kernel64.linear_launch(
            size=grad_output.size,
            args=(grad_output, T, U, V, grad_input)
        )
    else:
        raise ValueError()
