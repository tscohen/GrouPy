import chainer.functions as F
from chainer import as_variable


def plane_group_spatial_max_pooling(x, ksize, stride=None, pad=0, cover_all=True, use_cudnn=True):
    # tuple gets passed in; wrap it in a chainer.Variable
    x = as_variable(x)
    xs = x.data.shape
    x = F.reshape(x, (xs[0], xs[1] * xs[2], xs[3], xs[4]))
    # chainer 5.1.0 does not use use_cudnn
    x = F.max_pooling_2d(x, ksize, stride, pad, cover_all)
    x = F.reshape(x, (xs[0], xs[1], xs[2], x.data.shape[2], x.data.shape[3]))
    return x