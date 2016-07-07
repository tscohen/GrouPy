import chainer.functions as F


def plane_group_spatial_max_pooling(x, ksize, stride=None, pad=0, cover_all=True, use_cudnn=True):
    xs = x.data.shape
    x = F.reshape(x, (xs[0], xs[1] * xs[2], xs[3], xs[4]))
    x = F.max_pooling_2d(x, ksize, stride, pad, cover_all, use_cudnn)
    x = F.reshape(x, (xs[0], xs[1], xs[2], x.data.shape[2], x.data.shape[3]))
    return x
