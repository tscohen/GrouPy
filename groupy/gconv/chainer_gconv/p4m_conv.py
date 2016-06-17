from groupy.gconv.chainer_gconv.splitgconv2d import SplitGConv2D
from groupy.gconv.make_gconv_indices import make_d4_z2_indices, make_d4_p4m_indices


class P4MConvZ2(SplitGConv2D):

    input_stabilizer_size = 1
    output_stabilizer_size = 8

    def make_transformation_indices(self, ksize):
        return make_d4_z2_indices(ksize=ksize)


class P4MConvP4M(SplitGConv2D):

    input_stabilizer_size = 8
    output_stabilizer_size = 8

    def make_transformation_indices(self, ksize):
        return make_d4_p4m_indices(ksize=ksize)
