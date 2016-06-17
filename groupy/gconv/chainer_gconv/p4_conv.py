from groupy.gconv.chainer_gconv.splitgconv2d import SplitGConv2D
from groupy.gconv.make_gconv_indices import make_c4_z2_indices, make_c4_p4_indices


class P4ConvZ2(SplitGConv2D):

    input_stabilizer_size = 1
    output_stabilizer_size = 4

    def make_transformation_indices(self, ksize):
        return make_c4_z2_indices(ksize=ksize)


class P4ConvP4(SplitGConv2D):

    input_stabilizer_size = 4
    output_stabilizer_size = 4

    def make_transformation_indices(self, ksize):
        return make_c4_p4_indices(ksize=ksize)