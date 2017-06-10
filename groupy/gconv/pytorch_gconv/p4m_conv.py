from groupy.gconv.pytorch_gconv.splitgconv2d import SplitGConv2D
from groupy.gconv.make_gconv_indices import make_d4_z2_indices, \
    make_d4_p4m_indices, flatten_indices


class P4MConvZ2(SplitGConv2D):

    @property
    def input_stabilizer_size(self):
        return 1

    @property
    def output_stabilizer_size(self):
        return 8

    def make_transformation_indices(self, ksize):
        return flatten_indices(make_d4_z2_indices(ksize=ksize))


class P4MConvP4M(SplitGConv2D):

    @property
    def input_stabilizer_size(self):
        return 8

    @property
    def output_stabilizer_size(self):
        return 8

    def make_transformation_indices(self, ksize):
        return flatten_indices(make_d4_p4m_indices(ksize=ksize))
