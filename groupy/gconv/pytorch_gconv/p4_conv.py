from groupy.gconv.pytorch_gconv.splitgconv2d import SplitGConv2D
from groupy.gconv.make_gconv_indices import make_c4_z2_indices, \
    make_c4_p4_indices, flatten_indices


class P4ConvZ2(SplitGConv2D):

    @property
    def input_stabilizer_size(self):
        return 1

    @property
    def output_stabilizer_size(self):
        return 4

    def make_transformation_indices(self, ksize):
        return flatten_indices(make_c4_z2_indices(ksize=ksize))


class P4ConvP4(SplitGConv2D):

    @property
    def input_stabilizer_size(self):
        return 4

    @property
    def output_stabilizer_size(self):
        return 4

    def make_transformation_indices(self, ksize):
        return flatten_indices(make_c4_p4_indices(ksize=ksize))
