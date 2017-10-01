import numpy as np
import torch
from torch.autograd import Variable
from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4, P4MConvZ2, P4MConvP4M


def test_p4_net_equivariance():
    from groupy.gfunc import Z2FuncArray, P4FuncArray
    import groupy.garray.C4_array as c4a

    im = np.random.randn(1, 1, 11, 11).astype('float32')
    check_equivariance(
        im=im,
        layers=[
            P4ConvZ2(in_channels=1, out_channels=2, kernel_size=3),
            P4ConvP4(in_channels=2, out_channels=3, kernel_size=3)
        ],
        input_array=Z2FuncArray,
        output_array=P4FuncArray,
        point_group=c4a,
    )


def test_p4m_net_equivariance():
    from groupy.gfunc import Z2FuncArray, P4MFuncArray
    import groupy.garray.D4_array as d4a

    im = np.random.randn(1, 1, 11, 11).astype('float32')
    check_equivariance(
        im=im,
        layers=[
            P4MConvZ2(in_channels=1, out_channels=2, kernel_size=3),
            P4MConvP4M(in_channels=2, out_channels=3, kernel_size=3)
        ],
        input_array=Z2FuncArray,
        output_array=P4MFuncArray,
        point_group=d4a,
    )


def test_g_z2_conv_equivariance():
    from groupy.gfunc import Z2FuncArray, P4FuncArray, P4MFuncArray
    import groupy.garray.C4_array as c4a
    import groupy.garray.D4_array as d4a

    im = np.random.randn(1, 1, 11, 11).astype('float32')
    check_equivariance(
        im=im,
        layers=[P4ConvZ2(1, 2, 3)],
        input_array=Z2FuncArray,
        output_array=P4FuncArray,
        point_group=c4a,
    )

    check_equivariance(
        im=im,
        layers=[P4MConvZ2(1, 2, 3)],
        input_array=Z2FuncArray,
        output_array=P4MFuncArray,
        point_group=d4a,
    )


def test_p4_p4_conv_equivariance():
    from groupy.gfunc import P4FuncArray
    import groupy.garray.C4_array as c4a

    im = np.random.randn(1, 1, 4, 11, 11).astype('float32')
    check_equivariance(
        im=im,
        layers=[P4ConvP4(1, 2, 3)],
        input_array=P4FuncArray,
        output_array=P4FuncArray,
        point_group=c4a,
    )


def test_p4m_p4m_conv_equivariance():
    from groupy.gfunc import P4MFuncArray
    import groupy.garray.D4_array as d4a

    im = np.random.randn(1, 1, 8, 11, 11).astype('float32')
    check_equivariance(
        im=im,
        layers=[P4MConvP4M(1, 2, 3)],
        input_array=P4MFuncArray,
        output_array=P4MFuncArray,
        point_group=d4a,
    )


def check_equivariance(im, layers, input_array, output_array, point_group):

    # Transform the image
    f = input_array(im)
    g = point_group.rand()
    gf = g * f
    im1 = gf.v
    # Apply layers to both images
    im = Variable(torch.Tensor(im))
    im1 = Variable(torch.Tensor(im1))

    fmap = im
    fmap1 = im1
    for layer in layers:
        fmap = layer(fmap)
        fmap1 = layer(fmap1)

    # Transform the computed feature maps
    fmap1_garray = output_array(fmap1.data.numpy())
    r_fmap1_data = (g.inv() * fmap1_garray).v

    fmap_data = fmap.data.numpy()
    assert np.allclose(fmap_data, r_fmap1_data, rtol=1e-5, atol=1e-3)
