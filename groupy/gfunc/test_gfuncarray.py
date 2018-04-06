import numpy as np


def test_ot_func():
    from groupy.gfunc.otfunc_array import OtFuncArray
    import groupy.garray.O_array as o

    v = np.random.randn(2, 6, 24, 5, 5, 5)
    f = OtFuncArray(v=v)

    g = o.rand(size=(1,))
    h = o.rand(size=(1,))
    print(g)
    print(h)

    check_associative(g, h, f)
    check_identity(o, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)


def test_oht_func():
    from groupy.gfunc.ohtfunc_array import OhtFuncArray
    import groupy.garray.Oh_array as oh

    v = np.random.randn(2, 6, 48, 1, 1, 1)
    f = OhtFuncArray(v=v)

    g = oh.rand(size=(1,))
    h = oh.rand(size=(1,))

    check_associative(g, h, f)
    check_identity(oh, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)

def test_c4ht_func():
    from groupy.gfunc.c4htfunc_array import C4htFuncArray
    import groupy.garray.C4h_array as c4h

    v = np.random.randn(2, 6, 8, 3, 3, 3)
    f = C4htFuncArray(v=v)

    g = c4h.rand(size=(1,))
    h = c4h.rand(size=(1,))

    check_associative(g, h, f)
    check_identity(c4h, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)

def test_d4ht_func():
    from groupy.gfunc.d4htfunc_array import D4htFuncArray
    import groupy.garray.D4h_array as d4h

    v = np.random.randn(2, 6, 16, 3, 3, 3)
    f = D4htFuncArray(v=v)

    g = d4h.rand(size=(1,))
    h = d4h.rand(size=(1,))

    check_associative(g, h, f)
    check_identity(d4h, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)


def test_p4_func():
    from groupy.gfunc.p4func_array import P4FuncArray
    import groupy.garray.C4_array as c4a

    v = np.random.randn(2, 6, 4, 5, 5)
    f = P4FuncArray(v=v)

    g = c4a.rand(size=(1,))
    h = c4a.rand(size=(1,))

    check_associative(g, h, f)
    check_identity(c4a, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)


def test_p4m_func():
    from groupy.gfunc.p4mfunc_array import P4MFuncArray
    import groupy.garray.D4_array as d4a

    v = np.random.randn(2, 6, 8, 5, 5)
    f = P4MFuncArray(v=v)

    g = d4a.rand(size=(1,))
    h = d4a.rand(size=(1,))

    check_associative(g, h, f)
    check_identity(d4a, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)


def test_z2_func():
    from groupy.gfunc.z2func_array import Z2FuncArray
    import groupy.garray.C4_array as c4a
    import groupy.garray.C4_array as d4a

    v = np.random.randn(2, 6, 5, 5)
    f = Z2FuncArray(v=v)

    g = c4a.rand(size=(1,))
    h = c4a.rand(size=(1,))
    check_associative(g, h, f)
    check_identity(c4a, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)

    g = d4a.rand(size=(1,))
    h = d4a.rand(size=(1,))
    check_associative(g, h, f)
    check_identity(d4a, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)


def test_z3_func():
    from groupy.gfunc.z3func_array import Z3FuncArray
    import groupy.garray.O_array as o
    import groupy.garray.Oh_array as oh

    v = np.random.randn(2, 6, 5, 5, 5)
    f = Z3FuncArray(v=v)

    g = o.rand(size=(1,))
    h = o.rand(size=(1,))
    check_associative(g, h, f)
    check_identity(o, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)

    v = np.random.randn(2, 6, 5, 5, 5)
    f = Z3FuncArray(v=v)

    g = oh.rand(size=(1,))
    h = oh.rand(size=(1,))
    check_associative(g, h, f)
    check_identity(oh, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)


def check_associative(g, h, f):
    gh = g * h
    hf = h * f
    gh_f = gh * f
    g_hf = g * hf
    assert (gh_f.v == g_hf.v).all()


def check_identity(garray_module, a):
    e = garray_module.identity()
    assert ((e * a).v == a.v).all()


def check_invertible(g, f):
    a = g.inv()
    b = (g * f)
    c = a * b
    assert (c.v == f.v).all()


def check_i2g_g2i_invertible(f):
    i2g = f.i2g
    i = f.g2i(i2g)
    inds = [i[..., j] for j in range(i.shape[-1])]
    assert (i2g[inds] == i2g).all()
