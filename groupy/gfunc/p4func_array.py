
import numpy as np
import groupy.garray.p4_array as p4a
from groupy.gfunc.gfuncarray import GFuncArray


class P4FuncArray(GFuncArray):

    def __init__(self, v, umin=None, umax=None, vmin=None, vmax=None):

        if umin is None or umax is None or vmin is None or vmax is None:
            if not (umin is None and umax is None and vmin is None and vmax is None):
                raise ValueError('Either all or none of umin, umax, vmin, vmax must equal None')

            # If (u, v) ranges are not given, determine them from the shape of v,
            # assuming the grid is centered.
            nu, nv = v.shape[-2:]

            hnu = nu // 2
            hnv = nv // 2

            umin = -hnu
            umax = hnu - (nu % 2 == 0)
            vmin = -hnv
            vmax = hnv - (nv % 2 == 0)

        self.umin = umin
        self.umax = umax
        self.vmin = vmin
        self.vmax = vmax

        i2g = p4a.meshgrid(
            r=p4a.r_range(0, 4),
            u=p4a.u_range(self.umin, self.umax + 1),
            v=p4a.v_range(self.vmin, self.vmax + 1)
        )

        super(P4FuncArray, self).__init__(v=v, i2g=i2g)

    def g2i(self, g):
        # TODO: check validity of indices and wrap / clamp if necessary
        # (or do this in a separate function, so that this function can be more easily tested?)

        gint = g.reparameterize('int').data.copy()
        gint[..., 1] -= self.umin
        gint[..., 2] -= self.vmin
        return gint


def tst():

    from groupy.garray.p4_array import P4Array, meshgrid, u_range, v_range, rotation, translation

    x = np.random.randn(4, 3, 3)
    c = meshgrid(u=u_range(-1, 2), v=v_range(-1, 2))

    f = P4FuncArray(v=x)

    g = rotation(1, center=(0, 0))
    li = f.left_translation_indices(g)
    lp = f.left_translation_points(g)

    # gfi = f[li]
    gfp = f(lp)
    gf = g * f
    gfi = f.v[li[..., 0], li[..., 1], li[..., 2]]

    return x, c, f, li, gf, gfp, gfi