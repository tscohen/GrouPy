import groupy.garray.C4ht_array as c4ht
from groupy.gfunc.gfuncarray import GFuncArray


class C4htFuncArray(GFuncArray):
    def __init__(self, v, umin=None, umax=None, vmin=None, vmax=None, wmin=None, wmax=None):

        # TODO: error message
        if umin is None or umax is None or vmin is None or vmax is None:
            if not (umin is None and umax is None and vmin is None and vmax is None):
                raise ValueError('Either all or none of umin, umax, vmin, vmax must equal None')

            # If (u, v, w) ranges are not given, determine them from the shape of v,
            # assuming the grid is centered.
            nu, nv, nw = v.shape[-3:]

            hnu = nu // 2
            hnv = nv // 2
            hnw = nw // 2

            umin = -hnu
            umax = hnu - (nu % 2 == 0)
            vmin = -hnv
            vmax = hnv - (nv % 2 == 0)
            wmin = -hnw
            wmax = hnw - (nw % 2 == 0)

        self.umin = umin
        self.umax = umax + 1
        self.vmin = vmin
        self.vmax = vmax + 1
        self.wmin = wmin
        self.wmax = wmax + 1

        i2g = c4ht.meshgrid(
            minu=self.umin,
            maxu=self.umax,
            minv=self.vmin,
            maxv=self.vmax,
            minw=self.wmin,
            maxw=self.wmax
        )
        i2g = i2g.reshape(v.shape[-4:])

        super(C4htFuncArray, self).__init__(v=v, i2g=i2g)

    def g2i(self, g):
        gint = g.reparameterize('int').data.copy()
        gint[..., 2] -= self.umin
        gint[..., 3] -= self.vmin
        gint[..., 4] -= self.wmin

        # flat stabilizer: instead of (4, 2, ...) use (8, ...)
        gint[..., 1] += gint[..., 0] * 4    # multiply by 4 because 4 z rotations for each y
        gint = gint[..., 1:]

        return gint
