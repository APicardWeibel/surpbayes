"""Custom numba typing abbrev shared through the package"""

import numba as nb

i = nb.int64
f = nb.float64
b = nb.boolean

i1D = nb.int64[:]
i2D = nb.int64[:, :]

f1Dru = nb.types.Array(dtype=nb.float64, ndim=1, layout="A", readonly=True)
f2Dru = nb.types.Array(dtype=nb.float64, ndim=2, layout="A", readonly=True)

f1D = nb.float64[:]
f1D_C = nb.types.Array(dtype=nb.float64, ndim=1, layout="C")
f2D = nb.float64[:, :]
f2D_C = nb.types.Array(dtype=nb.float64, ndim=2, layout="C")
f3D = nb.float64[:, :, :]

b1D = nb.types.Array(dtype=nb.boolean, ndim=1, layout="A")
Tuple = nb.types.Tuple
UTuple = nb.types.UniTuple
string = nb.types.unicode_type
