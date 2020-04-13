"""
    module BasisSplines

Approximate and interpolate functions using B-splines.

## Notation

Different definitions of the spline order are used in the literature and in
numerical packages.
Here we use the definition used by de Boor (2003), where a B-spline of order `k`
is a piecewise polynomial of degree `k - 1`.
Hence, for instance, cubic splines correspond to `k = 4`.
"""
module BasisSplines

export Collocation

export BSplineBasis, Spline
export knots, order
export augment_knots
export evaluate_bspline, evaluate_bspline!
export coefficients

using Reexport
using StaticArrays: MVector

include("knots.jl")
include("basis.jl")
include("spline.jl")

include("Collocation.jl")

@reexport using .Collocation

end # module
