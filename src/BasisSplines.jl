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

export BSplineBasis, Spline, BSpline, BSplineOrder, Derivative
export knots, order, coefficients
export augment_knots
export evaluate_bspline, evaluate_bspline!
export integral
export galerkin_matrix, galerkin_matrix!

using BandedMatrices: BandedMatrix
using FastGaussQuadrature: gausslegendre
using Reexport
using LinearAlgebra: Symmetric
using SparseArrays
using StaticArrays: MVector

# We're transitioning to using the registered BSplines package...
import BSplines
using BSplines: Derivative

"""
    BSplineOrder(k::Integer)

Specifies the B-spline order `k`.
"""
struct BSplineOrder{k} end

BSplineOrder(k::Integer) = BSplineOrder{k}()

include("knots.jl")
include("basis.jl")
include("spline.jl")
include("galerkin.jl")

include("Collocation.jl")

@reexport using .Collocation

end # module
