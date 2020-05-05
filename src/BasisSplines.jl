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

export AbstractBSplineBasis, BSplineBasis, RecombinedBSplineBasis
export RecombineMatrix, recombination_matrix
export Spline, BSpline, BSplineOrder, Derivative
export knots, order, coefficients, boundaries, order_bc
export augment_knots
export evaluate_bspline, evaluate_bspline!
export integral
export galerkin_matrix, galerkin_matrix!

using BandedMatrices: BandedMatrix
using FastGaussQuadrature: gausslegendre
using Reexport
using LinearAlgebra: Symmetric, Adjoint, Transpose
using SparseArrays
using StaticArrays

import LinearAlgebra

# We're transitioning to using the registered BSplines package...
import BSplines
using BSplines: Derivative
import BSplines: order, knots

include("BandedTensors.jl")
@reexport using .BandedTensors

"""
    BSplineOrder(k::Integer)

Specifies the B-spline order `k`.
"""
struct BSplineOrder{k} end

BSplineOrder(k::Integer) = BSplineOrder{k}()

include("knots.jl")
include("basis.jl")

const AnyBSplineBasis = Union{<:AbstractBSplineBasis, BSplines.BSplineBasis}

include("basis_function.jl")

include("recombine_matrix.jl")
include("recombined.jl")

include("spline.jl")

include("galerkin.jl")
include("Collocation.jl")

@reexport using .Collocation

end # module
