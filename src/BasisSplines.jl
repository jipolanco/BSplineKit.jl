module BasisSplines

export Collocation

export AbstractBSplineBasis, BSplineBasis, RecombinedBSplineBasis
export RecombineMatrix, recombination_matrix
export Spline, BSpline, BSplineOrder, Derivative
export knots, order, coefficients, boundaries, order_bc
export augment_knots
export evaluate_bspline, evaluate_bspline!
export integral
export galerkin_matrix, galerkin_matrix!, galerkin_tensor

using BandedMatrices: BandedMatrix
using FastGaussQuadrature: gausslegendre
using Reexport
using LinearAlgebra: Symmetric, Adjoint, Transpose, UniformScaling
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
