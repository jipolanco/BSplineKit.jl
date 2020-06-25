module BSplineKit

export Collocation

export AbstractBSplineBasis, BSplineBasis, RecombinedBSplineBasis
export Spline, BSpline, BSplineOrder, Derivative
export knots, order, coefficients, boundaries
export augment_knots
export evaluate_bspline, evaluate_bspline!
export integral
export galerkin_matrix, galerkin_matrix!, galerkin_tensor

using BandedMatrices: BandedMatrix
using FastGaussQuadrature: gausslegendre
using Reexport
using LinearAlgebra: Hermitian
using StaticArrays: MVector
using SparseArrays

import LinearAlgebra

include("BandedTensors.jl")
@reexport using .BandedTensors

"""
    BSplineOrder(k::Integer)

Specifies the B-spline order `k`.
"""
struct BSplineOrder{k} end

@inline BSplineOrder(k::Integer) = BSplineOrder{k}()

include("DifferentialOps.jl")
using .DifferentialOps

include("knots.jl")
include("basis.jl")
include("basis_function.jl")

include("Recombinations/Recombinations.jl")
@reexport using .Recombinations
import .Recombinations: num_constraints  # used in galerkin

include("spline.jl")

include("galerkin.jl")
include("Collocation.jl")

@reexport using .Collocation

end # module
