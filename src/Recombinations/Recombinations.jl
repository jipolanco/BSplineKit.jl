"""
    Recombinations

Basis recombination module.

Defines [`RecombinedBSplineBasis`](@ref) and [`RecombineMatrix`](@ref) types.
"""
module Recombinations

export RecombinedBSplineBasis, RecombineMatrix
export recombination_matrix, nzrows, constraints

import Base: @propagate_inbounds
import LinearAlgebra

using LinearAlgebra: UniformScaling, ldiv!, dot
using StaticArrays

using ..BSplines
using ..DifferentialOps

# These are redefined for RecombinedBSplineBasis.
import ..BSplines:
    boundaries, knots, order, evaluate, nonzero_in_segment, support, summary_basis

import ..Splines: Spline

include("matrices.jl")
include("bases.jl")

"""
    Spline(R::RecombinedBSplineBasis, coefs::AbstractVector)

Construct a [`Spline`](@ref) from a recombined B-spline basis and a vector of
coefficients in the recombined basis.
"""
Spline(R::RecombinedBSplineBasis, coefs) =
    Spline(parent(R), recombination_matrix(R) * coefs)

end
