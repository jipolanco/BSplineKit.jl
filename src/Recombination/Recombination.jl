"""
    Recombination

Basis recombination module.

Defines [`RecombinedBSplineBasis`](@ref) and [`RecombineMatrix`](@ref) types.
"""
module Recombination

export RecombinedBSplineBasis, RecombineMatrix
export recombination_matrix, nzrows, constraints

import Base: @propagate_inbounds
import LinearAlgebra

using LinearAlgebra: UniformScaling, ldiv!
using StaticArrays

using ..BasisSplines
using ..BasisSplines.DifferentialOps

import ..BasisSplines:
    AbstractBSplineBasis,
    # These are redefined for RecombinedBSplineBasis:
    boundaries, knots, order, evaluate_bspline, support

include("matrices.jl")
include("bases.jl")

end
