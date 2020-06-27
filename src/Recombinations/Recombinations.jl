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
    boundaries, knots, order, evaluate, support

include("matrices.jl")
include("bases.jl")

end
