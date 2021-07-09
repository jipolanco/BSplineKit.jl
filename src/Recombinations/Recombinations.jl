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
using ..Splines

# These are redefined for RecombinedBSplineBasis.
import ..BSplines:
    boundaries, knots, order, evaluate, nonzero_in_segment, support, summary_basis

include("matrices.jl")
include("bases.jl")
include("splines.jl")

end
