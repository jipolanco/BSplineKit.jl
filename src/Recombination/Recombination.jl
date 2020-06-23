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

# TODO put more stuff inside modules...
import ..BasisSplines:
    AbstractBSplineBasis,
    # These will be in a DifferentialOps module:
    AbstractDifferentialOp, get_orders, max_order,
    # These are redefined for RecombinedBSplineBasis:
    boundaries, knots, order, evaluate_bspline, support

include("matrices.jl")
include("bases.jl")

end
