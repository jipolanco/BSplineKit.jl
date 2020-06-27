module BSplineKit

export Collocation

export Derivative
export Spline, coefficients, integral
export galerkin_matrix, galerkin_matrix!, galerkin_tensor, galerkin_tensor!

using BandedMatrices: BandedMatrix
using FastGaussQuadrature: gausslegendre
using Reexport
using LinearAlgebra: Hermitian
using StaticArrays: MVector
using SparseArrays

import LinearAlgebra

include("BandedTensors.jl")
@reexport using .BandedTensors

include("DifferentialOps.jl")
using .DifferentialOps

include("BSplines/BSplines.jl")
@reexport using .BSplines

include("Recombinations/Recombinations.jl")
@reexport using .Recombinations
import .Recombinations: num_constraints  # used in galerkin

import .BSplines: basis, knots, order  # for spline.jl
include("spline.jl")

include("galerkin.jl")
include("Collocation.jl")

@reexport using .Collocation

end # module
