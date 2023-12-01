module Galerkin

export
    galerkin_projection,
    galerkin_projection!,
    galerkin_matrix,
    galerkin_matrix!,
    galerkin_tensor,
    galerkin_tensor!

using ..BandedTensors
using ..BSplines
using ..DifferentialOps
using ..Recombinations: num_constraints
using ..Collocation: collocation_matrix  # for Documenter

using BandedMatrices
using FastGaussQuadrature: FastGaussQuadrature
using LinearAlgebra: Hermitian, ⋅
using SparseArrays
using StaticArrays

const DerivativeCombination{N} = Tuple{Vararg{Derivative,N}}

include("quadratures.jl")
include("projection.jl")  # galerkin_projection
include("linear.jl")      # galerkin_matrix
include("quadratic.jl")   # galerkin_tensor

end
