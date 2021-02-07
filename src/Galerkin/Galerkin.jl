module Galerkin

export
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
using FastGaussQuadrature: gausslegendre
using LinearAlgebra: Hermitian
using SparseArrays
using StaticArrays

const DerivativeCombination{N} = Tuple{Vararg{Derivative,N}}

include("quadratures.jl")
include("linear.jl")     # galerkin_matrix
include("quadratic.jl")  # galerkin_tensor

end
