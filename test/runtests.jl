using BSplineKit

using BandedMatrices
using LinearAlgebra
using Random
using SparseArrays
using Test

import BSplineKit:
    AbstractDifferentialOp,
    DifferentialOpSum,
    LeftNormal,
    RightNormal

import BSplineKit.Recombinations:
    NoUniqueSolutionError,
    num_constraints,
    num_recombined

# Chebyshev (Gauss-Lobatto) points.
gauss_lobatto_points(N) = [-cos(π * n / N) for n = 0:N]

include("diffops.jl")
include("knots.jl")
include("splines.jl")
include("recombination.jl")
include("collocation.jl")
include("galerkin.jl")
include("banded_tensors.jl")
include("airy.jl")
