using BSplineKit

using BandedMatrices
using LinearAlgebra
using Random
using SparseArrays
using Test

using BSplineKit:
    AbstractDifferentialOp,
    DifferentialOpSum,
    LeftNormal,
    RightNormal

using BSplineKit.Recombinations:
    NoUniqueSolutionError,
    num_constraints,
    num_recombined

# Chebyshev (Gauss-Lobatto) points.
gauss_lobatto_points(N) = [-cos(π * n / N) for n = 0:N]

include("static.jl")
include("diffops.jl")
include("knots.jl")
include("bsplines.jl")
include("splines.jl")
include("periodic.jl")
include("natural.jl")
include("interpolation.jl")
include("smoothing.jl")
include("extrapolation.jl")
include("approximation.jl")
include("recombination.jl")
include("collocation.jl")
include("galerkin.jl")
include("banded_tensors.jl")
include("airy.jl")
