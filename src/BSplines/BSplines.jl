"""
    BSplines

Module defining B-spline bases and B-spline related functions.
"""
module BSplines

export
    AbstractBSplineBasis,
    BasisFunction,
    BSplineBasis,
    BSplineOrder,
    boundaries,
    order,
    knots,
    basis,
    evaluate,
    evaluate!,
    support

using ..DifferentialOps

"""
    AbstractBSplineBasis{k,T}

Abstract type defining a B-spline basis, or more generally, a functional basis
defined from B-splines.

The basis is represented by a B-spline order `k` and a knot element type `T`.
"""
abstract type AbstractBSplineBasis{k,T} end

"""
    BSplineOrder(k::Integer)

Specifies the B-spline order `k`.
"""
struct BSplineOrder{k} end

@inline BSplineOrder(k::Integer) = BSplineOrder{Int(k)}()

include("basis.jl")
include("basis_function.jl")
include("knots.jl")

end
