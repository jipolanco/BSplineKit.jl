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
    evaluate_all,
    isindomain,
    nonzero_in_segment,
    has_parent_basis,
    find_knot_interval,
    basis_to_array_index,
    support

using Static: StaticInt, dynamic
using StaticArraysCore: SVector

using ..DifferentialOps

"""
    AbstractBSplineBasis{k,T}

Abstract type defining a B-spline basis, or more generally, a functional basis
defined from B-splines.

The basis is represented by a B-spline order `k` and a knot element type `T`.

---

    (B::AbstractBSplineBasis)(
        x::Real, [op = Derivative(0)], [T = float(typeof(x))];
        [ileft = nothing],
    ) -> (i, bs)

Evaluates all basis functions which are non-zero at `x`.

This is a convenience alias for `evaluate_all`.
See [`evaluate_all`](@ref) for details on optional arguments and on the returned values.

# Examples

```jldoctest
julia> B = BSplineBasis(BSplineOrder(4), -1:0.1:1)
23-element BSplineBasis of order 4, domain [-1.0, 1.0]
 knots: [-1.0, -1.0, -1.0, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4  …  0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0]

julia> i, bs = B(0.42)
(18, (0.0013333333333333268, 0.28266666666666657, 0.6306666666666667, 0.08533333333333339))

julia> sum(bs)
1.0

julia> bs[1] - B[i](0.42)
0.0

julia> bs[2] - B[i - 1](0.42)
-5.551115123125783e-17

julia> B(0.44; ileft = i)
(18, (0.01066666666666666, 0.4146666666666667, 0.5386666666666665, 0.03599999999999999))

julia> B(0.42, Float32)
(18, (0.0013333336f0, 0.28266668f0, 0.6306667f0, 0.085333325f0))

julia> B(0.42, Derivative(1))
(18, (0.19999999999999937, 6.4, -3.3999999999999977, -3.200000000000001))
```
"""
abstract type AbstractBSplineBasis{k,T} end

@inline (B::AbstractBSplineBasis)(args...; kws...) =
    evaluate_all(B, args...; kws...)

"""
    basis_to_array_index(B::AbstractBSplineBasis, axs, i::Int) -> Int

Converts from a basis index `i` (for basis function ``b_i``) to a valid index in
an array `A` with indices `axs`, where `axs` is typically the result of
`axes(A, dim)` for a given dimension `dim`.

This is mainly relevant for periodic bases, in which case `i` may be outside of
the "main" period of the basis.

It is *assumed* that `length(B) == length(axs)` (this is not checked by this function).
"""
basis_to_array_index(::AbstractBSplineBasis, axs, i::Int) = i

"""
    BSplineOrder(k::Integer)

Specifies the B-spline order `k`.
"""
struct BSplineOrder{k} end

@inline BSplineOrder(k::Integer) = BSplineOrder{Int(k)}()

"""
    has_parent_basis(::Type{<:AbstractBSplineBasis}) -> Bool
    has_parent_basis(::AbstractBSplineBasis) -> Bool

Trait determining whether a basis has a parent B-spline basis.

This is notably the case for `RecombinedBSplineBasis`, which are derived
from regular B-spline bases.
"""
function has_parent_basis end

has_parent_basis(::Type{<:AbstractBSplineBasis}) = false
has_parent_basis(B::AbstractBSplineBasis) = has_parent_basis(typeof(B))

include("basis.jl")
include("basis_function.jl")
include("knots.jl")
include("evaluate_all.jl")

include("periodic.jl")

end
