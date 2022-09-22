export PeriodicBSplineBasis,
       PeriodicKnots,
       period

"""
    period(B::PeriodicBSplineBasis) -> Real
    period(ts::PeriodicKnots) -> Real

Returns the period `L` associated to a periodic B-spline basis.
"""
function period end

## ========================================================================== ##
## PeriodicKnots                                                              ##
## ========================================================================== ##

"""
    PeriodicKnots{T} <: AbstractVector{T}

Describes an infinite vector of knots with periodicity `L`.

Note that the vector has an effective length `N` associated to a single period,
but it is possible to index it outside of this "main" interval.
"""
struct PeriodicKnots{T, Knots <: AbstractVector{T}} <: AbstractVector{T}
    # Knots in a single period (length = N).
    # The knot vector is such that data[end] - data[begin] < period.
    data       :: Knots

    N          :: Int
    period     :: T
    boundaries :: NTuple{2, T}

    function PeriodicKnots(
            ts::AbstractVector{T}, period::Real,
        ) where {T}
        L = convert(T, period)
        a = first(ts)
        if last(ts) - a ≥ L
            throw(ArgumentError("the knot extent (t[end] - t[begin]) must be *strictly* smaller than the period L"))
        end
        boundaries = (a, a + L)
        Knots = typeof(ts)
        N = length(ts)
        new{T, Knots}(ts, N, L, boundaries)
    end
end

Base.parent(ts::PeriodicKnots) = ts.data
boundaries(ts::PeriodicKnots) = ts.boundaries
period(ts::PeriodicKnots) = ts.period

Base.axes(ts::PeriodicKnots) = axes(parent(ts))
Base.length(ts::PeriodicKnots) = ts.N
Base.checkbounds(::Type{Bool}, ts::PeriodicKnots, i) = true  # all indices are accepted

_knot_zone(::PeriodicKnots, x) = 0

# Note that the returned zone is always 0 for periodic knots, meaning that any
# location `x` is a valid location where (B-)splines can be evaluated.
@inline function find_knot_interval(ts::PeriodicKnots, x::Real, ::Nothing)
    data = parent(ts)
    a, b = boundaries(ts)
    i = 0
    while x < a
        x += period(ts)
        i -= length(ts)
    end
    while x ≥ b
        x -= period(ts)
        i += length(ts)
    end
    i += searchsortedlast(data, x)
    zone = 0
    i, zone
end

@inline function Base.getindex(ts::PeriodicKnots, i::Int)
    data = parent(ts)
    x = zero(eltype(ts))
    while i < firstindex(data)
        i += length(ts)
        x -= period(ts)
    end
    while i > lastindex(data)
        i -= length(ts)
        x += period(ts)
    end
    @inbounds x + data[i]
end

## ========================================================================== ##
## PeriodicBSplineBasis                                                       ##
## ========================================================================== ##

"""
    PeriodicBSplineBasis{k, T}

B-spline basis for splines of order `k` and knot element type `T <: Real`.

The basis is defined by a set of knots and by the B-spline order.

---

    PeriodicBSplineBasis(order::BSplineOrder{k}, ts::AbstractVector, L::Real)

Create periodic B-spline basis of order `k` with knots `ts` and period `L`.

The knot vector `ts` must be in non-decreasing order, and must satisfy
`(ts[end] - ts[begin]) < L`.
In other words, it must *not* include the endpoint `ts[begin] + L`.

# Examples

Create B-spline basis on periodic domain with period ``L = 2``.

```jldoctest
julia> L = 2;

julia> ts = range(-1, 1; length = 21)[1:20]
-1.0:0.1:0.9

julia> B = PeriodicBSplineBasis(BSplineOrder(4), ts, L)
20-element PeriodicBSplineBasis of order 4, domain [-1.0, 1.0), period 2.0
 knots: [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

julia> period(B)
2.0

julia> length(B)
20

julia> boundaries(B)
(-1.0, 1.0)

julia> B(-0.42)
(6, (0.08533333333333341, 0.6306666666666667, 0.28266666666666657, 0.0013333333333333263))

julia> B(-0.42 + 2)
(26, (0.08533333333333351, 0.6306666666666669, 0.28266666666666623, 0.0013333333333333346))
```
"""
struct PeriodicBSplineBasis{
        k, T, Knots <: PeriodicKnots{T},
    } <: AbstractBSplineBasis{k, T}
    N :: Int    # number of B-splines / knots in the "main" interval
    t :: Knots

    function PeriodicBSplineBasis(
            ::BSplineOrder{k}, ts::AbstractVector{T},
            period::Real,
        ) where {k, T}
        k :: Integer
        if k <= 0
            throw(ArgumentError("B-spline order must be k ≥ 1"))
        end
        ts_per = PeriodicKnots(ts, period)
        Knots = typeof(ts_per)
        N = length(parent(ts_per))
        new{k, T, Knots}(N, ts_per)
    end
end

@inline PeriodicBSplineBasis(k::Integer, args...; kwargs...) =
    PeriodicBSplineBasis(BSplineOrder(k), args...; kwargs...)

Base.length(B::PeriodicBSplineBasis) = B.N
knots(B::PeriodicBSplineBasis) = B.t
boundaries(B::PeriodicBSplineBasis) = boundaries(knots(B))
period(B::PeriodicBSplineBasis) = period(knots(B))

function summary_basis(io, B::PeriodicBSplineBasis)
    a, b = boundaries(B)
    print(io, length(B), "-element ", nameof(typeof(B)))
    print(io, " of order ", order(B), ", domain [", a, ", ", b, ")")
    print(io, ", period ", period(B))
    nothing
end

Base.:(==)(A::PeriodicBSplineBasis, B::PeriodicBSplineBasis) =
    A === B ||
    order(A) == order(B) && knots(A) == knots(B)

@propagate_inbounds function evaluate_all(
        B::PeriodicBSplineBasis, x::Real, op::Derivative, ::Type{T};
        kws...,
    ) where {T <: Number}
    _evaluate_all(knots(B), x, BSplineOrder(order(B)), op, T; kws...)
end
