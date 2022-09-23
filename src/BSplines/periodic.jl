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

abstract type AbstractPeriodicVector{T} <: AbstractVector{T} end

Base.parent(ts::AbstractPeriodicVector) = ts.data
Base.checkbounds(::Type{Bool}, ts::AbstractPeriodicVector, i) = true  # all indices are accepted

# Modify `show` to make it clear that this is an "infinite" vector.
Base.show(io::IO, ts::AbstractPeriodicVector) =
    Base.show_vector(io, ts, "[..., ", ", ...]")

"""
    PeriodicKnots{T} <: AbstractVector{T}

Describes an infinite vector of knots with periodicity `L`.

---

    PeriodicKnots(ξs::AbstractVector{T}, L::Real, ::BSplineOrder{k})

Construct a periodic knot sequence with period `L` from breakpoints `ξs`.

The breakpoints should be in non-decreasing order.
They represent a single period, and must *not* include the endpoint.
In other words, the last point must be such that `ξs[end] < ξs[begin] + L`.

Note that the indices of the returned knots `ts` are offset with respect to the
input `ξs` according to `ts[i] = ξs[i + offset]` where `offset = k ÷ 2`.
"""
struct PeriodicKnots{
        T, k, Knots <: AbstractVector{T},
    } <: AbstractPeriodicVector{T}
    # Knots in a single period (length = N).
    # The knot vector is such that data[end] - data[begin] < period.
    data       :: Knots

    N          :: Int
    period     :: T
    boundaries :: NTuple{2, T}

    function PeriodicKnots(
            breaks::AbstractVector{T}, period::Real,
            ::BSplineOrder{k},
        ) where {T, k}
        k :: Int
        L = convert(T, period)
        a = first(breaks)
        if last(breaks) - a ≥ L
            throw(ArgumentError("the knot extent (t[end] - t[begin]) must be *strictly* smaller than the period L"))
        end
        boundaries = (a, a + L)
        Knots = typeof(breaks)
        N = length(breaks)
        new{T, k, Knots}(breaks, N, L, boundaries)
    end
end

boundaries(ts::PeriodicKnots) = ts.boundaries
period(ts::PeriodicKnots) = ts.period
order(::PeriodicKnots{T,k}) where {T,k} = k

# This offset is to make sure collocation matrices are diagonally-dominant.
index_offset(ts::PeriodicKnots) = order(ts) ÷ 2

# For consistency with regular B-spline bases, the length of the knot vector is N + k.
Base.length(ts::PeriodicKnots) = period_length(ts) + order(ts)
Base.size(ts::PeriodicKnots) = (length(ts),)

# This is such that ts[i] and ts[i + N] correspond to the same position due to
# periodicity.
period_length(ts::PeriodicKnots) = ts.N

_knot_zone(::PeriodicKnots, x) = 0

# Note that the returned zone is always 0 for periodic knots, meaning that any
# location `x` is a valid location where (B-)splines can be evaluated.
@inline function find_knot_interval(ts::PeriodicKnots, x::Real, ::Nothing)
    data = parent(ts)
    a, b = boundaries(ts)
    i = index_offset(ts)
    while x < a
        x += period(ts)
        i -= period_length(ts)
    end
    while x ≥ b
        x -= period(ts)
        i += period_length(ts)
    end
    i += searchsortedlast(data, x)
    zone = 0
    i, zone
end

@inline function Base.getindex(ts::PeriodicKnots, i::Int)
    data = parent(ts)
    x = zero(eltype(ts))
    i -= index_offset(ts)
    while i < firstindex(data)
        x -= period(ts)
        i += period_length(ts)
    end
    while i > lastindex(data)
        x += period(ts)
        i -= period_length(ts)
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

julia> ts = range(-1, 1; length = 11)[1:10]
-1.0:0.2:0.8

julia> B = PeriodicBSplineBasis(BSplineOrder(4), ts, L)
10-element PeriodicBSplineBasis of order 4, domain [-1.0, 1.0), period 2.0
 knots: [..., -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, ...]

julia> period(B)
2.0

julia> length(B)
10

julia> boundaries(B)
(-1.0, 1.0)

julia> B(-0.42)
(5, (0.12150000000000002, 0.6571666666666667, 0.22116666666666668, 0.00016666666666666563))

julia> B(-0.42 + 2)
(15, (0.12150000000000015, 0.6571666666666667, 0.22116666666666657, 0.00016666666666666674))
```
"""
struct PeriodicBSplineBasis{
        k, T, Knots <: PeriodicKnots{T},
    } <: AbstractBSplineBasis{k, T}
    N :: Int    # number of B-splines / knots in the "main" interval
    t :: Knots

    function PeriodicBSplineBasis(
            ord::BSplineOrder{k}, ts::AbstractVector{T},
            period::Real,
        ) where {k, T}
        k :: Integer
        if k <= 0
            throw(ArgumentError("B-spline order must be k ≥ 1"))
        end
        ts_per = PeriodicKnots(ts, period, ord)
        Knots = typeof(ts_per)
        N = length(parent(ts_per))
        new{k, T, Knots}(N, ts_per)
    end
end

@inline PeriodicBSplineBasis(k::Integer, args...; kwargs...) =
    PeriodicBSplineBasis(BSplineOrder(k), args...; kwargs...)

Base.parent(B::PeriodicBSplineBasis) = B

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
