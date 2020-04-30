"""
    AbstractBSplineBasis{k,T}

Abstract type defining a B-spline basis, or more generally, a functional basis
defined from B-splines.

The basis is represented by a B-spline order `k` and an element type `T`.
"""
abstract type AbstractBSplineBasis{k,T} end

"""
    BSplineBasis{k}

B-spline basis for splines of order `k`.

The basis is defined by a set of knots and by the B-spline order.

---

    BSplineBasis(k::Union{BSplineOrder,Integer}, knots::Vector; augment=true)

Create B-spline basis of order `k` with the given knots.

If `augment=true` (default), knots will be "augmented" so that both knot ends
have multiplicity `k`. See also [`augment_knots`](@ref).
"""
struct BSplineBasis{k,T} <: AbstractBSplineBasis{k,T}
    N :: Int             # number of B-splines ("resolution")
    t :: Vector{T}       # knots (length = N + k)
    function BSplineBasis(::BSplineOrder{k}, knots::AbstractVector{T};
                          augment=true) where {k,T}
        k :: Integer
        if k <= 0
            throw(ArgumentError("B-spline order must be k ≥ 1"))
        end
        t = augment ? augment_knots(knots, k) : knots
        N = length(t) - k
        new{k, T}(N, t)
    end
end

@inline BSplineBasis(k::Integer, args...; kwargs...) =
    BSplineBasis(BSplineOrder(k), args...; kwargs...)

# Make BSplineBasis behave as scalar when broadcasting.
Broadcast.broadcastable(B::AbstractBSplineBasis) = Ref(B)

"""
    length(g::BSplineBasis)

Returns the number of B-splines composing a spline.
"""
Base.length(g::BSplineBasis) = g.N
Base.size(g::AbstractBSplineBasis) = (length(g), )
Base.parent(g::BSplineBasis) = g

"""
    boundaries(B::AbstractBSplineBasis)

Returns `(xmin, xmax)` tuple with the boundaries of the domain supported by the
basis.
"""
function boundaries(B::BSplineBasis)
    k = order(B)
    N = length(B)
    t = knots(B)
    t[k], t[N + 1]
end

"""
    knots(g::BSplineBasis)
    knots(g::Spline)

Returns the knots of the spline grid.
"""
knots(g::BSplineBasis) = g.t

"""
    order(::Type{BSplineBasis}) -> Int
    order(::Type{Spline}) -> Int
    order(::BSplineOrder) -> Int

Returns order of B-splines as an integer.
"""
order(::Type{<:BSplineBasis{k}}) where {k} = k
order(b::BSplineBasis) = order(typeof(b))
order(::BSplineOrder{k}) where {k} = k

# TODO rename to BasisFunction?
"""
    BSpline{B <: AbstractBSplineBasis}

Describes a single basis function.

---

    BSpline(basis::AbstractBSplineBasis, i::Int, [T = Float64])

Construct i-th basis function of the given basis.

The constructed function can be evaluated as `b(x)`, returning a value of type
`T`.
"""
struct BSpline{Basis <: AbstractBSplineBasis, T}
    basis :: Basis
    i     :: Int
    function BSpline(b::AbstractBSplineBasis, i, ::Type{T} = Float64) where {T}
        Basis = typeof(b)
        new{Basis, T}(b, i)
    end
end

basis(b::BSpline) = b.basis
knots(b::BSpline) = knots(basis(b))
order(b::BSpline) = order(basis(b))
Base.eltype(::Type{BSpline{B,T}}) where {B,T} = T

"""
    support(b::BSpline) -> UnitRange{Int}

Get range of knots supported by the B-spline.

Returns the range `i:j` if the B-spline is non-zero between knots `t[i]` and
`t[j]`.
"""
function support(b::BSpline{<:BSplineBasis})
    k = order(b)
    i = b.i
    i:(i + k)
end

"""
    common_support(b1::BSpline, b2::BSpline, ...) -> UnitRange{Int}

Get range of knots commonly supported by different B-splines.

If the supports don't intersect, an empty range is returned (e.g. `6:5`),
following the behaviour of `intersect`. The lack of intersection can be checked
using `isempty`, which returns `true` for such a range.
"""
common_support(bs::Vararg{BSpline}) = ∩(support.(bs)...)

"""
    (b::BSpline)(x, [deriv = Derivative(0)])

Evaluate B-spline at coordinate `x`.

To evaluate a derivative, pass the `deriv` parameter with the wanted
differentiation order.
"""
(b::BSpline)(x, deriv=Derivative(0)) =
    evaluate_bspline(basis(b), b.i, x, deriv, eltype(b))

"""
    evaluate_bspline(
        B::AbstractBSplineBasis, i::Integer, x, [deriv=Derivative(0)], [T=Float64]
    )

Evaluate i-th B-spline in the given basis at `x` (can be a coordinate or a
vector of coordinates).

The `N`-th derivative of bᵢ(x) may be evaluated by passing `Derivative(N)`.

See also [`evaluate_bspline!`](@ref).
"""
function evaluate_bspline(B::BSplineBasis, i::Integer, x::Real,
                          deriv::Derivative = Derivative(0),
                          ::Type{T} = Float64) where {T}
    N = length(B)
    if !(1 <= i <= N)
        throw(DomainError(i, "B-spline index must be in 1:$N"))
    end
    k = order(B)
    t = knots(B)
    evaluate_bspline_diff(deriv, BSplineOrder(k), t, i, x, T)
end

# No derivative
evaluate_bspline_diff(::Derivative{0}, ::BSplineOrder{k}, t, i, x,
                      ::Type{T}) where {k,T} =
    _evaluate_bspline(BSplineOrder(k), t, i, x, T)

# N-th derivative
function evaluate_bspline_diff(::Derivative{N}, ::BSplineOrder{k}, t, i, x,
                               ::Type{T}) where {N,k,T}
    @assert N > 0
    y = zero(T)
    dt = t[i + k - 1] - t[i]
    if !iszero(dt)
        # Recursively evaluate derivative `N - 1` of B-spline of order `k - 1`.
        y += evaluate_bspline_diff(
            Derivative(N - 1), BSplineOrder(k - 1), t, i, x, T) / dt
    end
    dt = t[i + k] - t[i + 1]
    if !iszero(dt)
        y -= evaluate_bspline_diff(
            Derivative(N - 1), BSplineOrder(k - 1), t, i + 1, x, T) / dt
    end
    y * (k - 1)
end

evaluate_bspline(B::BSplineBasis, i, x::AbstractVector, args...) =
    evaluate_bspline.(B, i, x, args...)

"""
    evaluate_bspline!(b::AbstractVector, B::BSplineBasis, i::Integer,
                      x::AbstractVector, args...)

Evaluate i-th B-spline at positions `x` and write result to `b`.

See also [`evaluate_bspline`](@ref).
"""
function evaluate_bspline!(b::AbstractVector{T}, B::AbstractBSplineBasis, i,
                           x::AbstractVector, args...) where {T}
    broadcast!(x -> evaluate_bspline(B, i, x, args..., T), b, x)
end

# Specialisation for first order B-splines.
function _evaluate_bspline(::BSplineOrder{1}, t::AbstractVector, i::Integer,
                           x::Real, ::Type{T}) where {T}
    # Local support of the B-spline.
    @inbounds ta = t[i]
    @inbounds tb = t[i + 1]

    in_local_support(x, t, ta, tb) ? one(T) : zero(T)
end

# General case of order k >= 2.
function _evaluate_bspline(::BSplineOrder{k}, t::AbstractVector, i::Integer,
                           x::Real, ::Type{T}) where {T,k}
    k::Int
    @assert k >= 2

    # Local support of the B-spline.
    @inbounds ta = t[i]
    @inbounds tb = t[i + k]

    if !in_local_support(x, t, ta, tb)
        return zero(T)
    end

    @inbounds ta1 = t[i + 1]
    @inbounds tb1 = t[i + k - 1]

    # Recursively evaluate lower-order B-splines.
    y = zero(T)

    if tb1 != ta
        y += _evaluate_bspline(BSplineOrder(k - 1), t, i, x, T) *
            (x - ta) / (tb1 - ta)
    end

    if ta1 != tb
        y += _evaluate_bspline(BSplineOrder(k - 1), t, i + 1, x, T) *
            (tb - x) / (tb - ta1)
    end

    y
end

# In principle, the support of a B-spline is [ta, tb[.
# The exception is when tb is the last point in the domain, in which case we
# include it in the interval (otherwise the right boundary always has zero value).
@inline in_local_support(x, t, ta, tb) = (ta <= x < tb) || (x == tb == t[end])
