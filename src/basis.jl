"""
    BSplineBasis{k}

B-spline basis for splines of order `k`.

The basis is defined by a set of knots and by the B-spline order.

---

    BSplineBasis(k::Union{Val,Integer}, knots::Vector; augment=true)

Create B-spline basis of order `k` with the given knots.

If `augment=true` (default), knots will be "augmented" so that both knot ends
have multiplicity `k`. See also [`augment_knots`](@ref).
"""
struct BSplineBasis{k, T}  # k: B-spline order
    N :: Int             # number of B-splines ("resolution")
    t :: Vector{T}       # knots (length = N + k)
    function BSplineBasis(::Val{k}, knots::AbstractVector{T};
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
    BSplineBasis(Val(k), args...; kwargs...)

# Make BSplineBasis behave as scalar when broadcasting.
Broadcast.broadcastable(B::BSplineBasis) = Ref(B)

"""
    length(g::BSplineBasis)

Returns the number of B-splines composing a spline.
"""
Base.length(g::BSplineBasis) = g.N
Base.size(g::BSplineBasis) = (g.N, )

"""
    knots(g::BSplineBasis)
    knots(g::Spline)

Returns the knots of the spline grid.
"""
knots(g::BSplineBasis) = g.t

"""
    order(::Type{BSplineBasis})
    order(::Type{Spline})

Returns order of B-splines.
"""
order(::Type{<:BSplineBasis{k}}) where {k} = k
order(b::BSplineBasis) = order(typeof(b))

"""
    BSpline{B <: BSplineBasis}

Describes a single B-spline in the given basis.

---

    BSpline(basis::BSplineBasis, i::Int, [T = Float64])

Construct i-th B-spline in the given basis.

The constructed BSpline can be evaluated as `b(x)`, returning a value of type
`T`.
"""
struct BSpline{Basis <: BSplineBasis, T}
    basis :: Basis
    i     :: Int
    function BSpline(b::BSplineBasis, i, ::Type{T} = Float64) where {T}
        Basis = typeof(b)
        new{Basis, T}(b, i)
    end
end

basis(b::BSpline) = b.basis
knots(b::BSpline) = knots(basis(b))
order(b::BSpline) = order(basis(b))
Base.eltype(::Type{BSpline{B,T}}) where {B,T} = T

"""
    (b::BSpline)(x, [Ndiff = Val(0)])

Evaluate B-spline at coordinate `x`.

To evaluate a derivative, pass the `Ndiff` parameter with the wanted
differentiation order.
"""
(b::BSpline)(x, Ndiff=Val(0)) =
    evaluate_bspline(basis(b), b.i, x, eltype(b), Ndiff=Ndiff)

"""
    evaluate_bspline(
        B::BSplineBasis, i::Integer, x, [T=Float64];
        Ndiff::{Integer, Val} = Val(0),
    )

Evaluate i-th B-spline in the given basis at `x` (can be a coordinate or a
vector of coordinates).

The `N`-th derivative of bᵢ(x) may be evaluated by passing `Ndiff = Val(N)`.

See also [`evaluate_bspline!`](@ref).
"""
function evaluate_bspline(B::BSplineBasis, i::Integer, x::Real,
                          ::Type{T} = Float64;
                          Ndiff::Union{Val,Integer} = Val(0)) where {T,D}
    N = length(B)
    if !(1 <= i <= N)
        throw(DomainError(i, "B-spline index must be in 1:$N"))
    end
    k = order(B)
    t = knots(B)
    Ndiff_val = _make_val(Ndiff)
    evaluate_bspline_diff(Ndiff_val, Val(k), t, i, x, T)
end

@inline _make_val(x) = Val(x)
@inline _make_val(x::Val) = x

# No derivative
evaluate_bspline_diff(::Val{0}, ::Val{k}, t, i, x, ::Type{T}) where {k,T} =
    _evaluate_bspline(Val(k), t, i, x, T)

# N-th derivative
function evaluate_bspline_diff(::Val{N}, ::Val{k}, t, i, x,
                               ::Type{T}) where {N,k,T}
    @assert N > 0
    y = zero(T)
    dt = t[i + k - 1] - t[i]
    if !iszero(dt)
        # Recursively evaluate derivative `N - 1` of B-spline of order `k - 1`.
        y += evaluate_bspline_diff(Val(N - 1), Val(k - 1), t, i, x, T) / dt
    end
    dt = t[i + k] - t[i + 1]
    if !iszero(dt)
        y -= evaluate_bspline_diff(Val(N - 1), Val(k - 1), t, i + 1, x, T) / dt
    end
    y * (k - 1)
end

evaluate_bspline(B::BSplineBasis, i, x::AbstractVector, args...; kwargs...) =
    evaluate_bspline.(B, i, x, args...; kwargs...)

"""
    evaluate_bspline!(b::AbstractVector, B::BSplineBasis, i::Integer,
                      x::AbstractVector; kwargs...)

Evaluate i-th B-spline at positions `x` and write result to `b`.

See also [`evaluate_bspline`](@ref).
"""
function evaluate_bspline!(b::AbstractVector{T}, B::BSplineBasis, i,
                           x::AbstractVector; kwargs...) where {T}
    broadcast!(x -> evaluate_bspline(B, i, x, T; kwargs...), b, x)
end

# Specialisation for first order B-splines.
function _evaluate_bspline(::Val{1}, t::AbstractVector, i::Integer, x::Real,
                           ::Type{T}) where {T}
    # Local support of the B-spline.
    @inbounds ta = t[i]
    @inbounds tb = t[i + 1]

    in_local_support(x, t, ta, tb) ? one(T) : zero(T)
end

# General case of order k >= 2.
function _evaluate_bspline(::Val{k}, t::AbstractVector, i::Integer, x::Real,
                           ::Type{T}) where {T,k}
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
        y += _evaluate_bspline(Val(k - 1), t, i, x, T) *
            (x - ta) / (tb1 - ta)
    end

    if ta1 != tb
        y += _evaluate_bspline(Val(k - 1), t, i + 1, x, T) *
            (tb - x) / (tb - ta1)
    end

    y
end

# In principle, the support of a B-spline is [ta, tb[.
# The exception is when tb is the last point in the domain, in which case we
# include it in the interval (otherwise the right boundary always has zero value).
@inline in_local_support(x, t, ta, tb) = (ta <= x < tb) || (x == tb == t[end])
