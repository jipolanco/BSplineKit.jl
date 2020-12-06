"""
    BSplineBasis{k}

B-spline basis for splines of order `k`.

The basis is defined by a set of knots and by the B-spline order.

---

    BSplineBasis(
        k::Union{BSplineOrder,Integer}, knots::Vector;
        augment::Val = Val(true),
    )

Create B-spline basis of order `k` with the given knots.

If `augment = Val(true)` (default), knots will be "augmented" so that both knot
ends have multiplicity `k`. See also [`augment_knots`](@ref).
"""
struct BSplineBasis{k, T, Knots <: AbstractVector{T}} <: AbstractBSplineBasis{k,T}
    N :: Int    # number of B-splines
    t :: Knots  # knots (length = N + k)
    function BSplineBasis(
            ::BSplineOrder{k}, knots::AbstractVector{T};
            augment::Val{Augment} = Val(true),
        ) where {k,T,Augment}
        k :: Integer
        Augment :: Bool
        if k <= 0
            throw(ArgumentError("B-spline order must be k ≥ 1"))
        end
        t = Augment ? augment_knots(knots, k) : knots
        N = length(t) - k
        Knots = typeof(t)
        new{k, T, Knots}(N, t)
    end
end

@inline BSplineBasis(k::Integer, args...; kwargs...) =
    BSplineBasis(BSplineOrder(k), args...; kwargs...)

"""
    getindex(B::AbstractBSplineBasis, i, [::Type{T} = Float64])

Get ``i``-th basis function.

This is an alias for `BSpline(B, i, T)`.

The returned object can be evaluated at any point within the boundaries defined
by the basis (see [`BSpline`](@ref) for details).
"""
@inline function Base.getindex(
        B::AbstractBSplineBasis, i::Integer, ::Type{T} = Float64) where {T}
    @boundscheck checkbounds(B, i)
    BSpline(B, i, T)
end

@inline function Base.checkbounds(B::AbstractBSplineBasis, I)
    checkbounds(eachindex(B), I)
end

@inline Base.eachindex(B::AbstractBSplineBasis) = Base.OneTo(length(B))

@inline function Base.iterate(B::AbstractBSplineBasis, i = 0)
    i == length(B) && return nothing
    i += 1
    B[i], i
end

function Base.show(io::IO, B::BSplineBasis)
    # This is inspired by the BSplines package.
    println(io, length(B), "-element ", typeof(B), ':')
    println(io, " order: ", order(B))
    print(io, " knots: ", knots(B))
    nothing
end

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

Returns the knots of the B-spline basis.
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
