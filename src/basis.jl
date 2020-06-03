"""
    AbstractBSplineBasis{k,T}

Abstract type defining a B-spline basis, or more generally, a functional basis
defined from B-splines.

The basis is represented by a B-spline order `k` and a knot element type `T`.
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

function Base.show(io::IO, B::BSplineBasis)
    # This is somewhat consistent with the output of the BSplines package.
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
