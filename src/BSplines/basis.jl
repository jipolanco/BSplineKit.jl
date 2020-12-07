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
    getindex(B::AbstractBSplineBasis, i, [T = Float64])

Get ``i``-th basis function.

This is an alias for `BSpline(B, i, T)` (see [`BSpline`](@ref) for details).

The returned object can be evaluated at any point within the boundaries defined
by the basis.

# Examples

```jldoctest
julia> B = BSplineBasis(BSplineOrder(4), -1:0.1:1)
23-element BSplineBasis: order 4, domain [-1.0, 1.0]

julia> B[6]
Basis function i = 6
  from 23-element BSplineBasis: order 4, domain [-1.0, 1.0]
  support: [-0.8, -0.4] (6:10)

julia> B[6](-0.5)
0.16666666666666666

julia> B[6, Float32](-0.5)
0.16666667f0

julia> B[6](-0.5, Derivative(1))
-5.000000000000001
```
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

Base.show(io::IO, B::AbstractBSplineBasis) = summary(io, B)
Base.summary(io::IO, B::BSplineBasis) = summary_basis(io, B)

function summary_basis(io, B::AbstractBSplineBasis)
    a, b = boundaries(B)
    print(io, length(B), "-element ", nameof(typeof(B)))
    print(io, ": order ", order(B), ", domain [", a, ", ", b, "]")
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
