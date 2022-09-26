"""
    BSplineBasis{k, T}

B-spline basis for splines of order `k` and knot element type `T <: Real`.

The basis is defined by a set of knots and by the B-spline order.

---

    BSplineBasis(order::BSplineOrder{k}, ts::AbstractVector; augment = Val(true))

Create B-spline basis of order `k` with breakpoints `ts`.

If `augment = Val(true)`, breakpoints will be "augmented" so that boundary knots
have multiplicity `k`. Note that, if they are passed as a regular `Vector`, the
input may be modified. See [`augment_knots!`](@ref) for details.

# Examples

```jldoctest BSplineBasis
julia> breaks = range(-1, 1; length = 21)
-1.0:0.1:1.0

julia> B = BSplineBasis(BSplineOrder(5), breaks)
24-element BSplineBasis of order 5, domain [-1.0, 1.0]
 knots: [-1.0, -1.0, -1.0, -1.0, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5  …  0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0]
```

Note that first and last knots are repeated ``k = 5`` times.

If `augment = Val(false)`, input breakpoints are taken without modification as
the knots ``t_i`` of the B-spline basis. Note that the valid domain is reduced to
``[-0.6, 0.6]``. The domain is always defined as the range ``[t_k, t_{N + 1}]``,
where ``N`` is the length of the basis (below, ``N = 16``).

```jldoctest BSplineBasis
julia> Bn = BSplineBasis(5, breaks, augment = Val(false))
16-element BSplineBasis of order 5, domain [-0.6, 0.6]
 knots: -1.0:0.1:1.0
```

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
        # TODO when `knots` is a regular Vector, it would be nice to avoid
        # modifying the vector and to instead use an AugmentedKnots instance
        # (commented line below).
        # However, for now that seems to be sligthly slower when evaluating
        # splines, not sure why...
        t = Augment ? augment_knots!(knots, k) : knots
        # t = Augment ? AugmentedKnots{k}(knots) : knots
        N = length(t) - k
        Knots = typeof(t)
        new{k, T, Knots}(N, t)
    end
end

@inline BSplineBasis(k::Integer, args...; kwargs...) =
    BSplineBasis(BSplineOrder(k), args...; kwargs...)

function basis_derivative(B::BSplineBasis, ::Derivative{n}) where {n}
    @assert n ≥ 0
    ts = knots(B)
    k = order(B)

    # The derived B-spline basis has 2n fewer knots.
    ts′ = view(ts, (firstindex(ts) + n):(lastindex(ts) - n))

    BSplineBasis(BSplineOrder(k - n), ts′; augment = Val(false))
end

Base.:(==)(A::BSplineBasis, B::BSplineBasis) =
    A === B ||
    order(A) == order(B) && knots(A) == knots(B)

"""
    getindex(B::AbstractBSplineBasis, i, [T = Float64])

Get ``i``-th basis function.

This is an alias for `BasisFunction(B, i, T)` (see [`BasisFunction`](@ref) for details).

The returned object can be evaluated at any point within the boundaries defined
by the basis.

# Examples

```jldoctest
julia> B = BSplineBasis(BSplineOrder(4), -1:0.1:1)
23-element BSplineBasis of order 4, domain [-1.0, 1.0]
 knots: [-1.0, -1.0, -1.0, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4  …  0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0]

julia> B[6]
Basis function i = 6
  from 23-element BSplineBasis of order 4, domain [-1.0, 1.0]
  support: [-0.8, -0.4) (knots 6:10)

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
    BasisFunction(B, i, T)
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

function Base.show(io::IO, B::AbstractBSplineBasis)
    summary(io, B)
    let io = IOContext(io, :compact => true, :limit => true)
        print(io, "\n knots: ", knots(B))
    end
    nothing
end

Base.summary(io::IO, B::AbstractBSplineBasis) = summary_basis(io, B)

function summary_basis(io, B::AbstractBSplineBasis)
    a, b = boundaries(B)
    print(io, length(B), "-element ", nameof(typeof(B)))
    print(io, " of order ", order(B), ", domain [", a, ", ", b, "]")
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
function boundaries end

function boundaries(B::BSplineBasis)
    k = order(B)
    N = length(B)
    t = knots(B)
    t[k], t[N + 1]
end

"""
    knots(g::AbstractBSplineBasis)
    knots(g::Spline)

Returns the knots of the B-spline basis.
"""
function knots end

knots(g::BSplineBasis) = g.t

"""
    order(::Type{AbstractBSplineBasis}) -> Int
    order(::AbstractBSplineBasis) -> Int
    order(::Type{Spline}) -> Int
    order(::BSplineOrder) -> Int

Returns order of B-splines as an integer.
"""
function order end

order(::Type{<:AbstractBSplineBasis{k}}) where {k} = k
order(b::AbstractBSplineBasis) = order(typeof(b))
order(::BSplineOrder{k}) where {k} = k
