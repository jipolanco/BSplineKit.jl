"""
    AugmentedKnots{T,k} <: AbstractVector{T}

Pads from both sides a vector of B-spline breakpoints, making sure that the
first and last values are repeated `k` times.
"""
struct AugmentedKnots{T, k, Breakpoints <: AbstractVector{T}} <: AbstractVector{T}
    Nt :: Int  # total number of knots
    x  :: Breakpoints
    function AugmentedKnots{k}(x::AbstractVector) where {k}
        Base.require_one_based_indexing(x)
        T = eltype(x)
        Nt = 2 * (k - 1) + length(x)
        new{T, k, typeof(x)}(Nt, x)
    end
end

# This is the number of knots added on each side.
padding(t::AugmentedKnots{T,k}) where {T,k} = k - 1
breakpoints(t::AugmentedKnots) = t.x

Base.size(t::AugmentedKnots) = (t.Nt, )
Base.IndexStyle(::Type{<:AugmentedKnots}) = IndexLinear()

@inline function Base.getindex(t::AugmentedKnots, i::Int)
    @boundscheck checkbounds(t, i)
    x = breakpoints(t)
    j = clamp(i - padding(t), 1, length(x))
    @inbounds x[j]
end

# NOTE: Modifying knots is dangerous, and may fail if the underlying breakpoints
# are immutable (for instance if they're AbstractRange).
# Also, we don't verify that knots stay sorted as they should.
# If `i` is within one of the padded regions, we update all the elements in that
# region to the new value.
@inline function Base.setindex!(t::AugmentedKnots, v, i::Int)
    @boundscheck checkbounds(t, i)
    x = breakpoints(t)
    j = clamp(i - padding(t), 1, length(x))
    @inbounds x[j] = i
end

"""
    augment_knots(knots::AbstractVector, k::Union{Integer,BSplineOrder})

Modifies the input knots to make sure that the first and last knot have
multiplicity `k` for splines of order `k`.

It is assumed that border knots have multiplicity 1 at the borders.
That is, border coordinates should *not* be repeated in the input.
"""
augment_knots(knots::AbstractVector, ::BSplineOrder{k}) where {k} =
    AugmentedKnots{k}(knots)

@inline augment_knots(knots, k::Integer) = augment_knots(knots, BSplineOrder(k))

"""
    multiplicity(knots, i)

Determine multiplicity of knot `knots[i]`.
"""
function multiplicity(knots::AbstractVector, i)
    Base.require_one_based_indexing(knots)
    v = knots[i]
    m = 1

    # Check in both directions
    j = i - 1
    while j > 0 && knots[j] == v
        j -= 1
        m += 1
    end

    j = i + 1
    N = length(knots)
    while j <= N && knots[j] == v
        j += 1
        m += 1
    end

    m
end
