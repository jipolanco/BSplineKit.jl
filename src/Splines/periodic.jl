using ..BSplines:
    AbstractPeriodicVector

"""
    PeriodicVector{T} <: AbstractVector{T}

Describes a periodic (or "circular") vector wrapping a regular vector.

Used to store the coefficients of periodic splines.

The vector has an effective length `N` associated to a single period, but it is
possible to index it outside of this "main" interval.

This is similar to [`BSplines.PeriodicKnots`](@ref).
It is simpler though, since here there is no notion of coordinates or of a
period `L`.
Periodicity is only manifest in the indexation of the vector, e.g. a
`PeriodicVector` `vs` satisfies `vs[i + N] == vs[i]`.

---

    PeriodicVector(cs::AbstractVector)

Wraps coefficient vector `cs` such that it can be indexed in a periodic manner.
"""
struct PeriodicVector{
        T, Data <: AbstractVector{T},
    } <: AbstractPeriodicVector{T}
    data :: Data
    N    :: Int

    function PeriodicVector(cs::AbstractVector{T}) where {T}
        N = length(cs)
        Data = typeof(cs)
        new{T, Data}(cs, N)
    end
end

Base.length(vs::PeriodicVector) = vs.N
Base.axes(vs::PeriodicVector) = axes(parent(vs))

@inline function index_to_main_period(vs::PeriodicVector, i::Int)
    data = parent(vs)
    while i < firstindex(data)
        i += length(vs)
    end
    while i > lastindex(data)
        i -= length(vs)
    end
    i
end

@inline function Base.getindex(vs::PeriodicVector, i::Int)
    i = index_to_main_period(vs, i)
    @inbounds parent(vs)[i]
end

@inline function Base.setindex!(vs::PeriodicVector, val, i::Int)
    i = index_to_main_period(vs, i)
    @inbounds parent(vs)[i] = val
end

wrap_coefficients(::PeriodicBSplineBasis, cs::AbstractVector) = PeriodicVector(cs)

# Make sure that we don't re-wrap already wrapped coefficients.
wrap_coefficients(::PeriodicBSplineBasis, cs::PeriodicVector) = cs

unwrap_coefficients(::PeriodicBSplineBasis, cs::PeriodicVector) = parent(cs)
