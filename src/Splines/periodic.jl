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

function Base.similar(vs::PeriodicVector, ::Type{S}, dims::Dims) where {S}
    data = similar(parent(vs), S, dims)
    PeriodicVector(data)
end

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

function _derivative(
        B::PeriodicBSplineBasis, S::Spline, op::Derivative{Ndiff},
    ) where {Ndiff}
    Ndiff :: Integer
    @assert Ndiff >= 1

    u = coefficients(S) :: PeriodicVector
    k = order(S)

    if Ndiff >= k
        throw(ArgumentError(
            "cannot differentiate order $k spline $Ndiff times!"))
    end

    B′ = BSplines.basis_derivative(B, op)
    δ = BSplines.index_offset(knots(B)) - BSplines.index_offset(knots(B′))
    @assert δ ≥ 0
    t = knots(B′)

    du = similar(u) :: PeriodicVector

    # Copy coefficients with possible offset to account for different starting
    # point of B-spline knots.
    @inbounds for i ∈ eachindex(u)
        du[i - δ] = u[i]
    end

    @inbounds for m = 1:Ndiff
        # We need to save this value due to periodicity.
        # Note that du[0] and du[N] point to the same value, so that when we
        # modify du[N], we also modify du[0].
        du₀ = last(du)
        for i in Iterators.reverse(eachindex(du))
            dt = t[i + k - m] - t[i]
            @assert !iszero(dt)
            du_prev = i == firstindex(du) ? du₀ : du[i - 1]
            du[i] = (k - m) * (du[i] - du_prev) / dt
        end
    end

    Spline(B′, du)
end

# Note that the integral of a periodic function is in general not periodic
# (unless the function has zero mean over a single period).
# Maybe we could return a spline in a regular BSplineBasis?
_integral(::PeriodicBSplineBasis, ::Spline) =
    error("integration of periodic splines is currently not supported")
