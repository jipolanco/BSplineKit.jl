module SplineExtrapolations

export
    SplineExtrapolation,
    extrapolate,
    Flat,
    Smooth

using ..BSplines
using ..Splines

"""
    AbstractExtrapolationMethod

Abstract type representing an extrapolation method.
"""
abstract type AbstractExtrapolationMethod end

"""
    Flat <: AbstractExtrapolationMethod

Represents a flat extrapolation: spline values at domain limits are extended to
the left and to the right.
"""
struct Flat <: AbstractExtrapolationMethod end

function extrapolate_at_point(::Flat, S::Spline, x)
    a, b = boundaries(basis(S))
    x′ = clamp(x, a, b)
    S(x′)
end

"""
    Smooth <: AbstractExtrapolationMethod

Represents a smooth extrapolation: derivatives up to order ``k - 2`` are
continuous at the boundaries.
"""
struct Smooth <: AbstractExtrapolationMethod end

function extrapolate_at_point(::Smooth, S::Spline, x)
    ts = knots(S)
    N = length(S)
    k = order(S)
    # This is the same as the usual spline evaluation except when zone ≠ 0.
    n, zone = Splines.find_knot_interval(ts, x)
    i = clamp(n, k, N + k)
    Splines.evaluate(S, x, i)
end

"""
    SplineExtrapolation

Represents a spline which can be evaluated outside of its limits according to a
given extrapolation method.
"""
struct SplineExtrapolation{
        S <: Spline,
        M <: AbstractExtrapolationMethod,
    } <: SplineWrapper{S}
    spline :: S
    method :: M

    SplineExtrapolation(sp::Spline, method::AbstractExtrapolationMethod) =
        new{typeof(sp), typeof(method)}(sp, method)
end

method(f::SplineExtrapolation) = f.method

SplineExtrapolation(S::SplineWrapper, args...) = SplineExtrapolation(spline(S), args...)

function Base.show(io::IO, ::MIME"text/plain", f::SplineExtrapolation)
    println(io, nameof(typeof(f)), " containing the ", spline(f))
    let io = IOContext(io, :compact => true, :limit => true)
        print(io, " extrapolation method: ", method(f))
    end
    nothing
end

(f::SplineExtrapolation)(x) = extrapolate_at_point(method(f), spline(f), x)

"""
    extrapolate(S::Union{Spline, SplineWrapper}, method::AbstractExtrapolationMethod)

Construct a [`SplineExtrapolation`](@ref) from the given spline `S` (which can
also be the result of an interpolation).
"""
extrapolate(sp::Union{Spline, SplineWrapper}, method::AbstractExtrapolationMethod) =
    SplineExtrapolation(sp, method)

end
