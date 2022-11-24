module SplineExtrapolations

export
    SplineExtrapolation,
    extrapolate,
    Flat

using ..BSplines
using ..Splines

"""
    AbstractExtrapolationMethod

Abstract type representing an extrapolation method.
"""
abstract type AbstractExtrapolationMethod end

"""
    Flat <: AbstractExtrapolationMethod

Singleton type representing a flat extrapolation.
"""
struct Flat <: AbstractExtrapolationMethod end

function extrapolate_at_point(::Flat, S::Spline, x)
    a, b = boundaries(basis(S))
    x′ = clamp(x, a, b)
    S(x′)
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

SplineExtrapolation(S::SplineWrapper, args...) = SplineExtrapolation(spline(S), args...)

(f::SplineExtrapolation)(x) = extrapolate_at_point(f.method, f.spline, x)

"""
    extrapolate(S::Union{Spline, SplineWrapper}, method::AbstractExtrapolationMethod)

Construct a [`SplineExtrapolation`](@ref) from the given spline `S` (which can
also be the result of an interpolation).
"""
extrapolate(sp::Union{Spline, SplineWrapper}, method::AbstractExtrapolationMethod) =
    SplineExtrapolation(sp, method)

end
