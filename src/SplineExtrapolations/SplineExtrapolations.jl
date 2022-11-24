module SplineExtrapolations

export
    SplineExtrapolation,
    extrapolate,
    Flat

using ..BSplines
using ..Splines

abstract type AbstractExtrapolationType end

"""
    Flat <: AbstractExtrapolationType

Singleton type representing a flat extrapolation.
"""
struct Flat <: AbstractExtrapolationType end

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
        ExtType <: AbstractExtrapolationType,
    } <: SplineWrapper{S}
    spline :: S
    type   :: ExtType

    SplineExtrapolation(sp::Spline, type::AbstractExtrapolationType) =
        new{typeof(sp), typeof(type)}(sp, type)
end

SplineExtrapolation(S::SplineWrapper, args...) = SplineExtrapolation(spline(S), args...)

(f::SplineExtrapolation)(x) = extrapolate_at_point(f.type, f.spline, x)

"""
    extrapolate(S::Union{Spline, SplineWrapper}, type::AbstractExtrapolationType)

Construct a [`SplineExtrapolation`](@ref) from the given spline `S` (which can
also be the result of an interpolation).
"""
extrapolate(sp::Union{Spline, SplineWrapper}, type::AbstractExtrapolationType) =
    SplineExtrapolation(sp, type)

end
