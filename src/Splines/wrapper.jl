"""
    SplineWrapper{S <: Spline} <: Function

Abstract type representing a type that wraps a [`Spline`](@ref).

Such a type implements all common operations on splines, including evaluation,
differentiation, etc…
"""
abstract type SplineWrapper{S <: Spline} <: Function end

"""
    spline(w::SplineWrapper) -> Spline

Returns the [`Spline`](@ref) wrapped by the object.
"""
spline(w::SplineWrapper) = w.spline

Base.show(io::IO, S::SplineWrapper) = show(io, MIME("text/plain"), S)

Base.eltype(::Type{<:SplineWrapper{S}}) where {S} = eltype(S)

(I::SplineWrapper)(x) = spline(I)(x)
Base.diff(I::SplineWrapper, etc...) = diff(spline(I), etc...)
Base.:*(op::Derivative, I::SplineWrapper) = op * spline(I)
Base.:(==)(o1::SplineWrapper, o2::SplineWrapper) = spline(o1) == spline(o2)

for f in (:basis, :order, :knots, :coefficients, :integral)
    @eval $f(I::SplineWrapper) = $f(spline(I))
end
