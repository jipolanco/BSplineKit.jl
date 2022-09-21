"""
    SplineWrapper{S <: Spline}

Abstract type representing a type that wraps a [`Spline`](@ref).

Such a type implements all common operations on splines, including evaluation,
differentiation, etc…
"""
abstract type SplineWrapper{S <: Spline} end

"""
    spline(w::SplineWrapper) -> Spline

Returns the [`Spline`](@ref) wrapped by the object.
"""
spline(w::SplineWrapper) = w.spline

Base.eltype(::Type{<:SplineWrapper{S}}) where {S} = eltype(S)

(I::SplineWrapper)(xs...) = spline(I)(xs...)
Base.diff(I::SplineWrapper, etc...) = diff(spline(I), etc...)
Base.:*(op::Derivative, I::SplineWrapper) = op * spline(I)
Base.ndims(I::SplineWrapper) = ndims(spline(I))
Base.size(I::SplineWrapper) = size(spline(I))

for f in (
        :basis, :bases, :order, :orders, :knots,
        :allknots, :coefficients, :integral,
    )
    @eval $f(I::SplineWrapper) = $f(spline(I))
end
