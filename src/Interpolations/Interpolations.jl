module Interpolations

using ..BSplines
using ..Collocation
using ..Splines

using BandedMatrices
using LinearAlgebra: ldiv!

export spline, interpolate, interpolate!

"""
    Interpolation

Spline interpolation.

This is the type returned by [`interpolate`](@ref).
Generally, it should not be directly constructed.

An `Interpolation` `I` can be evaluated at any point `x` using the `I(x)` syntax.

It can also be updated with new data on the same data points using
[`interpolate!`](@ref).
"""
struct Interpolation{
        T <: Number,
        S <: Spline{T},
        CMatrix <: AbstractMatrix,
    }
    s :: S
    C :: CMatrix  # collocation matrix
    function Interpolation(B::BSplineBasis, C)
        N = length(B)
        if size(C) != (N, N)
            throw(DimensionMismatch("collocation matrix has wrong dimensions"))
        end
        T = eltype(C)
        s = Spline(undef, B, T)  # uninitialised spline
        new{T, typeof(s), typeof(C)}(s, C)
    end
end

"""
    spline(I::Interpolation) -> Spline

Returns the [`Spline`](@ref) associated to the interpolation.
"""
spline(I::Interpolation) = I.s

(I::Interpolation)(x) = spline(I)(x)

"""
    interpolate!(I::Interpolation, y::AbstractVector)

Update spline interpolation with new data.

This function allows to reuse an [`Interpolation`](@ref) returned by a previous
call to [`interpolate`](@ref), using new data on the same locations `x`.

See [`interpolate`](@ref) for details.
"""
function interpolate!(I::Interpolation, y::AbstractVector)
    s = spline(I)
    if length(y) != length(s)
        throw(DimensionMismatch("input data has incorrect length"))
    end
    ldiv!(coefficients(s), I.C, y)  # determine B-spline coefficients
    I
end

"""
    interpolate(x, y, k) -> Interpolation

Interpolate values `y` at locations `x` using B-splines of order `k`.

Grid points `x` must be real-valued and are assumed to be in increasing order.

See also [`interpolate!`](@ref).
"""
function interpolate(x::AbstractVector, y::AbstractVector, k::BSplineOrder)
    t = make_knots(x, order(k))
    B = BSplineBasis(k, t, augment=false)  # it's already augmented!
    T = float(eltype(y))
    C = collocation_matrix(B, x, BandedMatrix{T})
    interpolate!(Interpolation(B, C), y)
end

@inline interpolate(x, y, k::Integer) = interpolate(x, y, BSplineOrder(k))

# Define B-spline knots from collocation points and B-spline order.
# Note that the choice is not unique.
function make_knots(x::AbstractVector{Tx}, k) where {Tx <: Real}
    Base.require_one_based_indexing(x)
    T = float(Tx)  # just in case Tx is Integer...
    N = length(x)
    t = Array{T}(undef, N + k)  # knots

    # First and last knots have multiplicity `k`.
    t[1:k] .= x[1]
    t[(N + 1):(N + k)] .= x[end]

    # Use initial guess for optimal set of knot locations (de Boor 2001, p. 193).
    # One could get the actual optimal using Newton iterations...
    kinv = one(T) / (k - 1)
    for i = 1:(N - k)
        ti = zero(T)
        for j = 1:(k - 1)
            ti += x[i + j]
        end
        t[k + i] = ti * kinv
    end

    t
end

end
