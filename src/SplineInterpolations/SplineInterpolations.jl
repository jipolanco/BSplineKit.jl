module SplineInterpolations

using ..BSplines
using ..Collocation
using ..Splines

using BandedMatrices
using LinearAlgebra

import ..BSplines:
    order, knots, basis

import ..Splines:
    coefficients, integral

import Interpolations:
    interpolate, interpolate!

export
    SplineInterpolation, spline, interpolate, interpolate!

"""
    SplineInterpolation

Spline interpolation.

This is the type returned by [`interpolate`](@ref).

A `SplineInterpolation` `I` can be evaluated at any point `x` using the `I(x)`
syntax.

It can also be updated with new data on the same data points using
[`interpolate!`](@ref).

---

    SplineInterpolation(undef, B::AbstractBSplineBasis, x::AbstractVector, [T = eltype(x)])

Initialise a `SplineInterpolation` from B-spline basis and a set of
interpolation (or collocation) points `x`.

Note that the length of `x` must be equal to the number of B-splines.

Use [`interpolate!`](@ref) to actually interpolate data known on the `x`
locations.
"""
struct SplineInterpolation{
        S <: Spline,
        F <: Factorization,
        Points <: AbstractVector,
    } <: SplineWrapper{S}

    spline :: S
    C :: F  # factorisation of collocation matrix
    x :: Points

    function SplineInterpolation(
            B::AbstractBSplineBasis, C::Factorization, x::AbstractVector,
        )
        N = length(B)
        size(C) == (N, N) ||
            throw(DimensionMismatch("collocation matrix has wrong dimensions"))
        length(x) == N ||
            throw(DimensionMismatch("wrong number of collocation points"))
        T = eltype(C)
        s = Spline(undef, B, T)  # uninitialised spline
        new{typeof(s), typeof(C), typeof(x)}(s, C, x)
    end
end

# Construct SplineInterpolation from basis and collocation points.
function SplineInterpolation(
        ::UndefInitializer, B, x::AbstractVector, ::Type{T},
    ) where {T}
    # Here we construct the collocation matrix and its LU factorisation.
    N = length(B)
    if length(x) != N
        throw(DimensionMismatch(
            "incompatible lengths of B-spline basis and collocation points"))
    end
    C = collocation_matrix(B, x, CollocationMatrix{T})
    SplineInterpolation(B, lu!(C), x)
end

interpolation_points(S::SplineInterpolation) = S.x

SplineInterpolation(init, B, x::AbstractVector) =
    SplineInterpolation(init, B, x, eltype(x))

function Base.show(io::IO, I::SplineInterpolation)
    println(io, nameof(typeof(I)), " containing the ", spline(I))
    let io = IOContext(io, :compact => true, :limit => true)
        println(io, " interpolation points: ", interpolation_points(I))
    end
    nothing
end

"""
    interpolate!(I::SplineInterpolation, y::AbstractVector)

Update spline interpolation with new data.

This function allows to reuse a [`SplineInterpolation`](@ref) returned by a
previous call to [`interpolate`](@ref), using new data on the same locations
`x`.

See [`interpolate`](@ref) for details.
"""
function interpolate!(I::SplineInterpolation, y::AbstractVector)
    s = spline(I)
    if length(y) != length(s)
        throw(DimensionMismatch("input data has incorrect length"))
    end
    ldiv!(coefficients(s), I.C, y)  # determine B-spline coefficients
    I
end

"""
    interpolate(x, y, BSplineOrder(k))

Interpolate values `y` at locations `x` using B-splines of order `k`.

Grid points `x` must be real-valued and are assumed to be in increasing order.

Returns a [`SplineInterpolation`](@ref) which can be evaluated at any intermediate
point.

See also [`interpolate!`](@ref).

# Examples

```jldoctest
julia> xs = -1:0.1:1;

julia> ys = cospi.(xs);

julia> itp = interpolate(xs, ys, BSplineOrder(4))
SplineInterpolation containing the 21-element Spline{Float64}:
 basis: 21-element BSplineBasis of order 4, domain [-1.0, 1.0]
 order: 4
 knots: [-1.0, -1.0, -1.0, -1.0, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3  …  0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.0, 1.0, 1.0]
 coefficients: [-1.0, -1.00111, -0.8975, -0.597515, -0.314147, 1.3265e-6, 0.314142, 0.597534, 0.822435, 0.96683  …  0.96683, 0.822435, 0.597534, 0.314142, 1.3265e-6, -0.314147, -0.597515, -0.8975, -1.00111, -1.0]
 interpolation points: -1.0:0.1:1.0

julia> itp(-1)
-1.0

julia> itp(0)
1.0

julia> itp(0.42)
0.2486897676885842
```
"""
function interpolate(x::AbstractVector, y::AbstractVector, k::BSplineOrder)
    t = make_knots(x, order(k))
    B = BSplineBasis(k, t; augment = Val(false))  # it's already augmented!
    T = float(eltype(y))
    itp = SplineInterpolation(undef, B, x, T)
    interpolate!(itp, y)
end

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
    @inbounds for i = 1:(N - k)
        ti = zero(T)
        for j = 1:(k - 1)
            ti += x[i + j]
        end
        t[k + i] = ti * kinv
    end

    t
end

end
