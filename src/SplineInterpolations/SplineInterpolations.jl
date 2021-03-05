module SplineInterpolations

using ..BSplines
using ..Collocation
using ..Splines

using BandedMatrices
using LinearAlgebra

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
        F <: Factorization,
    }
    s :: S
    C :: F  # factorisation of collocation matrix

    function Interpolation(B::BSplineBasis, C::Factorization)
        N = length(B)
        if size(C) != (N, N)
            throw(DimensionMismatch("collocation matrix has wrong dimensions"))
        end
        T = eltype(C)
        s = Spline(undef, B, T)  # uninitialised spline
        new{T, typeof(s), typeof(C)}(s, C)
    end

    # Construct Interpolation from basis and collocation points.
    function Interpolation(B, x::AbstractVector, ::Type{T}) where {T}
        # Here we construct the collocation matrix and its LU factorisation.
        N = length(B)
        if length(x) != N
            throw(DimensionMismatch(
                "incompatible lengths of B-spline basis and collocation points"))
        end
        C = collocation_matrix(B, x, CollocationMatrix{T})
        Interpolation(B, lu!(C))
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
    interpolate(x, y, k::Integer)
    interpolate(x, y, BSplineOrder(k))

Interpolate values `y` at locations `x` using B-splines of order `k`.

Grid points `x` must be real-valued and are assumed to be in increasing order.

Returns an [`Interpolation`](@ref) which can be evaluated at any intermediate
point.

The second form using [`BSplineOrder`](@ref) ensures that the type of the
returned [`Interpolation`](@ref) is fully inferred by the compiler.

See also [`interpolate!`](@ref).

# Examples

```jldoctest
julia> xs = -1:0.1:1;

julia> ys = cospi.(xs);

julia> itp = interpolate(xs, ys, BSplineOrder(4));

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
    B = BSplineBasis(k, t, augment=Val(false))  # it's already augmented!
    T = float(eltype(y))
    itp = Interpolation(B, x, T)
    interpolate!(itp, y)
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
