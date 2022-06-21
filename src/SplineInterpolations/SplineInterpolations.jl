module SplineInterpolations

using ..BSplines
using ..Collocation
using ..Splines

using BandedMatrices
using LinearAlgebra
using Base.Cartesian: @ntuple

import ..BSplines:
    order, knots, basis

import ..Splines:
    coefficients, integral

using ..Recombinations:
    RecombinedBSplineBasis

using ..BoundaryConditions

import Interpolations:
    interpolate, interpolate!

export
    SplineInterpolation, spline, interpolate, interpolate!

const BasisTuple{N} = Tuple{Vararg{AbstractBSplineBasis, N}} where {N}
const CollocationPointTuple{N} = Tuple{Vararg{AbstractVector, N}} where {N}

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
        N,
        S <: Spline{N},
        Facts <: Tuple{Vararg{Factorization, N}},
        Points <: Tuple{Vararg{AbstractVector, N}},
    } <: SplineWrapper{S}

    spline :: S
    Cs :: Facts   # factorisation of collocation matrices
    xs :: Points  # collocation points associated to each basis

    function SplineInterpolation(
            Bs::Tuple{Vararg{AbstractBSplineBasis, N}},
            Cs::Tuple{Vararg{Factorization, N}},
            xs::Tuple{Vararg{AbstractVector, N}},
            ::Type{T},
        ) where {N, T}
        foreach(Bs, Cs, xs) do B, C, x
            M = length(B)
            size(C) == (M, M) ||
                throw(DimensionMismatch("collocation matrix has wrong dimensions"))
            length(x) == M ||
                throw(DimensionMismatch("wrong number of collocation points"))
        end
        s = Spline(undef, Bs, T)  # uninitialised spline
        new{N, typeof(s), typeof(Cs), typeof(xs)}(s, Cs, xs)
    end
end

# Construct SplineInterpolation from bases and collocation points.
function SplineInterpolation(
        ::UndefInitializer, Bs::BasisTuple{N},
        xs::CollocationPointTuple{N}, ::Type{T},
    ) where {N, T}
    Cs = map(Bs, xs) do B, x
        if length(x) != length(B)
            throw(DimensionMismatch(
                "incompatible lengths of B-spline basis and collocation points"))
        end
        Tx = eltype(x)
        Tc = float(Tx)
        # Construct collocation matrix and its LU factorisation.
        C = collocation_matrix(B, x, CollocationMatrix{Tc})
        lu!(C)
    end
    SplineInterpolation(Bs, Cs, xs, T)
end

function SplineInterpolation(
        init, Bs::Tuple{Vararg{AbstractBSplineBasis}},
        xs::Tuple{Vararg{AbstractVector}},
    )
    T = promote_type(map(eltype, xs)...)
    SplineInterpolation(init, Bs, xs, T)
end

# For 1D interpolations
function SplineInterpolation(
        init, B::AbstractBSplineBasis, x::AbstractVector, args...,
    )
    SplineInterpolation(init, (B,), (x,), args...)
end

interpolation_points(S::SplineInterpolation) = S.xs

function Base.show(io::IO, I::SplineInterpolation)
    println(io, nameof(typeof(I)), " containing the ", spline(I))
    xs = interpolation_points(I)
    let io = IOContext(io, :compact => true, :limit => true)
        if length(xs) == 1
            println(io, " interpolation points: ", xs[1])
        else
            println(io, " interpolation points:")
            for (i, x) ∈ enumerate(xs)
                println(io, "  ($i) ", x)
            end
        end
    end
    nothing
end

"""
    interpolate!(I::SplineInterpolation{N}, y::AbstractArray{T,N})

Update spline interpolation with new data.

This function allows to reuse a [`SplineInterpolation`](@ref) returned by a
previous call to [`interpolate`](@ref), using new data on the same locations
`x`.

See [`interpolate`](@ref) for details.
"""
function interpolate!(I::SplineInterpolation{N}, y::AbstractArray{T,N}) where {T,N}
    if size(y) != size(I)
        throw(DimensionMismatch("input data has incorrect dimensions"))
    end
    _interpolate!(I, y)
end

# 1D case
function _interpolate!(I::SplineInterpolation{1}, y::AbstractArray{T,1} where T)
    s = spline(I)
    C, = I.Cs  # this is a length-1 tuple
    ldiv!(coefficients(s), C, y)  # determine B-spline coefficients
    I
end

# General ND case
function _interpolate!(I::SplineInterpolation{N}, y::AbstractArray{T,N}) where {T,N}
    @assert N > 1
    s = spline(I)
    Cs = I.Cs
    coefs = coefficients(s)
    _interpolate!(coefs, Cs, y)
    I
end

@inline function _interpolate!(
        coefs::AbstractArray{T, N},  # output coefficients
        Cs::Tuple{Vararg{Factorization, N}},  # factorised collocation matrices
        fdata::AbstractArray{T, N},  # input data
    ) where {T, N}
    @assert N ≥ 2
    _interpolate_dim!(coefs, fdata, Cs...)
end

# Interpolate over dimension j
# Note that coefs and rhs may be aliased (point to the same data).
@inline function _interpolate_dim!(coefs, rhs, Cj, Cnext...)
    N = ndims(coefs)
    R = length(Cnext)
    j = N - R
    L = j - 1
    inds = axes(coefs)
    inds_l = CartesianIndices(ntuple(d -> @inbounds(inds[d]), Val(L)))
    inds_r = CartesianIndices(ntuple(d -> @inbounds(inds[j + d]), Val(R)))
    @inbounds for J ∈ inds_r, I ∈ inds_l
        coefs_ij = @view coefs[I, :, J]
        rhs_ij = @view rhs[I, :, J]
        ldiv!(coefs_ij, Cj, rhs_ij)
    end
    _interpolate_dim!(coefs, coefs, Cnext...)
end

@inline _interpolate_dim!(coefs, rhs) = coefs

"""
    interpolate(x, y, BSplineOrder(k), [bc = nothing])

Interpolate values `y` at locations `x` using B-splines of order `k`.

Grid points `x` must be real-valued and are assumed to be in increasing order.

Returns a [`SplineInterpolation`](@ref) which can be evaluated at any intermediate
point.

Optionally, one may pass one of the boundary conditions listed in the [Boundary
conditions](@ref boundary-conditions-api) section.
For now, only the [`Natural`](@ref) boundary condition is available.

See also [`interpolate!`](@ref).

# Examples

```jldoctest
julia> xs = -1:0.1:1;

julia> ys = cospi.(xs);

julia> itp = interpolate(xs, ys, BSplineOrder(4))
SplineInterpolation containing the 21-element Spline{1, Float64}:
 basis: 21-element BSplineBasis of order 4, domain [-1.0, 1.0]
 order: 4
 knots: [-1.0, -1.0, -1.0, -1.0, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3  …  0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.0, 1.0, 1.0]
 coefficients: [-1.0, -1.00111, -0.8975, -0.597515, -0.314147, 1.3265e-6, 0.314142, 0.597534, 0.822435, 0.96683  …  0.96683, 0.822435, 0.597534, 0.314142, 1.3265e-6, -0.314147, -0.597515, -0.8975, -1.00111, -1.0]
 interpolation points: -1.0:0.1:1.0

julia> itp(-1)
-1.0

julia> (Derivative(1) * itp)(-1)
-0.01663433622896893

julia> (Derivative(2) * itp)(-1)
10.52727328755495

julia> Snat = interpolate(xs, ys, BSplineOrder(4), Natural())
SplineInterpolation containing the 21-element Spline{1, Float64}:
 basis: 21-element RecombinedBSplineBasis of order 4, domain [-1.0, 1.0], BCs {left => (D{2},), right => (D{2},)}
 order: 4
 knots: [-1.0, -1.0, -1.0, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4  …  0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0]
 coefficients: [-0.833333, -0.647516, -0.821244, -0.597853, -0.314057, -2.29076e-5, 0.314148, 0.597532, 0.822435, 0.96683  …  0.96683, 0.822435, 0.597532, 0.314148, -2.29076e-5, -0.314057, -0.597853, -0.821244, -0.647516, -0.833333]
 interpolation points: -1.0:0.1:1.0

julia> Snat(-1)
-1.0

julia> (Derivative(1) * Snat)(-1)
0.28726186708894824

julia> (Derivative(2) * Snat)(-1)
0.0

```
"""
function interpolate(
        xs::Tuple{Vararg{AbstractVector, N}},
        y::AbstractArray{Ty, N},
        k::BSplineOrder,
        ::Nothing = nothing,
    ) where {Ty, N}
    Bs = map(xs) do x
        t = make_knots(x, order(k))
        BSplineBasis(k, t; augment = Val(false))  # it's already augmented!
    end

    # If input data is integer, convert the spline element type to float.
    # This also does the right thing when eltype(y) <: StaticArray.
    T = float(Ty)

    itp = SplineInterpolation(undef, Bs, xs, T)
    interpolate!(itp, y)
end

function interpolate(
        xs::Tuple{Vararg{AbstractVector, N}},
        y::AbstractArray{Ty, N},
        k::BSplineOrder,
        bc::Natural,
    ) where {Ty, N}
    # For natural BCs, the number of required unique knots is equal to the
    # number of data points, and therefore we just make them equal.
    Rs = map(xs) do x
        B = BSplineBasis(k, copy(x))  # note that this modifies x, so we create a copy...
        RecombinedBSplineBasis(bc, B)
    end
    T = float(Ty)
    itp = SplineInterpolation(undef, Rs, xs, T)
    interpolate!(itp, y)
end

# For 1D interpolations
function interpolate(
        x::AbstractVector, y::AbstractVector, k::BSplineOrder, args...,
    )
    interpolate((x,), y, k, args...)
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
