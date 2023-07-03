module SplineInterpolations

using ..BSplines
using ..Collocation
using ..Collocation: CyclicTridiagonalMatrix
using ..Splines

using BandedMatrices
using LinearAlgebra
using SparseArrays

import ..BSplines:
    order, knots, basis

import ..Splines:
    coefficients, integral

using ..Recombinations:
    RecombinedBSplineBasis

using ..BoundaryConditions

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
        ColMatCopy <: Union{Nothing, AbstractMatrix},
    } <: SplineWrapper{S}

    spline :: S
    C :: F           # factorisation of collocation matrix
    x :: Points

    # Optional copies of collocation matrix and collocation points.
    # Right now, these are used for cubic periodic splines, which modify `C`
    # and `x` (as well as `y`) when `ldiv!` is called.
    # See `CyclicTridiagonalMatrix` for details.
    C_copy :: ColMatCopy

    function SplineInterpolation(
            B::AbstractBSplineBasis, C::Factorization, x::AbstractVector,
            ::Type{T};
            C_copy = nothing,
        ) where {T}
        N = length(B)
        size(C) == (N, N) ||
            throw(DimensionMismatch("collocation matrix has wrong dimensions"))
        length(x) == N ||
            throw(DimensionMismatch("wrong number of collocation points"))
        s = Spline(undef, B, T)  # uninitialised spline
        new{typeof(s), typeof(C), typeof(x), typeof(C_copy)}(
            s, C, x, C_copy,
        )
    end
end

# Construct SplineInterpolation from basis and collocation points.
function SplineInterpolation(
        ::UndefInitializer, B, x::AbstractVector{Tx}, ::Type{T},
    ) where {Tx <: Real, T}
    # Here we construct the collocation matrix and its LU factorisation.
    N = length(B)
    if length(x) != N
        throw(DimensionMismatch(
            "incompatible lengths of B-spline basis and collocation points"))
    end
    Tc = float(Tx)
    C = collocation_matrix(B, x, Collocation.default_matrix_type(B, Tc))
    C_copy = _maybe_copy_matrix(C)
    SplineInterpolation(B, _factorise!(C), x, T; C_copy)
end

# Factorise in-place when possible.
_factorise!(C::AbstractMatrix) = lu!(C)
_factorise!(C::SparseMatrixCSC) = lu(C)  # SparseMatrixCSC doesn't support `lu!`

# Create copy if required by the matrix type.
_maybe_copy_matrix(::AbstractMatrix) = nothing
_maybe_copy_matrix(C::CyclicTridiagonalMatrix) = copy(C)  # periodic cubic splines

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
function interpolate!(I::SplineInterpolation, ys::AbstractVector)
    S = spline(I)
    cs = Splines.unwrap_coefficients(S)
    if length(ys) != length(cs)
        throw(DimensionMismatch("input data has incorrect length"))
    end
    (; C, C_copy,) = I
    if C_copy !== nothing
        copy!(C, C_copy)  # this is for methods that modify C (case of CyclicTridiagonalMatrix)
    end
    ldiv!(cs, C, ys)  # determine B-spline coefficients
    I
end

"""
    interpolate(x, y, BSplineOrder(k), [bc = nothing])

Interpolate values `y` at locations `x` using B-splines of order `k`.

Grid points `x` must be real-valued and are assumed to be in increasing order.

Returns a [`SplineInterpolation`](@ref) which can be evaluated at any intermediate
point.

Optionally, one may pass one of the boundary conditions listed in the [Boundary
conditions](@ref boundary-conditions-api) section.
Currently, the [`Natural`](@ref) and [`Periodic`](@ref) boundary conditions are
available.

See also [`interpolate!`](@ref).

!!! note "Periodic boundary conditions"

    Periodic boundary conditions should be used if the interpolated data is
    supposed to represent a periodic signal.
    In this case, pass `bc = Period(L)`, where `L` is the period of the x-axis.
    Note that the endpoint `x[begin] + L` should *not* be included in the `x`
    vector.

!!! note "Cubic periodic splines"

    *Cubic* periodic splines (`BSplineOrder(4)`) are particularly well
    optimised compared to periodic splines of other orders.
    Just note that interpolations using cubic periodic splines modify their
    input (including `x` and `y` values).

# Examples

```jldoctest interpolate
julia> xs = -1:0.1:1;

julia> ys = cospi.(xs);

julia> S = interpolate(xs, ys, BSplineOrder(4))
SplineInterpolation containing the 21-element Spline{Float64}:
 basis: 21-element BSplineBasis of order 4, domain [-1.0, 1.0]
 order: 4
 knots: [-1.0, -1.0, -1.0, -1.0, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3  …  0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.0, 1.0, 1.0]
 coefficients: [-1.0, -1.00111, -0.8975, -0.597515, -0.314147, 1.3265e-6, 0.314142, 0.597534, 0.822435, 0.96683  …  0.96683, 0.822435, 0.597534, 0.314142, 1.3265e-6, -0.314147, -0.597515, -0.8975, -1.00111, -1.0]
 interpolation points: -1.0:0.1:1.0

julia> S(-1)
-1.0

julia> (Derivative(1) * S)(-1)
-0.01663433622896893

julia> (Derivative(2) * S)(-1)
10.52727328755495

julia> Snat = interpolate(xs, ys, BSplineOrder(4), Natural())
SplineInterpolation containing the 21-element Spline{Float64}:
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

## Periodic boundary conditions

Interpolate ``f(x) = \\cos(πx)`` for ``x ∈ [-1, 1)``.
Note that the period is ``L = 2`` and that the endpoint (``x = 1``) must *not*
be included in the data points.

```jldoctest interpolate
julia> xp = -1:0.1:0.9;

julia> yp = cospi.(xp);

julia> Sper = interpolate(xp, yp, BSplineOrder(4), Periodic(2))
SplineInterpolation containing the 20-element Spline{Float64}:
 basis: 20-element PeriodicBSplineBasis of order 4, domain [-1.0, 1.0), period 2.0
 order: 4
 knots: [..., -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3  …  0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, ...]
 coefficients: [..., -1.01659, -0.96683, -0.822435, -0.597534, -0.314142, 1.10589e-17, 0.314142, 0.597534, 0.822435, 0.96683, 1.01659, 0.96683, 0.822435, 0.597534, 0.314142, 3.90313e-18, -0.314142, -0.597534, -0.822435, -0.96683, ...]
 interpolation points: -1.0:0.1:0.9
```

As expected, the periodic spline does a better job at approximating the periodic
function ``f(x) = \\cos(πx)`` near the boundaries than the other interpolations:

```jldoctest interpolate
julia> x = -0.99; cospi(x), Sper(x), Snat(x), S(x)
(-0.9995065603657316, -0.9995032595823043, -0.9971071640321146, -0.9996420091470221)

julia> x = 0.998; cospi(x), Sper(x), Snat(x), S(x)
(-0.9999802608561371, -0.9999801044078943, -0.9994253145274461, -1.0000122303614758)
```
"""
function interpolate(
        x::AbstractVector, y::AbstractVector, k::BSplineOrder,
        bc::Nothing = nothing,
    )
    t = make_knots(x, k, bc)
    B = BSplineBasis(k, t; augment = Val(false))  # it's already augmented!

    # If input data is integer, convert the spline element type to float.
    # This also does the right thing when eltype(y) <: StaticArray.
    T = float(eltype(y))

    itp = SplineInterpolation(undef, B, x, T)
    interpolate!(itp, y)
end

function interpolate(
        x::AbstractVector, y::AbstractVector, k::BSplineOrder, bc::Natural,
    )
    # For natural BCs, the number of required unique knots is equal to the
    # number of data points, and therefore we just make them equal.
    B = BSplineBasis(k, copy(x))  # note that this modifies x, so we create a copy...
    R = RecombinedBSplineBasis(B, bc)
    T = float(eltype(y))
    itp = SplineInterpolation(undef, R, x, T)
    interpolate!(itp, y)
end

function interpolate(
        x::AbstractVector, y::AbstractVector, k::BSplineOrder, bc::Periodic,
    )
    ts = make_knots(x, k, bc)
    L = BoundaryConditions.period(bc)
    @assert ts[end] - ts[begin] ≈ L  # endpoint is included in the knots
    B = PeriodicBSplineBasis(k, ts)
    T = float(eltype(y))
    itp = SplineInterpolation(undef, B, x, T)
    interpolate!(itp, y)
end

# Define B-spline knots from collocation points and B-spline order.
# Note that the choice is not unique.
# This is for the case without BCs.
function make_knots(
        x::AbstractVector{Tx}, ::BSplineOrder{k}, ::Nothing,
    ) where {Tx <: Real, k}
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

# Case of periodic BCs.
# For even-order splines, put knots at the same positions as interpolation
# points.
# Note that this doesn't work great for odd-order splines, since it leads to an
# ill-defined linear system (non-invertible collocation matrix).
function make_knots(
        xs::AbstractVector{Tx}, ::BSplineOrder{k}, bc::Periodic,
    ) where {Tx <: Real, k}
    L = BoundaryConditions.period(bc)
    first(xs) + L > last(xs) ||
        error("endpoint x₁ + L must *not* be included in the interpolation points")

    ts = similar(xs, length(xs) + 1)  # note: knots *do* include the endpoint

    if iseven(k)
        @views ts[begin:(end - 1)] .= xs
        ts[end] = first(xs) + L
        return ts
    end

    x₀ = first(xs)
    xₙ = x₀ + L
    xprev = x₀
    i = firstindex(ts)
    @inbounds while i < lastindex(xs)
        x = xs[i + 1]
        ts[i] = (xprev + x) / 2
        xprev = x
        i += 1
    end
    L = BoundaryConditions.period(bc)
    ts[end - 1] = (xprev + xₙ) / 2

    # Make sure first knot == first point.
    # This is somewhat arbitrary, but it's needed so that the boundaries of the
    # B-spline basis correspond to those of the data.
    # Unfortunately, this produces some extra spacing between the first knot and
    # the rest.
    # Not sure if there's a better way of doing this...
    ts[begin] = x₀
    ts[end] = xₙ  # must be x₀ + L

    ts
end

end
