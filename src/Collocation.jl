module Collocation

using ..BSplines

using BandedMatrices

export collocation_points, collocation_points!
export collocation_matrix, collocation_matrix!

"""
    SelectionMethod

Abstract type defining a method for choosing collocation points from spline
knots.
"""
abstract type SelectionMethod end

"""
    AtMaxima <: SelectionMethod

Select collocation points as the locations where each B-spline has its
maximum value.
"""
struct AtMaxima <: SelectionMethod end

"""
    AvgKnots <: SelectionMethod

Each collocation point is chosen as a sliding average over `k - 1` knots.

The resulting collocation points are sometimes called Greville sites
(de Boor 2001) or Marsden--Schoenberg points (e.g. Botella & Shariff IJCFD
2003).
"""
struct AvgKnots <: SelectionMethod end

# Check dimensions of collocation points.
function checkdims(B::BSplineBasis, x::AbstractVector)
    N = length(B)
    if N != length(x)
        throw(ArgumentError(
            "number of collocation points must match number $N of B-splines"))
    end
    nothing
end

"""
    collocation_points(B::BSplineBasis; method=Collocation.AvgKnots())

Define and return adapted collocation points for evaluation of splines.

The number of returned collocation points is equal to the number of B-splines in
the spline basis.

Collocation points can be selected in different ways.
The selection method can be chosen via the `method` argument, which accepts the
following values:

- [`Collocation.AvgKnots()`](@ref) (this is the default)
- [`Collocation.AtMaxima()`](@ref) (actually, not yet supported!)

See also [`collocation_points!`](@ref).
"""
function collocation_points(
        B::BSplineBasis; method::SelectionMethod=AvgKnots())
    x = similar(knots(B), length(B))
    collocation_points!(x, B, method=method)
end

"""
    collocation_points!(x::AbstractVector, B::BSplineBasis;
                        method::SelectionMethod=AvgKnots())

Fill vector with collocation points for evaluation of splines.

See [`collocation_points`](@ref) for details.
"""
function collocation_points!(
        x::AbstractVector, B::BSplineBasis;
        method::SelectionMethod=AvgKnots())
    checkdims(B, x)
    collocation_points!(method, x, B)
end

function collocation_points!(::AvgKnots, x, B)
    N = length(B)
    @assert length(x) == N
    k = order(B)
    t = knots(B)
    j = 0
    T = eltype(x)
    v::T = inv(k - 1)
    a::T, b::T = t[k], t[N + 1]  # domain boundaries
    for i in eachindex(x)
        j += 1  # generally i = j, unless x has weird indexation

        xi = zero(T)
        for n = 1:k-1
            xi += T(t[j + n])
        end
        xi *= v

        # Make sure that the point is inside the domain.
        # This may not be the case if end knots have multiplicity less than k.
        x[i] = clamp(xi, a, b)
    end

    x
end

function collocation_points!(::S, x, B) where {S <: SelectionMethod}
    throw(ArgumentError("'$S' selection method not yet supported!"))
    x
end

# TODO
# - support derivatives
"""
    collocation_matrix(B::BSplineBasis, x::AbstractVector,
                       [MatrixType=BandedMatrix{Float64}])

Return banded collocation matrix used to obtain B-spline coefficients from data
at the collocation points `x`.

The matrix elements are given by the B-splines evaluated at the collocation
points:

    C[i, j] = bⱼ(x[i]).

Due to the compact support of B-splines, the collocation matrix is
[banded](https://en.wikipedia.org/wiki/Band_matrix) if the collocation points are
By default, a `BandedMatrix` (from the
[BandedMatrices](https://github.com/JuliaMatrices/BandedMatrices.jl) package) is
returned.
This assumes that the collocation points are "well" distributed
The type of the returned matrix can be changed via the `MatrixType` argument.

See also [`collocation_matrix!`](@ref).
"""
function collocation_matrix(B::BSplineBasis, x::AbstractVector,
        ::Type{MatrixType} = BandedMatrix{Float64}) where {MatrixType}
    C = allocate_collocation_matrix(MatrixType, length(B), order(B))
    collocation_matrix!(C, B, x)
end

allocate_collocation_matrix(::Type{M}, N, k) where {M <: AbstractMatrix} =
    M(undef, N, N)

function allocate_collocation_matrix(::Type{M}, N, k) where {M <: BandedMatrix}
    # Number of upper and lower diagonal bands.
    #  - for even k: Nb = k / 2       (total = k + 1 bands)
    #  - for odd  k: Nb = (k + 1) / 2 (total = k + 2 bands)
    Nb = div(k + 1, 2)
    M(undef, (N, N), (Nb, Nb))
end

"""
    collocation_matrix!(C::AbstractMatrix, B::BSplineBasis, x::AbstractVector)

Fill preallocated collocation matrix.

See also [`collocation_matrix`](@ref).
"""
function collocation_matrix!(C::AbstractMatrix{T}, B::BSplineBasis,
                             x::AbstractVector) where {T}
    checkdims(B, x)
    N, N2 = size(C)
    if !(N == N2 == length(B))
        throw(ArgumentError("wrong dimensions of collocation matrix"))
    end
    fill!(C, 0)
    for j = 1:N, i = 1:N
        b = evaluate_bspline(B, j, x[i], T)
        # This will fail if C is a BandedMatrix, b is non-zero, and (i, j) is
        # outside the allowed bands. That will happen if the collocation points
        # are not properly distributed.
        C[i, j] = b
    end
    C
end

end
