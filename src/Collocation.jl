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

"""
    collocation_matrix(B::BSplineBasis, x::AbstractVector,
                       [MatrixType=BandedMatrix{Float64}];
                       Ndiff::Val = Val(0))

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

To obtain a matrix associated to a B-spline derivative, set the `Ndiff` argument.

See also [`collocation_matrix!`](@ref).
"""
function collocation_matrix(B::BSplineBasis, x::AbstractVector,
        ::Type{MatrixType} = BandedMatrix{Float64};
        kwargs...) where {MatrixType}
    C = allocate_collocation_matrix(MatrixType, length(B), order(B))
    collocation_matrix!(C, B, x; kwargs...)
end

allocate_collocation_matrix(::Type{M}, N, k) where {M <: AbstractMatrix} =
    M(undef, N, N)

function allocate_collocation_matrix(::Type{M}, N, k) where {M <: BandedMatrix}
    # Number of upper and lower diagonal bands.
    #
    # The matrices **almost** have the following number of upper/lower bands (as
    # cited sometimes in the literature):
    #
    #  - for even k: Nb = (k - 2) / 2 (total = k - 1 bands)
    #  - for odd  k: Nb = (k - 1) / 2 (total = k bands)
    #
    # However, near the boundaries U/L bands can become much larger.
    # Specifically for the second and second-to-last collocation points (j = 2
    # and N - 1). For instance, the point j = 2 sees B-splines in 1:k, leading
    # to an upper band of size k - 2.
    #
    # TODO is there a way to reduce the number of bands??
    Nb = k - 2
    M(undef, (N, N), (Nb, Nb))
end

"""
    collocation_matrix!(C::AbstractMatrix{T}, B::BSplineBasis,
                        x::AbstractVector; Ndiff::Val = Val(0))

Fill preallocated collocation matrix.

See [`collocation_matrix`](@ref) for details.
"""
function collocation_matrix!(C::AbstractMatrix{T}, B::BSplineBasis,
                             x::AbstractVector; Ndiff::Val = Val(0)) where {T}
    checkdims(B, x)
    N, N2 = size(C)
    if !(N == N2 == length(B))
        throw(ArgumentError("wrong dimensions of collocation matrix"))
    end
    fill!(C, 0)
    b_lo, b_hi = bandwidths(C)
    for j = 1:N, i = 1:N
        b = evaluate_bspline(B, j, x[i], T, Ndiff=Ndiff)
        if !iszero(b) && ((i > j && i - j > b_lo) || (i < j && j - i > b_hi))
            # This can happen if C is a BandedMatrix, and (i, j) is outside
            # the allowed bands. This may be the case if the collocation
            # points are not properly distributed.
            @warn "Non-zero value outside of matrix bands: b[$j](x[$i]) = $b"
        end
        C[i, j] = b
    end
    C
end

end
