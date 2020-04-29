module Collocation

using ..BasisSplines
import BSplines
using BSplines: Derivative

using BandedMatrices
using SparseArrays

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

"""
    collocation_points(B::AbstractBSplineBasis; method=Collocation.AvgKnots())

Define and return adapted collocation points for evaluation of splines.

The number of returned collocation points is equal to the number of functions in
the basis.

Collocation points can be selected in different ways.
The selection method can be chosen via the `method` argument, which accepts the
following values:

- [`Collocation.AvgKnots()`](@ref) (this is the default)
- [`Collocation.AtMaxima()`](@ref) (actually, not yet supported!)

See also [`collocation_points!`](@ref).
"""
function collocation_points(
        B::AbstractBSplineBasis; method::SelectionMethod=AvgKnots())
    x = similar(knots(B), length(B))
    collocation_points!(x, B, method=method)
end

"""
    collocation_points!(x::AbstractVector, B::AbstractBSplineBasis;
                        method::SelectionMethod=AvgKnots())

Fill vector with collocation points for evaluation of splines.

See [`collocation_points`](@ref) for details.
"""
function collocation_points!(
        x::AbstractVector, B::AbstractBSplineBasis;
        method::SelectionMethod=AvgKnots())
    N = length(B)
    if N != length(x)
        throw(ArgumentError(
            "number of collocation points must match number of B-splines"))
    end
    collocation_points!(method, x, B)
end

# TODO make this work with RecombinedBSplineBasis (remove boundaries!)
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
    collocation_matrix(
        B::AbstractBSplineBasis,
        x::AbstractVector,
        [deriv::Derivative = Derivative(0)],
        [MatrixType = SparseMatrixCSC{Float64}];
        bc = nothing,
        clip_threshold = eps(eltype(MatrixType)),
    )

Return banded collocation matrix mapping B-spline coefficients to spline values
at the collocation points `x`.

The matrix elements are given by the B-splines evaluated at the collocation
points:

    C[i, j] = bⱼ(x[i])  for  i = 1:Nx and j = 1:Nb,

where `Nx = length(x)` is the number of collocation points, and
`Nb = length(B)` is the number of B-splines in `B`.

To obtain a matrix associated to the B-spline derivatives, set the `deriv`
argument to the order of the derivative.

Given the B-spline coefficients `{u[j], j = 1:Nb}`, one can recover the values
(or derivatives) of the spline at the collocation points as `v = C * u`.
Conversely, if one knows the values `v[i]` at the collocation points,
the coefficients `u` can be obtained by inversion of the linear system
`u = C \\ v`.

The `clip_threshold` argument allows to ignore spurious, very small values
obtained when evaluating B-splines. These values are typically unwanted, as they
artificially increase the number of elements (and sometimes the bandwidth) of
the matrix.
They may appear when a collocation point is located on a knot.
By default, `clip_threshold` is set to the machine epsilon associated to the
matrix element type (see
[`eps`](https://docs.julialang.org/en/v1/base/base/#Base.eps-Tuple{Type{#s4}%20where%20#s4%3C:AbstractFloat})).
Set it to zero to disable this behaviour.

## Matrix types

The `MatrixType` optional argument allows to select the type of returned matrix.

Due to the compact support of B-splines, the collocation matrix is
[banded](https://en.wikipedia.org/wiki/Band_matrix) if the collocation points
have a "good" distribution. Therefore, it makes sense to store it in a
`BandedMatrix` (from the
[BandedMatrices](https://github.com/JuliaMatrices/BandedMatrices.jl) package),
as this will lead to memory savings and especially to time savings if
the matrix needs to be inverted.

### Supported matrix types

- `SparseMatrixCSC{T}`: this is the default, as it correctly handles any matrix
shape.

- `BandedMatrix{T}`: usually performs better than sparse matrices for inversion
of linear systems. May not work for non-square matrix shapes, or if the
distribution of collocation points is not adapted. In these cases, the effective
bandwidth of the matrix may be larger than the expected bandwidth.

- `Matrix{T}`: a regular dense matrix. Generally performs worse than the
alternatives, especially for large problems.

## Boundary conditions

Basis recombination (see Boyd 2000, ch. 6) is a common technique for imposing
boundary conditions (BCs) in Galerkin methods. In this approach, the basis is
"recombined" so that each basis function individually satisfies the BCs.

The new basis, `{ϕⱼ(x), j ∈ 1:(N-2)}`, has two fewer functions than the original
B-spline basis, `{bⱼ(x), j ∈ 1:N}`. Due to this, the number of collocation
points needed to obtain a square collocation matrix is `N - 2`. In particular,
for the matrix to be invertible, there must be **no** collocation points at the
boundaries.

Thanks to the local support of B-splines, basis recombination involves only a
little portion of the original B-spline basis. For instance, since there is only
one B-spline that is non-zero at each boundary, removing that function from the
basis is enough to apply homogeneous Dirichlet BCs. Imposing BCs for derivatives
is a bit more complex, but not much.

The optional keyword argument `bc` allows specifying homogeneous BCs. It accepts
a `Derivative` object, which sets the order of the derivative to be imposed with
homogeneous BCs. Some typical choices are:

- `bc = Derivative(0)` sets homogeneous Dirichlet BCs (`u = 0` at the
  boundaries) by removing the first and last B-splines, i.e. ϕ₁ = b₂;

- `bc = Derivative(1)` sets homogeneous Neumann BCs (`du/dx = 0` at the
  boundaries) by adding the two first (and two last) B-splines,
  i.e. ϕ₁ = b₁ + b₂.

For now, the two boundaries are given the same BC (but this could be easily
extended...). And actually, only the two above choices are available for now!

See also [`collocation_matrix!`](@ref).
"""
function collocation_matrix(
        B::AbstractBSplineBasis, x::AbstractVector,
        deriv::Derivative = Derivative(0),
        ::Type{MatrixType} = SparseMatrixCSC{Float64};
        bc=nothing, kwargs...) where {MatrixType}
    Nx = length(x)
    Nb = length(B)
    if bc !== nothing
        Nb -= 2  # remove two basis functions if BCs are applied
    end
    C = allocate_collocation_matrix(MatrixType, (Nx, Nb), order(B); kwargs...)
    collocation_matrix!(C, B, x, deriv; bc=bc, kwargs...)
end

collocation_matrix(B, x, ::Type{M}; kwargs...) where {M} =
    collocation_matrix(B, x, Derivative(0), M; kwargs...)

allocate_collocation_matrix(::Type{M}, dims, k) where {M <: AbstractMatrix} =
    M(undef, dims...)

allocate_collocation_matrix(::Type{SparseMatrixCSC{T}}, dims, k) where {T} =
    spzeros(T, dims...)

function allocate_collocation_matrix(::Type{M}, dims, k) where {M <: BandedMatrix}
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
    M(undef, dims, (Nb, Nb))
end

"""
    collocation_matrix!(
        C::AbstractMatrix{T}, B::AbstractBSplineBasis, x::AbstractVector,
        [deriv::Derivative = Derivative(0)];
        bc = nothing, clip_threshold = eps(T))

Fill preallocated collocation matrix.

See [`collocation_matrix`](@ref) for details.
"""
function collocation_matrix!(
        C::AbstractMatrix{T}, B::AbstractBSplineBasis, x::AbstractVector,
        deriv::Derivative = Derivative(0);
        bc = nothing,
        clip_threshold = eps(T)) where {T}
    Nx, Nb = size(C)
    with_bc = bc !== nothing

    if Nx != length(x) || Nb != length(B) - 2 * with_bc
        throw(ArgumentError("wrong dimensions of collocation matrix"))
    end

    fill!(C, 0)
    b_lo, b_hi = bandwidths(C)

    for j = 1:Nb, i = 1:Nx
        b = recombine_bspline(bc, B, j, x[i], deriv, T)

        # Skip very small values (and zeros).
        # This is important for SparseMatrixCSC, which also stores explicit
        # zeros.
        abs(b) <= clip_threshold && continue

        if (i > j && i - j > b_lo) || (i < j && j - i > b_hi)
            # This will cause problems if C is a BandedMatrix, and (i, j) is
            # outside the allowed bands. This may be the case if the collocation
            # points are not properly distributed.
            @warn "Non-zero value outside of matrix bands: b[$j](x[$i]) = $b"
        end

        C[i, j] = b
    end

    C
end

# TODO define RecombinedBSplineBasis?
# Perform basis recombination for applying homogeneous BCs.
# No basis recombination.
recombine_bspline(::Nothing, B, j, args...) =
    evaluate_bspline(B, j, args...)

# For homogeneous Dirichlet BCs: just shift the B-spline basis (removing b₁).
recombine_bspline(::Derivative{0}, B, j, args...) =
    evaluate_bspline(B, j + 1, args...)

# Homogeneous Neumann BCs.
function recombine_bspline(::Derivative{1}, B, j, args...)
    Nb = length(B) - 2  # length of recombined basis
    if j == 1
        evaluate_bspline(B, 1, args...) + evaluate_bspline(B, 2, args...)
    elseif j == Nb
        evaluate_bspline(B, Nb + 1, args...) + evaluate_bspline(B, Nb + 2, args...)
    else
        # Same as for Dirichlet.
        recombine_bspline(Derivative(0), B, j, args...)
    end
end

end
