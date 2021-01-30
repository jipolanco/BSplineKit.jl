module Collocation

export
    collocation_points,
    collocation_points!,
    collocation_matrix,
    collocation_matrix!

using ..BSplines
using ..DifferentialOps
using ..Recombinations:
    num_constraints,
    RecombinedBSplineBasis  # for Documenter

using BandedMatrices
using SparseArrays

"""
    SelectionMethod

Abstract type defining a method for choosing collocation points from spline
knots.
"""
abstract type SelectionMethod end

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

Note that if `B` is a [`RecombinedBSplineBasis`](@ref) (adapted for boundary value
problems), collocation points are not included at the boundaries, since the
boundary conditions are implicitly satisfied by the basis.

In principle, the choice of collocation points is not unique.
The selection method can be chosen via the `method` argument.
For now, only a single method is accepted:

- [`Collocation.AvgKnots()`](@ref)

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

function collocation_points!(::AvgKnots, x, B::AbstractBSplineBasis)
    N = length(B)           # number of functions in basis
    @assert length(x) == N
    k = order(B)
    t = knots(B)
    T = eltype(x)

    # For recombined bases, skip points at the boundaries.
    # Note that j = 0 for non-recombined bases, i.e. boundaries are included.
    j, _ = num_constraints(B)

    v::T = inv(k - 1)
    a::T, b::T = boundaries(B)

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

# TODO should this be a square matrix?
"""
    CollocationMatrix{T}

B-spline collocation matrix, defined by

```math
C_{ij} = b_j(x_i)
```

where ``\\bm{x}`` are a set of collocation points.

Wraps a [`BandedMatrix`](@ref)
"""
struct CollocationMatrix{T} <: AbstractMatrix{T}
end

"""
    collocation_matrix(
        B::AbstractBSplineBasis,
        x::AbstractVector,
        [deriv::Derivative = Derivative(0)],
        [MatrixType = SparseMatrixCSC{Float64}];
        clip_threshold = eps(eltype(MatrixType)),
    )

Return collocation matrix mapping B-spline coefficients to spline values at the
collocation points `x`.

# Extended help

The matrix elements are given by the B-splines evaluated at the collocation
points:

```math
C_{ij} = b_j(x_i) \\quad \\text{for} \\quad
i ∈ [1, N_x] \\text{ and } j ∈ [1, N_b],
```

where `Nx = length(x)` is the number of collocation points, and
`Nb = length(B)` is the number of B-splines in `B`.

To obtain a matrix associated to the B-spline derivatives, set the `deriv`
argument to the order of the derivative.

Given the B-spline coefficients ``\\{u_j, 1 ≤ j ≤ N_b\\}``, one can recover the
values (or derivatives) of the spline at the collocation points as `v = C * u`.
Conversely, if one knows the values ``v_i`` at the collocation points,
the coefficients ``u`` of the spline passing by the collocation points may be
obtained by inversion of the linear system `u = C \\ v`.

The `clip_threshold` argument allows one to ignore spurious, negligible values
obtained when evaluating B-splines. These values are typically unwanted, as they
artificially increase the number of elements (and sometimes the bandwidth) of
the matrix.
They may appear when a collocation point is located on a knot.
By default, `clip_threshold` is set to the machine epsilon associated to the
matrix element type (see
[`eps`](https://docs.julialang.org/en/v1/base/base/#Base.eps-Tuple{Type{var%22#s23%22}%20where%20var%22#s23%22%3C:AbstractFloat})).
Set it to zero to disable this behaviour.

## Matrix types

The `MatrixType` optional argument allows to select the type of returned matrix.

Due to the compact support of B-splines, the collocation matrix is
[banded](https://en.wikipedia.org/wiki/Band_matrix) if the collocation points
are properly distributed. Therefore, it makes sense to store it in a
`BandedMatrix` (from the
[BandedMatrices](https://github.com/JuliaMatrices/BandedMatrices.jl) package),
as this will lead to memory savings and especially to time savings if
the matrix needs to be inverted.

### Supported matrix types

- `SparseMatrixCSC{T}`: this is the default, as it correctly handles any matrix
  shape.

- `BandedMatrix{T}`: generally performs better than sparse matrices for inversion
  of linear systems. On the other hand, for matrix-vector or matrix-matrix
  multiplications, `SparseMatrixCSC` may perform better (see
  [BandedMatrices issue](https://github.com/JuliaMatrices/BandedMatrices.jl/issues/110)).
  May fail with an error for non-square matrix shapes, or if the distribution of
  collocation points is not adapted. In these cases, the effective
  bandwidth of the matrix may be larger than the expected bandwidth.

- `Matrix{T}`: a regular dense matrix. Generally performs worse than the
  alternatives, especially for large problems.

See also [`collocation_matrix!`](@ref).
"""
function collocation_matrix(
        B::AbstractBSplineBasis, x::AbstractVector,
        deriv::Derivative = Derivative(0),
        ::Type{MatrixType} = SparseMatrixCSC{Float64};
        kwargs...) where {MatrixType}
    Nx = length(x)
    Nb = length(B)
    C = allocate_collocation_matrix(MatrixType, (Nx, Nb), order(B); kwargs...)
    collocation_matrix!(C, B, x, deriv; kwargs...)
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
    bands = collocation_bandwidths(k)
    M(undef, dims, bands)
end

collocation_bandwidths(k) = (k - 2, k - 2)

"""
    collocation_matrix!(
        C::AbstractMatrix{T}, B::AbstractBSplineBasis, x::AbstractVector,
        [deriv::Derivative = Derivative(0)]; clip_threshold = eps(T))

Fill preallocated collocation matrix.

See [`collocation_matrix`](@ref) for details.
"""
function collocation_matrix!(
        C::AbstractMatrix{T}, B::AbstractBSplineBasis, x::AbstractVector,
        deriv::Derivative = Derivative(0);
        clip_threshold = eps(T)) where {T}
    Nx, Nb = size(C)

    if Nx != length(x) || Nb != length(B)
        throw(ArgumentError("wrong dimensions of collocation matrix"))
    end

    fill!(C, 0)
    b_lo, b_hi = max_bandwidths(C)

    for j = 1:Nb, i = 1:Nx
        b = evaluate(B, j, x[i], deriv, T)

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

# Maximum number of bandwidths allowed in a matrix container.
max_bandwidths(A::BandedMatrix) = bandwidths(A)
max_bandwidths(A::AbstractMatrix) = size(A) .- 1

# LU factorisation without pivoting of banded matrix.
# Takes advantage of the totally positive property of collocation matrices
# appearing in spline calculations (de Boor 1978).
# The code is ported from Carl de Boor's BANFAC routine in FORTRAN77, via its
# FORTRAN90 version by John Burkardt.
function lu_no_pivot!(A::BandedMatrix)
    w = BandedMatrices.bandeddata(A)
    nbandl, nbandu = bandwidths(A)
    nrow = size(A, 1)
    nroww = size(w, 1)
    # @assert nrow == size(A, 2) == size(w, 2)
    @assert nroww == nbandl + nbandu + 1
    isempty(A) && error("matrix is empty")
    middle = nbandu + 1  # w[middle, :] contains the main diagonal of A

    # TODO
    # - add @inbounds
    # - test all 3 possible cases
    # - non-square matrices?

    if nrow == 1 && iszero(w[middle, nrow])
        error("singular matrix")
    end

    if nbandl == 0
        # A is upper triangular. Check that the diagonal is nonzero.
        for i = 1:nrow
            iszero(w[middle, i]) && error("upper triangular matrix has zero diagonal")
        end
        return A
    end

    if nbandu == 0
        # A is lower triangular. Check that the diagonal is nonzero and
        # divide each column by its diagonal.
        for i = 1:(nrow - 1)
            pivot = w[middle, i]
            iszero(pivot) && error("lower triangular matrix has zero diagonal")
            ipiv = inv(pivot)
            for j = 1:min(nbandl, nrow - i)
                w[middle + j, i] *= ipiv
            end
        end
        return A
    end

    # A is not just a triangular matrix.
    # Construct the LU factorization.
    for i = 1:(nrow - 1)
        pivot = w[middle, i]  # pivot for the i-th step
        iszero(pivot) && error("zero pivot encountered")
        ipiv = inv(pivot)

        # Divide each entry in column `i` below the diagonal by `pivot`.
        for j = 1:min(nbandl, nrow - i)
            w[middle + j, i] *= ipiv
        end

        # Subtract A[i, i+k] * (i-th column) from (i+k)-th column (below row `i`).
        for k = 1:min(nbandu, nrow - i)
            factor = w[middle - k, i + k]
            for j = 1:min(nbandl, nrow - i)
                w[middle - k + j, i + k] -= factor * w[middle + j, i]
            end
        end
    end

    # Check the last diagonal entry.
    iszero(w[middle, nrow]) && error("matrix is singular")

    A
end

# Solution of banded linear system A * x = y factorised by lu_no_pivot!.
# Takes advantage of the totally positive property of collocation matrices
# appearing in spline calculations (de Boor 1978).
# The code is ported from Carl de Boor's BANSLV routine in FORTRAN77, via its
# FORTRAN90 version by John Burkardt.
function ldiv_no_pivot!(A::BandedMatrix, b::AbstractVector)
    w = BandedMatrices.bandeddata(A)
    nroww = size(w, 1)
    nrow = size(A, 1)
    nbandl, nbandu = bandwidths(A)

    middle = nbandu + 1

    # TODO
    # - @inbounds
    # - test two possible cases
    # - non-square matrices?

    if nrow == 1
        b /= w[middle, 1]
        return b
    end

    # Forward pass:
    #
    # For i = 1:(nrow-1), subtract RHS[i] * (i-th column of L) from the
    # right hand side, below the i-th row.
    if nbandl != 0
        for i = 1:(nrow - 1)
            jmax = min(nbandl, nrow - i)
            for j = 1:jmax
                b[i + j] -= b[i] * w[middle + j, i]
            end
        end
    end

    # Backward pass:
    #
    # For i = nrow:-1:1, divide RHS[i] by the i-th diagonal entry of
    # U, then subtract RHS[i]*(i-th column of U) from right hand side, above the
    # i-th row.
    for i = nrow:-1:2
        b[i] /= w[middle, i]
        for j = 1:min(nbandu, i - 1)
            b[i - j] -= b[i] * w[middle - j, i]
        end
    end

    b[1] /= w[middle, 1]

    b
end

end
