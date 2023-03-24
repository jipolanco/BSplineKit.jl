module Collocation

export
    CollocationMatrix,
    collocation_points,
    collocation_points!,
    collocation_matrix,
    collocation_matrix!

using ..BSplines
using ..DifferentialOps
using ..Recombinations: num_constraints

using BandedMatrices
using SparseArrays
using StaticArrays: Size, SVector, MVector, SMatrix, MMatrix

# For Documenter only:
using ..Recombinations: RecombinedBSplineBasis
using ..BoundaryConditions: Natural

include("matrix.jl")
include("points.jl")

"""
    collocation_matrix(
        B::AbstractBSplineBasis,
        x::AbstractVector,
        [deriv::Derivative = Derivative(0)],
        [MatrixType = CollocationMatrix{Float64}];
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
matrix element type (see `eps` in the Julia documentation).
Set it to zero to disable this behaviour.

## Matrix types

The `MatrixType` optional argument allows to select the type of returned matrix.

Due to the compact support of B-splines, the collocation matrix is
[banded](https://en.wikipedia.org/wiki/Band_matrix) if the collocation points
are properly distributed. Therefore, it makes sense to store it in a
`BandedMatrix` (from the
[BandedMatrices](https://github.com/JuliaLinearAlgebra/BandedMatrices.jl) package),
as this will lead to memory savings and especially to time savings if
the matrix needs to be inverted.

### Supported matrix types

- `CollocationMatrix{T}`: thin wrapper over `BandedMatrix{T}`, with efficient LU
  factorisations without pivoting (see [`CollocationMatrix`](@ref) for details).
  This option performs much better than sparse matrices for inversion of linear
  systems.
  On the other hand, for matrix-vector or matrix-matrix
  multiplications, `SparseMatrixCSC` may perform better, especially when using OpenBLAS (see
  [BandedMatrices issue](https://github.com/JuliaLinearAlgebra/BandedMatrices.jl/issues/110)).
  May fail with an error for non-square matrix shapes, or if the distribution of
  collocation points is not adapted. In these cases, the effective
  bandwidth of the matrix may be larger than the expected bandwidth.

- `SparseMatrixCSC{T}`: regular sparse matrix; correctly handles all matrix
  shapes.

- `Matrix{T}`: a regular dense matrix. Generally performs worse than the
  alternatives, especially for large problems.

!!! note "Periodic B-spline bases"
    The default matrix type is `CollocationMatrix{T}`, *except* for
    periodic bases ([`PeriodicBSplineBasis`](@ref)), in which case the collocation
    matrix has a few out-of-bands entries due to periodicity.
    For periodic bases, `SparseMatrixCSC` is the default.
    Note that this may change in the future.

See also [`collocation_matrix!`](@ref).
"""
function collocation_matrix(
        B::AbstractBSplineBasis, x::AbstractVector,
        deriv::Derivative = Derivative(0),
        ::Type{MatrixType} = default_matrix_type(B, Float64);
        kwargs...,
    ) where {MatrixType}
    C = allocate_collocation_matrix(MatrixType, x, B; kwargs...)
    collocation_matrix!(C, B, x, deriv; kwargs...)
    consolidate_matrix(C)
end

consolidate_matrix(C) = C
consolidate_matrix(C::CollocationMatrix) = consolidate_matrix(C, bandeddata(C))
consolidate_matrix(C::CollocationMatrix, bandeddata) = C

function consolidate_matrix(C::CollocationMatrix, bandeddata::MMatrix)
    ncols = size(C, 2)
    l, u = bandwidths(C)
    CollocationMatrix(
        BandedMatrices._BandedMatrix(SMatrix(bandeddata), ncols, l, u)
    )
end

collocation_matrix(B, x, ::Type{M}; kwargs...) where {M} =
    collocation_matrix(B, x, Derivative(0), M; kwargs...)

default_matrix_type(::Type{<:AbstractBSplineBasis}, ::Type{T}) where {T} =
    CollocationMatrix{T}
default_matrix_type(::Type{<:PeriodicBSplineBasis}, ::Type{T}) where {T} =
    SparseMatrixCSC{T}
default_matrix_type(B::AbstractBSplineBasis, ::Type{T}) where {T} =
    default_matrix_type(typeof(B), T)

allocate_collocation_matrix(::Type{M}, x, B) where {M <: AbstractMatrix} =
    M(undef, length(x), length(B))

allocate_collocation_matrix(::Type{SparseMatrixCSC{T}}, x, B) where {T} =
    spzeros(T, length(x), length(B))

function allocate_collocation_matrix(
        ::Type{M}, x, B,
    ) where {M <: CollocationMatrix}
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
    k = order(B)
    bands = collocation_bandwidths(k)
    l, u = bands
    nrows = l + u + 1
    ncols = length(x)
    T = eltype(M)
    # data_raw = Matrix{T}(undef, nrows, ncols)
    data_raw = _collocation_matrix_raw(Val(nrows), x, T)
    data = BandedMatrices._BandedMatrix(data_raw, ncols, l, u)
    CollocationMatrix(data) :: M
end

_collocation_matrix_raw(::Val{nrows}, xs::AbstractVector, ::Type{T}) where {nrows, T} =
    Matrix{T}(undef, nrows, length(xs))

_collocation_matrix_raw(::Val{nrows}, xs::SVector{Nx}, ::Type{T}) where {nrows, Nx, T} =
    MMatrix{nrows, Nx, T, nrows * Nx}(undef)

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
        clip_threshold = eps(T),
    ) where {T <: AbstractFloat}
    if axes(C, 1) != axes(x, 1) || size(C, 2) != length(B)
        throw(ArgumentError("wrong dimensions of collocation matrix"))
    end

    fill!(C, 0)
    b_lo, b_hi = max_bandwidths(C)

    @inbounds for (i, xi) ∈ pairs(x)
        # If x is not in the domain of the basis, then all basis functions
        # evaluate to zero.
        isindomain(B, xi) || continue

        jlast, bs = evaluate_all(B, xi, deriv, T)  # = (b_{jlast + 1 - k}(x), …, b_{jlast}(x))

        for (δj, bj) ∈ pairs(bs)
            j = jlast + 1 - δj

            # This is relevant for periodic bases.
            j = basis_to_array_index(B, axes(C, 2), j)

            # Skip very small values (and zeros).
            # This is important for SparseMatrixCSC, which also stores explicit
            # zeros.
            abs(bj) <= clip_threshold && continue

            if (i > j && i - j > b_lo) || (i < j && j - i > b_hi)
                # This will cause problems if C is a BandedMatrix, and (i, j) is
                # outside the allowed bands. This may be the case if the collocation
                # points are not properly distributed.
                @warn lazy"Non-zero value outside of matrix bands: b[$j]($xi) = $bj"
            end

            C[i, j] = bj
        end
    end

    C
end

# Maximum number of bandwidths allowed in a matrix container.
max_bandwidths(A::BandedMatrix) = bandwidths(A)
max_bandwidths(A::CollocationMatrix) = max_bandwidths(parent(A))
max_bandwidths(A::AbstractMatrix) = size(A) .- 1

end
