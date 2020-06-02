module BandedTensors

using StaticArrays

import BandedMatrices: bandwidth

using LinearAlgebra: dot
import LinearAlgebra

export BandedTensor3D
export bandshift, bandwidth

"""
    BandedTensor3D{T,b}

Three-dimensional banded tensor with element type `T`.

# Extended help

## Band structure

The band structure is assumed to be symmetric, and is defined in terms of the
band width ``b``.
For a cubic banded tensor of dimensions ``N × N × N``, the element ``A_{ijk}``
may be non-zero only if ``|i - j| ≤ b``, ``|i - k| ≤ b`` and ``|j - k| ≤ b``.

## Storage

The data is stored as a `Vector` of small matrices, each with size
``r × r``, where ``r = 2b + 1`` is the total number of bands.
Each submatrix holds the non-zero values of a slice of the form `A[:, :, k]`.

For ``b = 2``, one of these matrices looks like the following, where dots indicate
out-of-bands values (equal to zero):

    | x  x  x  ⋅  ⋅ |
    | x  x  x  x  ⋅ |
    | x  x  x  x  x |
    | ⋅  x  x  x  x |
    | ⋅  ⋅  x  x  x |

These submatrices are stored as static matrices (`SMatrix`).

## Setting elements

To define the elements of the tensor, each slice `A[:, :, k]` must be set at
once. For instance:

```julia
A = BandedTensor3D(undef, 20, 2)  # tensor of size 20×20×20 and band width b = 2
for k in axes(A, 3)
    A[:, :, k] = rand(5, 5)
end
```

See [`setindex!`](@ref) for more details.

## Non-cubic tensors

A slight departure from cubic tensors is currently supported, with dimensions of
the form ``N × N × M``.
Moreover, bands may be shifted along the third dimension by an offset ``δ``.
In this case, the bands are given by ``|i - j| ≤ b``, ``|i - (k + δ)| ≤ b`` and
``|j - (k + δ)| ≤ b``.

---

    BandedTensor3D{T}(undef, (Ni, Nj, Nk), b; [bandshift = (0, 0, 0)])
    BandedTensor3D{T}(undef, N, b; ...)

Construct 3D banded tensor with band widths `b`.

Right now, the first two dimension sizes `Ni` and `Nj` of the tensor must be
equal.
In the second variant, the tensor dimensions are `N × N × N`.

The tensor is constructed uninitialised.
Each submatrix `A[:, :, k]` of size `(2b + 1, 2b + 1)`, for `k ∈ 1:Nk`, should be
initialised as in the following example:

    A[:, :, k] = rand(2b + 1, 2b + 1)

The optional `bandshift` argument should be a tuple of the form `(δi, δj, δk)`
describing a band shift.
Right now, band shifts are limited to `δi = δj = 0`, so this argument should
rather look like `(0, 0, δk)`.

"""
struct BandedTensor3D{T, b, r, M <: AbstractMatrix} <: AbstractArray{T,3}
    dims :: Dims{3}        # dimensions (with dims[1] == dims[2])
    bandshift :: Dims{3}  # band shifts (with bandshift[1] = bandshift[2] = 0)
    Nk   :: Int        # last dimension (= dims[3])
    δk   :: Int        # band shift along 3rd dimension (= bandshift[3])
    data :: Vector{M}  # vector of submatrices indexed by (i, j)

    function BandedTensor3D{T,b}(::UndefInitializer, Ni, Nj, Nk;
                                 bandshift::Dims{3} = (0, 0, 0)) where {T,b}
        b :: Int
        if Ni != Nj
            throw(ArgumentError(
                "the first two dimensions must have the same sizes"))
        end
        δi, δj, δk = bandshift
        if !(δi == δj == 0)
            throw(ArgumentError(
                "shifts along dimensions 1 and 2 are not currently supported"))
        end
        r = 2b + 1
        M = SMatrix{r, r, T, r * r}
        @assert isconcretetype(M)
        data = Vector{M}(undef, Nk)
        new{T, b, r, M}((Ni, Nj, Nk), bandshift, Nk, δk, data)
    end

    BandedTensor3D{T,b}(init, N::Integer; kwargs...) where {T,b} =
        BandedTensor3D{T,b}(init, N, N, N; kwargs...)

    BandedTensor3D{T,b}(init, dims::Dims; kwargs...) where {T,b} =
        BandedTensor3D{T,b}(init, dims...; kwargs...)
end

@inline BandedTensor3D{T}(init, dims, b::Int; kwargs...) where {T} =
    BandedTensor3D{T,b}(init, dims; kwargs...)

function Base.summary(io::IO, A::BandedTensor3D)
    print(io, Base.dims2string(size(A)), " BandedTensor3D{", eltype(A),
          "} with band width b = ", bandwidth(A),
          " and band shifts ", bandshift(A))
    nothing
end

Base.sizeof(A::BandedTensor3D) = sizeof(A.data)

"""
    SubMatrix{T} <: AbstractMatrix{T}

Represents the submatrix `A[:, :, k]` of a [`BandedTensor3D`](@ref) `A`.

Wraps the `SMatrix` holding the submatrix.
"""
struct SubMatrix{T, M <: SMatrix{T}, Indices <: AbstractRange} <: AbstractMatrix{T}
    dims :: Dims{2}
    data :: M
    k    :: Int
    inds :: Indices
end

@inline function SubMatrix(A::BandedTensor3D, k)
    @boundscheck checkbounds(A.data, k)
    dims = size(A, 1), size(A, 2)
    @inbounds SubMatrix(dims, A.data[k], k, band_indices(A, k))
end

Base.size(S::SubMatrix) = S.dims

function Base.show(io::IO, mime::MIME"text/plain", S::SubMatrix)
    summary(io, S)
    print(io, "\nData =\n")
    show(io, mime, S.data)
    nothing
end

function Base.summary(io::IO, S::SubMatrix)
    print(io, "Submatrix of BandedTensor3D holding data in (",
          S.inds, ", ", S.inds, ", ", S.k, ")")
    nothing
end

Base.parent(Asub::SubMatrix) = Asub.data

submatrix_type(::Type{BandedTensor3D{T,b,r,M}}) where {T,b,r,M} = M
submatrix_type(A::BandedTensor3D) = submatrix_type(typeof(A))

"""
    bandwidth(A::BandedTensor3D)

Get band width `b` of [`BandedTensor3D`](@ref).

The band width is defined here such that the element `A[i, j, k]` may be
non-zero only if ``|i - j| ≤ b``, ``|i - k| ≤ b`` and ``|j - k| ≤ b``.
This definition is consistent with the specification of the upper and lower band
widths in `BandedMatrices`.
"""
bandwidth(A::BandedTensor3D{T,b}) where {T,b} = b

"""
    bandshift(A::BandedTensor3D) -> (δi, δj, δk)

Return tuple with band shifts along each dimension.
"""
bandshift(A::BandedTensor3D) = A.bandshift

Base.size(A::BandedTensor3D) = A.dims

"""
    band_indices(A::BandedTensor3D, k)

Return the range of indices `a:b` for subarray `A[:, :, k]` where values may be
non-zero.
"""
@inline function band_indices(A::BandedTensor3D, k)
    b = bandwidth(A)
    k += A.δk
    (k - b):(k + b)
end

"""
    setindex!(A::BandedTensor3D, Ak::AbstractMatrix, :, :, k)

Set submatrix `A[:, :, k]` to the matrix `Ak`.

The `Ak` matrix must have dimensions `(r, r)`, where `r = 2b + 1` is the total
number of bands of `A`.
"""
function Base.setindex!(A::BandedTensor3D, Ak::AbstractMatrix,
                        ::Colon, ::Colon, k)
    @boundscheck checkbounds(A, :, :, k)
    @boundscheck size(Ak) === (size(A, 1), size(A, 2))
    @inbounds A.data[k] = Ak
end

# Get submatrix A[:, :, k].
Base.getindex(A::BandedTensor3D, ::Colon, ::Colon, k::Integer) = SubMatrix(A, k)
Base.view(A::BandedTensor3D, ::Colon, ::Colon, k::Integer) = A[:, :, k]

function Base.getindex(A::BandedTensor3D, ::Colon, ::Colon,
                       ks::AbstractUnitRange)
    shift = bandshift(A) .+ (0, 0, first(ks) - 1)
    Nk = length(ks)
    dims = (size(A, 1), size(A, 2), Nk)
    # TODO define and use `similar`
    B = BandedTensor3D{eltype(A)}(undef, dims, bandwidth(A), bandshift=shift)
    for (l, k) in enumerate(ks)
        B[:, :, l] = parent(A[:, :, k])
    end
    B
end

@inline function Base.getindex(A::BandedTensor3D,
                               i::Integer, j::Integer, k::Integer)
    @boundscheck checkbounds(A, i, j, k)
    @inbounds Asub = SubMatrix(A, k)
    ri = Asub.inds
    rj = Asub.inds
    (i ∈ ri) && (j ∈ rj) || return zero(eltype(A))
    b = bandwidth(A)
    ii = i - first(ri) + 1
    jj = j - first(rj) + 1
    @inbounds parent(Asub)[ii, jj]
end

function Base.fill!(A::BandedTensor3D, x)
    M = submatrix_type(A)
    T = eltype(M)
    L = length(M)
    Ak = M(ntuple(_ -> T(x), L))
    for k in axes(A, 3)
        A[:, :, k] = Ak
    end
    A
end

"""
    dot(x, Asub::SubMatrix, y)

Efficient implementation of the generalised dot product `dot(x, Asub * y)`.

To be used with a submatrix `Asub = A[:, :, k]` of a [`BandedTensor3D`](@ref) `A`.
"""
function LinearAlgebra.dot(u::AbstractVector, S::SubMatrix, v::AbstractVector)
    # TODO inbounds / checkbounds
    @boundscheck @assert axes(u) === axes(v)
    Base.require_one_based_indexing(u, v)

    A = parent(S) :: SMatrix

    uind = axes(u, 1)

    if issubset(S.inds, uind)
        @inbounds usub = @view u[S.inds]
        @inbounds vsub = @view v[S.inds]
        return dot(usub, A * vsub)
    end

    # Boundaries
    vec_inds = intersect(S.inds, uind)
    @inbounds usub = @view u[vec_inds]
    @inbounds vsub = @view v[vec_inds]

    l = length(vec_inds)
    n = size(A, 1)
    @assert n > l

    mat_inds = if first(S.inds) <= 0
        (n - l + 1):n  # lower right corner
    else
        1:l  # upper left corner
    end
    @inbounds Asub = @view A[mat_inds, mat_inds]

    dot(usub, Asub, vsub)  # requires Julia 1.4!!
end

end
