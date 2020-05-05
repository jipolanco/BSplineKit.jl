module BandedTensors

using StaticArrays

export BandedTensor3D

"""
    BandedTensor3D{T,b}

Three-dimensional banded cubic tensor with element type `T`.

# Band structure

The band structure is assumed to be symmetric, and is defined in terms of the
band width `b`.
The element `A[i, j, k]` may be non-zero only if `|i - j| ≤ b`, `|i - k| ≤ b`
and `|j - k| ≤ b`.

# Storage

The data is stored as a `Vector` of small matrices, each with size
`(r, r)`, where `r = 2b + 1` is the total number of bands.

For `b = 2`, one of these matrices looks like the following, where dots indicate
out-of-bands values:

    | x  x  x  ⋅  ⋅ |
    | x  x  x  x  ⋅ |
    | x  x  x  x  x |
    | ⋅  x  x  x  x |
    | ⋅  ⋅  x  x  x |

These submatrices are stored as static matrices (`SMatrix`).
"""
struct BandedTensor3D{T, b, r, M <: AbstractMatrix} <: AbstractArray{T,3}
    N    :: Int        # dimension
    data :: Vector{M}  # vector of submatrices indexed by (i, j)

    function BandedTensor3D{T,b}(::UndefInitializer, N) where {T,b}
        b :: Int
        r = 2b + 1
        M = SMatrix{r, r, T, r * r}
        @assert isconcretetype(M)
        data = Vector{M}(undef, N)
        new{T, b, r, M}(N, data)
    end
end

"""
    BandedTensor3D{T}(undef, N, b)

Construct cubic 3D banded tensor with bandwidths `b` and dimensions `N`×`N`×`N`.

The matrix is constructed uninitialised.

Each submatrix `A[:, :, k]` of size `(2b + 1, 2b + 1)`, for `k ∈ 1:N`, should be
initialised as in the following example:

    A[:, :, k] = rand(2b + 1, 2b + 1)

"""
@inline BandedTensor3D{T}(init, N, b::Int) where {T} =
    BandedTensor3D{T,b}(init, N)

Base.sizeof(A::BandedTensor3D) = sizeof(A.data)

function submatrix(A::BandedTensor3D, k)
    @boundscheck checkbounds(A.data, k)
    @inbounds A.data[k]
end

submatrix_type(::Type{BandedTensor3D{T,b,r,M}}) where {T,b,r,M} = M
submatrix_type(A::BandedTensor3D) = submatrix_type(typeof(A))

"""
    bandwidth(A::BandedTensor3D)

Get band width `b` of [`BandedTensor3D`](@ref).

The band width is defined here such that the element `A[i, j, k]` may be
non-zero only if `|i - j| ≤ b`, `|i - k| ≤ b` and `|j - k| ≤ b`.
This definition is consistent with the specification of the upper and lower band
widths in `BandedMatrices`.
"""
bandwidth(A::BandedTensor3D{T,b}) where {T,b} = b

function Base.size(A::BandedTensor3D)
    N = A.N
    (N, N, N)
end

"""
    band_axes(A::BandedTensor3D, k)

Return the range of indices `(i1:i2, j1:j2)` for subarray `A[:, :, k]`
where values may be non-zero.
"""
@inline function band_axes(A::BandedTensor3D, k)
    b = bandwidth(A)
    ax = (k - b):(k + b)
    ax, ax
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
Base.getindex(A::BandedTensor3D, ::Colon, ::Colon, k) = submatrix(A, k)

@inline function Base.getindex(A::BandedTensor3D,
                               i::Integer, j::Integer, k::Integer)
    @boundscheck checkbounds(A, i, j, k)
    ri, rj = band_axes(A, k)
    (i ∈ ri) && (j ∈ rj) || return zero(eltype(A))
    @inbounds Ak = submatrix(A, k)
    b = bandwidth(A)
    ii = i - first(ri) + 1
    jj = j - first(rj) + 1
    @inbounds Ak[ii, jj]
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

end
