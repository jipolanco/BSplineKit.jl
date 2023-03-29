using ArrayLayouts: MemoryLayout
using BandedMatrices

import BandedMatrices:
    AbstractBandedMatrix, bandwidths, bandeddata

using LinearAlgebra
import LinearAlgebra:
    LU, ZeroPivotException, BlasInt,
    ldiv!, lu!, lu, factorize

import Base: @propagate_inbounds

@static if !isdefined(LinearAlgebra, :NoPivot)
    const NoPivot = Val{false}
end

"""
    CollocationMatrix{T} <: AbstractBandedMatrix{T}

B-spline collocation matrix, defined by

```math
C_{ij} = b_j(x_i),
```

where ``\\bm{x}`` is a set of collocation points.

Provides an efficient LU factorisation without pivoting adapted from de Boor (1978).
The factorisation takes advantage of the total positivity of spline collocation
matrices (de Boor 2001, p. 175).

# Factorisation

`CollocationMatrix` supports in-place LU factorisation using [`lu!`](@ref), as
well as out-of-place factorisation using [`lu`](@ref). LU decomposition may also
be performed via `factorize`.

"""
struct CollocationMatrix{
        T,
        M <: AbstractMatrix{T},
    } <: AbstractBandedMatrix{T}
    # TODO try using HybridArrays?
    data :: M  # (Nbands, Nx)
    function CollocationMatrix(data::AbstractMatrix{T}) where {T}
        @assert !(data isa BandedMatrix)
        @assert isodd(size(data, 1))  # odd number of bands (= l + u + 1 = 2l + 1)
        new{T, typeof(data)}(data)
    end
end

# For compatibility with old versions.
function CollocationMatrix(B::BandedMatrix)
    l, u = bandwidths(B)
    l == u || error("expected same number of lower and upper bands")
    CollocationMatrix(bandeddata(B))
end

# This is to reuse BandedMatrices operations
function Base.convert(::Type{BandedMatrix}, C::CollocationMatrix)
    n = size(C, 1)
    l, u = bandwidths(C)
    BandedMatrices._BandedMatrix(C.data, n, l, u)
end

# This affects printing e.g. in the REPL.
MemoryLayout(::Type{<:CollocationMatrix{T,M}}) where {T,M} = MemoryLayout(M)

factorize(A::CollocationMatrix) = lu(A)
bandeddata(A::CollocationMatrix) = parent(A)

@inline function bandwidths(A::CollocationMatrix)
    # Assume lower = upper bands
    nbands = size(parent(A), 1)  # total number of bands
    h = (nbands - 1) ÷ 2
    h, h
end

Base.parent(A::CollocationMatrix) = A.data
Base.size(A::CollocationMatrix) = (n = size(parent(A), 2); (n, n))  # matrix is square
Base.copy(A::CollocationMatrix) = CollocationMatrix(copy(parent(A)))
Base.similar(A::CollocationMatrix) = CollocationMatrix(similar(parent(A)))

function Base.fill!(A::CollocationMatrix, v)
    iszero(v) || throw(ArgumentError("CollocationMatrix can only be `fill`ed with zeros"))
    fill!(A.data, v)
    A
end

LinearAlgebra.mul!(y::AbstractVector, A::CollocationMatrix, x::AbstractVector) =
    mul!(y, convert(BandedMatrix, A), x)

Base.:*(A::CollocationMatrix{T}, x::Vector{T}) where {T} = convert(BandedMatrix, A) * x

# Adapted from BandedMatrices
function Base.array_summary(
        io::IO, A::CollocationMatrix, inds::Tuple{Vararg{Base.OneTo}},
    )
    T = eltype(A)
    print(io, Base.dims2string(length.(inds)), " CollocationMatrix{$T} with bandwidths $(bandwidths(A))")
end

# Adapted from BandedMatrices
@inline function Base.getindex(A::CollocationMatrix, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    l, u = bandwidths(A)
    @inbounds BandedMatrices.banded_getindex(A.data, l, u, i, j)
end

# Adapted from BandedMatrices
@inline function Base.setindex!(A::CollocationMatrix, v, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    l, u = bandwidths(A)
    @inbounds BandedMatrices.banded_setindex!(A.data, l, u, v, i, j)
    v
end

const CollocationLU{T} = LU{T, <:CollocationMatrix{T}} where {T}

"""
    LinearAlgebra.lu!(C::CollocationMatrix, pivot = Val(false); check = true)

Perform in-place LU factorisation of collocation matrix without pivoting.

Takes advantage of the totally positive property of collocation matrices
appearing in spline calculations (de Boor 1978).

The code is ported from Carl de Boor's BANFAC routine in FORTRAN77, via its
[FORTRAN90 version by John Burkardt](https://people.math.sc.edu/Burkardt/f_src/pppack/pppack.html).
"""
function lu!(C::CollocationMatrix, ::NoPivot = NoPivot(); check = true)
    check || throw(ArgumentError("`check = false` not yet supported"))
    if size(C, 1) != size(C, 2)
        throw(DimensionMismatch(
            "factorisation of non-square collocation matrices not supported"
        ))
    end
    w = bandeddata(C)
    nbandl, nbandu = bandwidths(C)
    nrow = size(C, 1)
    nroww = size(w, 1)
    @assert nrow == size(w, 2)
    @assert nroww == nbandl + nbandu + 1
    isempty(C) && error("matrix is empty")
    middle = nbandu + 1  # w[middle, :] contains the main diagonal of A
    Cfact = LU(C, Int[], zero(BlasInt)) :: CollocationLU  # factors, ipiv, info

    if nrow == 1
        @inbounds iszero(w[middle, nrow]) && throw(ZeroPivotException(1))
        return Cfact
    end

    if nbandl == 0
        # A is upper triangular. Check that the diagonal is nonzero.
        for i = 1:nrow
            @inbounds iszero(w[middle, i]) && throw(ZeroPivotException(i))
        end
        return Cfact
    end

    if nbandu == 0
        # A is lower triangular. Check that the diagonal is nonzero and
        # divide each column by its diagonal.
        @inbounds for i = 1:(nrow - 1)
            pivot = w[middle, i]
            iszero(pivot) && throw(ZeroPivotException(i))
            ipiv = inv(pivot)
            for j = 1:min(nbandl, nrow - i)
                w[middle + j, i] *= ipiv
            end
        end
        return Cfact
    end

    # A is not just a triangular matrix.
    # Construct the LU factorization.
    @inbounds for i = 1:(nrow - 1)
        pivot = w[middle, i]  # pivot for the i-th step
        iszero(pivot) && throw(ZeroPivotException(i))
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
    @inbounds iszero(w[middle, nrow]) && throw(ZeroPivotException(nrow))

    Cfact
end

"""
    LinearAlgebra.lu(C::CollocationMatrix, pivot = NoPivot(); check = true)

Returns LU factorisation of collocation matrix.

See also [`lu!`](@ref).
"""
lu(C::CollocationMatrix, args...; kws...) = lu!(copy(C), args...; kws...)

ldiv!(F::CollocationLU, y::AbstractVector) = ldiv!(y, F, y)

# Solution of banded linear system A * x = y.
# The code is adapted from Carl de Boor's BANSLV routine in FORTRAN77, via its
# FORTRAN90 version by John Burkardt.
# Note that `x` and `y` may be the same vector.
function ldiv!(
        x::AbstractVector,
        F::CollocationLU,
        y::AbstractVector,
    )
    A = F.factors :: CollocationMatrix
    w = bandeddata(A)
    nrow = size(A, 1)
    nbandl, nbandu = bandwidths(A)
    middle = nbandu + 1
    @assert size(A, 1) == size(A, 2)

    if !(length(x) == length(y) == nrow)
        throw(DimensionMismatch("vectors `x` and `y` must have length $nrow"))
    end

    if x !== y
        copy!(x, y)
    end

    if nrow == 1
        @inbounds x[1] /= w[middle, 1]
        return x
    end

    # Forward pass:
    #
    # For i = 1:(nrow-1), subtract RHS[i] * (i-th column of L) from the
    # right hand side, below the i-th row.
    if nbandl != 0
        for i = 1:(nrow - 1)
            jmax = min(nbandl, nrow - i)
            for j = 1:jmax
                @inbounds x[i + j] -= x[i] * w[middle + j, i]
            end
        end
    end

    # Backward pass:
    #
    # For i = nrow:-1:1, divide RHS[i] by the i-th diagonal entry of
    # U, then subtract RHS[i]*(i-th column of U) from right hand side, above the
    # i-th row.
    @inbounds for i = nrow:-1:2
        x[i] /= w[middle, i]
        for j = 1:min(nbandu, i - 1)
            x[i - j] -= x[i] * w[middle - j, i]
        end
    end

    @inbounds x[1] /= w[middle, 1]

    x
end

function Base.getproperty(F::CollocationLU, d::Symbol)
    A = getfield(F, :factors) :: CollocationMatrix
    data = bandeddata(A)
    l, u = bandwidths(A)
    middle = u + 1
    T = eltype(A)
    if d === :L
        L = BandedMatrix{T}(undef, size(A), (l, 0))
        for j ∈ 1:l
            Ldata = bandeddata(L)
            @views Ldata[1 + j, :] .= data[middle + j, :]
        end
        L[Band(0)] .= 1
        LowerTriangular(L)
    elseif d === :U
        U = BandedMatrix{T}(undef, size(A), (0, u))
        Udata = bandeddata(U)
        for j ∈ 1:middle
            @views Udata[j, :] .= data[j, :]
        end
        UpperTriangular(U)
    elseif d === :p
        Base.OneTo(size(F, 1))    # no row permutations (no pivoting)
    elseif d === :P
        LinearAlgebra.I  # no row permutations
    else
        getfield(F, d)
    end
end
