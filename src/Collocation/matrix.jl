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

Wraps a `BandedMatrix`, providing an efficient LU factorisation without pivoting
adapted from de Boor (1978). The factorisation takes advantage of the total
positivity of spline collocation matrices (de Boor 2001, p. 175).

# Factorisation

`CollocationMatrix` supports in-place LU factorisation using [`lu!`](@ref), as
well as out-of-place factorisation using [`lu`](@ref). LU decomposition may also
be performed via `factorize`.

"""
struct CollocationMatrix{
        T,
        M <: BandedMatrix{T},
    } <: AbstractBandedMatrix{T}
    data :: M
    CollocationMatrix(x::BandedMatrix{T}) where {T} = new{T, typeof(x)}(x)
end

# This affects printing e.g. in the REPL.
MemoryLayout(::Type{<:CollocationMatrix{T,M}}) where {T,M} = MemoryLayout(M)

factorize(A::CollocationMatrix) = lu(A)
bandwidths(A::CollocationMatrix) = bandwidths(parent(A))
bandeddata(A::CollocationMatrix) = bandeddata(parent(A))

Base.parent(A::CollocationMatrix) = A.data
Base.size(A::CollocationMatrix) = size(parent(A))
Base.copy(A::CollocationMatrix) = CollocationMatrix(copy(parent(A)))

# Adapted from BandedMatrices
function Base.array_summary(
        io::IO, A::CollocationMatrix, inds::Tuple{Vararg{Base.OneTo}},
    )
    T = eltype(A)
    print(io, Base.dims2string(length.(inds)), " CollocationMatrix{$T} with bandwidths $(bandwidths(A))")
end

@inline @propagate_inbounds Base.getindex(A::CollocationMatrix, i...) =
    parent(A)[i...]

@inline @propagate_inbounds Base.setindex!(A::CollocationMatrix, v, i...) =
    parent(A)[i...] = v

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
    A = parent(F.factors) :: BandedMatrix
    w = BandedMatrices.bandeddata(A)
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
    factors = getfield(F, :factors) :: CollocationMatrix
    A = parent(factors) :: BandedMatrix
    data = bandeddata(A)
    l, u = bandwidths(A)
    middle = u + 1
    if d === :L
        L = similar(A, size(A)..., l, 0)
        bandeddata(L) .= @view data[middle:(middle + l), :]
        L[Band(0)] .= 1
        LowerTriangular(L)
    elseif d === :U
        U = similar(A, size(A)..., 0, u)
        bandeddata(U) .= @view data[1:middle, :]
        UpperTriangular(U)
    elseif d === :p
        Base.OneTo(size(F, 1))    # no row permutations (no pivoting)
    elseif d === :P
        LinearAlgebra.I  # no row permutations
    else
        getfield(F, d)
    end
end
