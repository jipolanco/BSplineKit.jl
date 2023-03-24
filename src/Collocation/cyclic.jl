# Optimisations for cyclic banded collocation matrices, appearing when using
# periodic B-spline bases.
# For now, optimisations are only applied to cubic periodic splines (k = 4), in
# which case the collocation matrix is tridiagonal.

using LinearAlgebra

"""
    CyclicTridiagonalMatrix{T} <: AbstractMatrix{T}

Represents an almost tridiagonal matrix with non-zero values at the lower-left
and upper-right corners.

This kind of matrix appears when working with periodic splines.

Linear systems involving this matrix [can be efficiently
solved](https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm#Variants)
using a combination of the Thomas algorithm (for regular tridiagonal matrices) and of
the [Sherman–Morrison formula](https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula).
"""
struct CyclicTridiagonalMatrix{T, Vec <: AbstractVector{T}} <: AbstractMatrix{T}
    n :: Int  # size (n × n)
    a :: Vec  # subdiagonal
    b :: Vec  # diagonal
    c :: Vec  # superdiagonal

    function CyclicTridiagonalMatrix(a::AbstractVector, b::AbstractVector, c::AbstractVector)
        n = length(b)
        n == length(a) == length(c) || throw(DimensionMismatch("all diagonals must have the same length"))
        Base.require_one_based_indexing.((a, b, c))
        new{eltype(a), typeof(a)}(n, a, b, c)
    end
end

diagonals(A::CyclicTridiagonalMatrix) = (A.a, A.b, A.c)

function CyclicTridiagonalMatrix{T}(init, n::Int, m::Int) where {T}
    n == m || throw(DimensionMismatch("the matrix must be square"))
    a = Vector{T}(init, n)
    b = Vector{T}(init, n)
    c = Vector{T}(init, n)
    CyclicTridiagonalMatrix(a, b, c)
end

Base.size(A::CyclicTridiagonalMatrix) = (A.n, A.n)

# Used by copy(::CyclicTridiagonalMatrix)
function Base.similar(A::CyclicTridiagonalMatrix, ::Type{S}, dims::Dims{2}) where {S}
    n′ = dims[1]
    n′ == dims[2] || throw(DimensionMismatch("the matrix must be square"))
    diags = map(v -> similar(v, S, n′), diagonals(A))
    CyclicTridiagonalMatrix(diags...)
end

# Used by copy(::CyclicTridiagonalMatrix)
function Base.copyto!(B::CyclicTridiagonalMatrix, A::CyclicTridiagonalMatrix)
    A.n == B.n || throw(DimensionMismatch("matrices must have the same size"))
    foreach(copyto!, diagonals(B), diagonals(A))
    B
end

function Base.fill!(A::CyclicTridiagonalMatrix, v)
    iszero(v) || throw(ArgumentError("can only fill with zeros"))
    foreach(x -> fill!(x, v), diagonals(A))
    A
end

# This is basically only used for printing and tests, doesn't need to be very fast.
@inline function Base.getindex(A::CyclicTridiagonalMatrix, i::Int, j::Int)
    (; n, a, b, c,) = A
    @boundscheck checkbounds(A, i, j)
    if i == j
        return b[i]
    end
    i₋ = i == 1 ? n : i - 1
    if j == i₋
        return a[i]
    end
    i₊ = i == n ? 1 : i + 1
    if j == i₊
        return c[i]
    end
    zero(eltype(A))
end

@inline function Base.setindex!(A::CyclicTridiagonalMatrix, v, i::Int, j::Int)
    (; n, a, b, c,) = A
    @boundscheck checkbounds(A, i, j)
    if i == j
        return b[i] = v
    end
    i₋ = i == 1 ? n : i - 1
    if j == i₋
        return a[i] = v
    end
    i₊ = i == n ? 1 : i + 1
    if j == i₊
        return c[i] = v
    end
    throw(DomainError((i, j), "outside of the matrix bands"))
end

"""
    solve_tridiagonal!(x::AbstractVector, A::CyclicTridiagonalMatrix, y::AbstractVector)

Solve cyclic tridiagonal linear system `Ax = y`.

Note that **all three arrays are modified** by this function, with the result being stored in `x`.

Uses the algorithm described in
[Wikipedia](https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm#Variants)
based on the Sherman–Morrison formula.
"""
function solve_tridiagonal!(x::AbstractVector, A::CyclicTridiagonalMatrix, d::AbstractVector)
    (; n, a, b, c,) = A
    @assert n == length(a) == length(b) == length(c)  # assume lengths haven't been modified
    n == length(x) == length(d) || throw(DimensionMismatch(lazy"all vectors must have length $n"))
    Base.require_one_based_indexing.((x, a, b, c, d))

    a₁ = a[1]
    cₙ = c[n]
    ac = a₁ * cₙ

    # The value of γ is kind of arbitrary. We choose the value leading to the
    # same perturbation of the first and last diagonal entries, i.e. b[1] and b[n].
    γ = sqrt(ac)  # value required to perturb b[1] and b[n] with the same offset

    u = x  # alias to avoid allocations
    u[1] = γ
    u[n] = cₙ

    v1 = 1
    vn = a₁ / γ

    # Modify diagonal for non-cyclic tridiagonal matrix.
    b[1] -= γ
    b[n] -= ac / γ

    # This is logically true for the modified matrix, but not needed in the following.
    # a[1] = 0
    # c[n] = 0

    # Create some aliases for convenience and to avoid allocations.
    # Note that both `d` and `u` are modified by the solve.
    y = d
    q = u
    solve_thomas!((y, q), (a, b, c), (d, u))  # combine both solves for performance

    vy = v1 * y[1] + vn * y[n]
    vq = v1 * q[1] + vn * q[n]

    α = vy / (1 + vq)
    for i ∈ eachindex(x)
        @inbounds x[i] = y[i] - α * q[i]
    end

    x
end

# Simultaneously solve M (non-cyclic) tridiagonal linear systems using Thomas algorithm.
# Note that xs[i] and ds[i] can be aliased.
@fastmath function solve_thomas!(xs::NTuple{M}, (a, b, c), ds::NTuple{M}) where {M}
    Base.require_one_based_indexing.((a, b, c))
    foreach(Base.require_one_based_indexing, xs)
    foreach(Base.require_one_based_indexing, ds)
    n = length(a)
    @assert all(x -> length(x) == n, xs)
    @assert all(x -> length(x) == n, ds)
    @assert n == length(b)
    @assert n == length(c)
    @inbounds for i ∈ 2:n
        w = a[i] / b[i - 1]
        b[i] = b[i] - w * c[i - 1]
        foreach(ds) do d
            @inbounds d[i] = d[i] - w * d[i - 1]
        end
    end
    foreach(xs, ds) do x, d
        @inbounds x[n] = d[n] / b[n]
    end
    for i ∈ (n - 1):-1:1
        foreach(xs, ds) do x, d
            @inbounds x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
        end
    end
    xs
end

struct CyclicLUWrapper{T, Mat <: CyclicTridiagonalMatrix{T}} <: Factorization{T}
    A :: Mat
end

Base.size(F::CyclicLUWrapper) = size(F.A)

# TODO can we precompute some information? note that, for now, the
# "factorisation" can only be used once, after which the data is modified...
LinearAlgebra.lu!(A::CyclicTridiagonalMatrix) = lu(A)  # this is just to trick the interpolation functions
LinearAlgebra.lu(A::CyclicTridiagonalMatrix) = CyclicLUWrapper(A)
LinearAlgebra.ldiv!(x::AbstractVector, A::CyclicTridiagonalMatrix, y::AbstractVector) = solve_tridiagonal!(x, A, y)
LinearAlgebra.ldiv!(x::AbstractVector, F::CyclicLUWrapper, y::AbstractVector) = ldiv!(x, F.A, y)
