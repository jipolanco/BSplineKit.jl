"""
    RecombineMatrix{T,n} <: AbstractMatrix{T}

Matrix for transformation from B-spline basis to recombined basis.

The transform matrix ``M`` is defined by
```math
ϕ_i = M_{ij} b_j,
```
where ``b_j`` and ``ϕ_i`` are elements of the B-spline and recombined bases,
respectively.

Note that the matrix is not square: it has dimensions `(N - 2, N)`, where `N`
is the length of the B-spline basis.

As in [`RecombinedBSplineBasis`](@ref), the type parameter `n` indicates the
order of the boundary condition satisfied by the recombined basis.

Due to the local support of B-splines, the matrix is very sparse, being roughly
described by a diagonal of ones, plus some extra elements in the upper left and
lower right corners (more precisely, these "corners" are blocks of size
`(n, n + 1)`).

The matrix is stored in a memory-efficient way that also allows fast access to
its elements. For orders `n ∈ {0, 1}`, the matrix is made of zeroes and ones,
and the default element type is `Bool`.
"""
struct RecombineMatrix{T, n, n1,
                       Corner <: SMatrix{n,n1,T}} <: AbstractMatrix{T}
    M :: Int      # length of recombined basis (= N - 2)
    N :: Int      # length of B-spline basis
    ul :: Corner  # upper-left corner of matrix, size (n, n + 1)
    lr :: Corner  # lower-right corner of matrix, size (n, n + 1)
    function RecombineMatrix(N::Integer, ul::SMatrix, lr::SMatrix)
        n, n1 = size(ul)
        if n1 != n + 1
            throw(ArgumentError("matrices must have dimensions (n, n + 1)"))
        end
        T = eltype(ul)
        Corner = typeof(ul)
        new{T, n, n1, Corner}(N - 2, N, ul, lr)
    end
end

# Specialisation for Dirichlet BCs.
# In this case (and the Neumann case below), the default element type is Bool,
# since the matrix is made of zeroes and ones.
# This is not the case for orders n ≥ 2.
function RecombineMatrix(::Derivative{0}, B::BSplineBasis,
                         ::Type{T} = Bool) where {T}
    N = length(B)
    ul = SMatrix{0,1,T}()
    lr = copy(ul)
    RecombineMatrix(N, ul, lr)
end

# Specialisation for Neumann BCs.
function RecombineMatrix(::Derivative{1}, B::BSplineBasis,
                         ::Type{T} = Bool) where {T}
    N = length(B)
    ul = SMatrix{1,2,T}([1, 1])
    lr = copy(ul)
    RecombineMatrix(N, ul, lr)
end

# Generalisation to higher orders.
function RecombineMatrix(::Derivative{n}, B::BSplineBasis,
                         ::Type{T} = Float64) where {n,T}
    @assert n >= 0

    # Example for case n = 2.
    #
    # At each border, we want 2 independent linear combinations of
    # the first 3 B-splines such that ϕ₁'' = ϕ₂'' = 0 at x = a, and such that
    # lower-order derivatives keep at least one degree of freedom (i.e. they do
    # *not* vanish). A simple solution is to choose:
    #
    #     ϕ[1] = b[1] - α[1] b[3],
    #     ϕ[2] = b[2] - α[2] b[3],
    #
    # with α[i] = b[i]'' / b[3]''.
    #
    # This generalises to all orders `n ≥ 0`.

    a, b = boundaries(B)
    Ca = zeros(T, n, n + 1)
    Cb = copy(Ca)

    for i = 1:n
        # Evaluate n-th derivatives of bⱼ at the boundary.
        # TODO replace with analytical formula?
        x = a
        bi, bn1 = evaluate_bspline.(B, (i, n + 1), x, Derivative(n))
        α = bi / bn1
        Ca[i, i] = 1
        Ca[i, n + 1] = -α
    end

    N = length(B)
    M = N - 2

    for (m, i) in enumerate((N - n + 1):N)
        x = b
        bi, bn1 = evaluate_bspline.(B, (i, N - n), x, Derivative(n))
        α = bi / bn1
        Cb[m, 1] = -α
        Cb[m, m + 1] = 1
    end

    ul = SMatrix{n, n + 1}(Ca)
    lr = SMatrix{n, n + 1}(Cb)

    RecombineMatrix(N, ul, lr)
end

Base.size(A::RecombineMatrix) = (A.M, A.N)
order_bc(A::RecombineMatrix{T,n}) where {T,n} = n

# i: index in recombined basis
# M: length of recombined basis
function which_recombine_block(::Derivative{n}, i, M) where {n}
    i <= n && return 1
    i > M - n && return 3
    2
end

# Pretty-printing, adapted from BandedMatrices.jl code.
function Base.replace_in_print_matrix(
        A::RecombineMatrix, i::Integer, j::Integer, s::AbstractString)
    iszero(A[i, j]) ? Base.replace_with_centered_mark(s) : s
end

@inline function Base.getindex(A::RecombineMatrix, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    T = eltype(A)
    n = order_bc(A)
    M = size(A, 1)

    block = which_recombine_block(Derivative(n), i, M)

    if block == 2
        return T(i + 1 == j)  # δ_{i+1, j}
    end

    if block == 1
        C = A.ul
        if j <= size(C, 2)
            return @inbounds C[i, j] :: T
        end
        return zero(T)
    end

    # The lower-right corner starts at row h + 1.
    h = M - n

    @assert i > h
    C = A.lr
    jj = j - h - 1
    if jj ∈ axes(C, 2)
        return @inbounds C[i - h, jj] :: T
    end

    zero(T)
end

# Efficient implementation of matrix-vector multiplication.
# Generally much faster than using a regular sparse array.
function LinearAlgebra.mul!(y::AbstractVector, A::RecombineMatrix,
                            x::AbstractVector)
    checkdims_mul(y, A, x)
    M, N = size(A)

    n = order_bc(A)
    n1 = n + 1
    h = M - n

    @inbounds y[1:n] = A.ul * @view x[1:n1]

    for i = (n + 1):h
        @inbounds y[i] = x[i + 1]
    end

    @inbounds y[(h + 1):(h + n)] = A.lr * @view x[(N - n):N]

    y
end

# Five-argument mul!.
# Note that, since I have this, I could remove the 3-argument mul!  (which is
# equal to the 5-argument one with α = 1 and β = 0), but it would be a bit more
# inefficient to compute.
function LinearAlgebra.mul!(y::AbstractVector, A::RecombineMatrix,
                            x::AbstractVector, α::Number, β::Number)
    checkdims_mul(y, A, x)
    M, N = size(A)

    n = order_bc(A)
    n1 = n + 1
    h = M - n

    @inbounds y[1:n] = α * A.ul * view(x, 1:n1) + β * SVector{n}(view(y, 1:n))

    for i = (n + 1):h
        @inbounds y[i] = α * x[i + 1] + β * y[i]
    end

    r = (h + 1):(h + n)
    @inbounds y[r] = α * A.lr * view(x, (N - n):N) + β * SVector{n}(view(y, r))

    y
end

@inline function checkdims_mul(y, A, x)
    M, N = size(A)
    if length(y) != M
        throw(DimensionMismatch("first dimension of A, $M, " *
                                "does not match length of y, $(length(y))"))
    end
    if length(x) != N
        throw(DimensionMismatch("second dimension of A, $N, " *
                                "does not match length of x, $(length(x))"))
    end
    nothing
end
