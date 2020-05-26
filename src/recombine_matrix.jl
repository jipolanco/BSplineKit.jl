"""
    RecombineMatrix{T,orders} <: AbstractMatrix{T}

Matrix for transformation from B-spline basis to recombined basis.

The transform matrix ``M`` is defined by
```math
ϕ_j = M_{ij} b_i,
```
where ``b_j`` and ``ϕ_i`` are elements of the B-spline and recombined bases,
respectively.

This matrix allows to pass from known coefficients ``u_j`` in the recombined
basis ``ϕ_j``, to the respective coefficients ``v_i`` in the B-spline basis
``b_i``:
```math
\\mathbf{v} = M \\mathbf{u}.
```

Note that the matrix is not square: it has dimensions `(N, M)`, where `N`
is the length of the B-spline basis, and `M < N` is that of the recombined
basis.

As in [`RecombinedBSplineBasis`](@ref), the type parameter `orders` indicates
the order of the boundary condition(s) satisfied by the recombined basis.

Due to the local support of B-splines, the matrix is very sparse, being roughly
described by a diagonal of ones, plus some extra elements in the upper left and
lower right corners (more precisely, these "corners" are blocks of size
`(n + 1, n)`).

The matrix is stored in a memory-efficient way that also allows fast access to
its elements. For orders `n ∈ {0, 1}`, the matrix is made of zeroes and ones,
and the default element type is `Bool`.
"""
struct RecombineMatrix{T, orders, n, n1,
                       Corner <: SMatrix{n1,n,T}} <: AbstractMatrix{T}
    M :: Int      # length of recombined basis (= N - 2 * length(orders))
    N :: Int      # length of B-spline basis
    ul :: Corner  # upper-left corner of matrix, size (n + 1, n)
    lr :: Corner  # lower-right corner of matrix, size (n + 1, n)
    function RecombineMatrix(N::Integer, ul::SMatrix, lr::SMatrix)
        n1, n = size(ul)
        if n1 != n + 1
            throw(ArgumentError("matrices must have dimensions (n + 1, n)"))
        end
        T = eltype(ul)
        Corner = typeof(ul)
        orders = (n, )
        new{T, orders, n, n1, Corner}(N - 2, N, ul, lr)
    end
end

# Specialisation for Dirichlet BCs.
# In this case (and the Neumann case below), the default element type is Bool,
# since the matrix is made of zeroes and ones.
# This is not the case for orders n ≥ 2.
function RecombineMatrix(::Derivative{0}, B::BSplineBasis,
                         ::Type{T} = Bool) where {T}
    N = length(B)
    ul = SMatrix{1,0,T}()
    lr = copy(ul)
    RecombineMatrix(N, ul, lr)
end

# Specialisation for Neumann BCs.
function RecombineMatrix(::Derivative{1}, B::BSplineBasis,
                         ::Type{T} = Bool) where {T}
    N = length(B)
    ul = SMatrix{2,1,T}([1, 1])
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
    Ca = zeros(T, n + 1, n)
    Cb = copy(Ca)

    for i = 1:n
        # Evaluate n-th derivatives of bⱼ at the boundary.
        # TODO replace with analytical formula?
        x = a
        bi, bn1 = evaluate_bspline.(B, (i, n + 1), x, Derivative(n))
        α = bi / bn1
        Ca[i, i] = 1
        Ca[n + 1, i] = -α
    end

    N = length(B)
    M = N - 2

    for (m, i) in enumerate((N - n + 1):N)
        x = b
        bi, bn1 = evaluate_bspline.(B, (i, N - n), x, Derivative(n))
        α = bi / bn1
        Cb[1, m] = -α
        Cb[m + 1, m] = 1
    end

    ul = SMatrix{n + 1, n}(Ca)
    lr = SMatrix{n + 1, n}(Cb)

    RecombineMatrix(N, ul, lr)
end

Base.size(A::RecombineMatrix) = (A.N, A.M)
order_bc(A::RecombineMatrix{T,n}) where {T,n} = n

# j: index in recombined basis
# M: length of recombined basis
function which_recombine_block(::Derivative{n}, j, M) where {n}
    j <= n && return 1
    j > M - n && return 3
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
    n, = order_bc(A) :: Tuple{Int}
    M = size(A, 2)

    block = which_recombine_block(Derivative(n), j, M)

    if block == 2
        return T(i == j + 1)  # δ_{i, j+1}
    end

    if block == 1
        C = A.ul
        if i <= size(C, 1)
            return @inbounds C[i, j] :: T
        end
        return zero(T)
    end

    # The lower-right corner starts at column h + 1.
    h = M - n

    @assert j > h
    C = A.lr
    ii = i - h - 1
    if ii ∈ axes(C, 1)
        return @inbounds C[ii, j - h] :: T
    end

    zero(T)
end

# Efficient implementation of matrix-vector multiplication.
# Generally much faster than using a regular sparse array.
function LinearAlgebra.mul!(y::AbstractVector, A::RecombineMatrix,
                            x::AbstractVector)
    checkdims_mul(y, A, x)
    N, M = size(A)

    n, = order_bc(A) :: Tuple{Int}
    n1 = n + 1
    h = M - n

    @inbounds y[1:n1] = A.ul * @view x[1:n]

    for i = (n + 1):h
        @inbounds y[i + 1] = x[i]
    end

    @inbounds y[(N - n):N] = A.lr * @view x[(h + 1):(h + n)]

    y
end

# Five-argument mul!.
# Note that, since I have this, I could remove the 3-argument mul!  (which is
# equal to the 5-argument one with α = 1 and β = 0), but it would be a bit more
# inefficient to compute.
function LinearAlgebra.mul!(y::AbstractVector, A::RecombineMatrix,
                            x::AbstractVector, α::Number, β::Number)
    checkdims_mul(y, A, x)
    N, M = size(A)

    n, = order_bc(A) :: Tuple{Int}
    n1 = n + 1
    h = M - n

    @inbounds y[1:n1] = α * A.ul * view(x, 1:n) + β * SVector{n1}(view(y, 1:n1))

    for i = (n + 1):h
        @inbounds y[i + 1] = α * x[i] + β * y[i + 1]
    end

    js = (N - n):N
    @inbounds y[js] =
        α * A.lr * view(x, (h + 1):(h + n)) + β * SVector{n1}(view(y, js))

    y
end

const TransposedRecombineMatrix =
    Union{Adjoint{T,M}, Transpose{T,M}} where {T, M <: RecombineMatrix{T}}

function LinearAlgebra.mul!(x::AbstractVector,
                            At::TransposedRecombineMatrix,
                            y::AbstractVector,
                           )
    checkdims_mul(x, At, y)
    A = parent(At)
    N, M = size(A)

    # Select transposition function (is there a better way to do this?)
    # In practice, the recombination matrix is always real-valued, so it
    # shouldn't make a difference.
    tr = At isa Adjoint ? adjoint : transpose

    n, = order_bc(A) :: Tuple{Int}
    n1 = n + 1
    h = M - n

    @inbounds x[1:n] = tr(A.ul) * @view y[1:n1]

    for i = (n + 1):h
        @inbounds x[i] = y[i + 1]
    end

    @inbounds x[(h + 1):(h + n)] = tr(A.lr) * @view y[(N - n):N]

    x
end

function LinearAlgebra.mul!(x::AbstractVector,
                            At::TransposedRecombineMatrix,
                            y::AbstractVector, α::Number, β::Number,
                           )
    checkdims_mul(x, At, y)
    A = parent(At)
    N, M = size(A)
    tr = At isa Adjoint ? adjoint : transpose

    n, = order_bc(A) :: Tuple{Int}
    n1 = n + 1
    h = M - n

    @inbounds x[1:n] =
        α * tr(A.ul) * view(y, 1:n1) + β * SVector{n}(view(x, 1:n))

    for i = (n + 1):h
        @inbounds x[i] = α * y[i + 1] + β * x[i]
    end

    r = (h + 1):(h + n)
    @inbounds x[r] =
        α * tr(A.lr) * view(y, (N - n):N) + β * SVector{n}(view(x, r))

    x
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
