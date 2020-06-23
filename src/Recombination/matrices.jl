"""
    RecombineMatrix{T, DiffOps} <: AbstractMatrix{T}

Matrix for transformation from B-spline basis to recombined basis.

The transform matrix ``M`` is defined by
```math
ϕ_j = M_{ij} b_i,
```
where ``b_j(x)`` and ``ϕ_i(x)`` are elements of the B-spline and recombined
bases, respectively.

This matrix allows to pass from known coefficients ``u_j`` in the recombined
basis ``ϕ_j``, to the respective coefficients ``v_i`` in the B-spline basis
``b_i``:
```math
\\bm{v} = \\mathbf{M} \\bm{u}.
```

Note that the matrix is not square: it has dimensions ``N × M``, where ``N``
is the length of the B-spline basis, and ``M < N`` is that of the recombined
basis.

As in [`RecombinedBSplineBasis`](@ref), the type parameter `DiffOps` indicates
the homogeneous boundary condition(s) satisfied by the recombined basis.

Due to the local support of B-splines, basis recombination can be performed
among neigbouring B-splines near the boundaries (see
[`RecombinedBSplineBasis`](@ref)).
This leads to a recombination matrix which is almost a diagonal of ones, plus
a few extra super- and sub-diagonal elements in the upper left and lower right
corners, respectively.
The matrix is stored in a memory-efficient way that also allows fast access to
its elements.

For boundary condition orders `n ∈ {0, 1}`, the matrix is made of zeroes and
ones, and the default element type is `Bool`.
"""
struct RecombineMatrix{T,
                       DiffOps <: Tuple{Vararg{AbstractDifferentialOp}},
                       n, n1,   # nc = n1 - n
                       Corner <: SMatrix{n1,n,T}} <: AbstractMatrix{T}
    ops :: DiffOps  # list of differential operators for BCs
    M :: Int      # length of recombined basis (= N - 2nc)
    N :: Int      # length of B-spline basis
    ul :: Corner  # upper-left corner of matrix, size (n1, n)
    lr :: Corner  # lower-right corner of matrix, size (n1, n)
    function RecombineMatrix(ops::Tuple{Vararg{AbstractDifferentialOp}},
                             N::Integer, ul::SMatrix, lr::SMatrix)
        n1, n = size(ul)
        nc = n1 - n  # number of BCs per boundary
        if nc <= 0
            throw(ArgumentError("matrices must have dimensions (m, n) with m > n"))
        end
        M = N - 2nc
        T = eltype(ul)
        Corner = typeof(ul)
        Ops = typeof(ops)
        new{T, Ops, n, n1, Corner}(ops, M, N, ul, lr)
    end
end

# Default element type of recombination matrix.
# In some specific cases we can use Bool...
_default_eltype(::Vararg{AbstractDifferentialOp}) = Float64
_default_eltype(::Derivative{0}) = Bool  # Dirichlet BCs
_default_eltype(::Derivative{1}) = Bool  # Neumann BCs

# Case (D(0), D(1), D(2), ...)
_default_eltype(::Derivative{0}, ::Derivative{1}, ::Vararg{Derivative}) = Bool

RecombineMatrix(ops::Tuple{Vararg{AbstractDifferentialOp}}, B::BSplineBasis) =
    RecombineMatrix(ops, B, _default_eltype(ops...))

# Specialisation for Dirichlet BCs.
function RecombineMatrix(op::Tuple{Derivative{0}}, B::BSplineBasis,
                         ::Type{T}) where {T}
    _check_bspline_order(op, B)
    N = length(B)
    ul = SMatrix{1,0,T}()
    lr = copy(ul)
    RecombineMatrix(op, N, ul, lr)
end

# Specialisation for Neumann BCs.
function RecombineMatrix(op::Tuple{Derivative{1}}, B::BSplineBasis,
                         ::Type{T}) where {T}
    _check_bspline_order(op, B)
    N = length(B)
    ul = SMatrix{2,1,T}(1, 1)
    lr = copy(ul)
    RecombineMatrix(op, N, ul, lr)
end

# Generalisation to higher orders and to more general differential operators
# (this includes Robin BCs, for instance).
function RecombineMatrix(ops::Tuple{AbstractDifferentialOp}, B::BSplineBasis,
                         ::Type{T}) where {T}
    _check_bspline_order(ops, B)
    op = first(ops)
    n = max_order(op)
    @assert n >= 0

    a, b = boundaries(B)
    Ca = zeros(T, n + 1, n)
    Cb = copy(Ca)

    let x = a
        is = ntuple(identity, Val(n + 1))  # = 1:(n + 1)
        # Evaluate n-th derivatives of bⱼ at the boundary.
        # TODO replace with analytical formula?
        bs = evaluate_bspline.(B, is, x, op)
        for m = 1:n
            b0, b1 = bs[m], bs[m + 1]
            @assert !(b0 ≈ b1)
            r = 2 / (b0 - b1)  # normalisation factor
            Ca[m, m] = -b1 * r
            Ca[m + 1, m] = b0 * r
        end
    end

    N = length(B)
    M = N - 2

    let x = b
        is = ntuple(d -> N - d + 1, Val(n + 1))  # = N:-1:(N - n)
        bs = evaluate_bspline.(B, is, x, op)
        for m = 1:n
            b0, b1 = bs[n - m + 2], bs[n - m + 1]
            @assert !(b0 ≈ b1)
            r = 2 / (b0 - b1)
            Cb[m, m] = -b1 * r
            Cb[m + 1, m] = b0 * r
        end
    end

    ul = SMatrix{n + 1, n}(Ca)
    lr = SMatrix{n + 1, n}(Cb)

    RecombineMatrix(ops, N, ul, lr)
end

# Mixed derivatives.
# Specific case (D(0), D(1), ..., D(Nc - 1)).
# This actually generalises the Dirichlet case above.
function RecombineMatrix(ops::Tuple{Derivative,Derivative,Vararg{Derivative}},
                         B::BSplineBasis, ::Type{T}) where {T}
    orders_in = get_orders(ops...)  # note that this is a compile-time constant...
    orders = _sort_orders(Val(orders_in))
    Nc = length(orders)
    @assert Nc >= 2  # case Nc = 1 is treated by different functions
    _check_bspline_order(ops, B)
    _check_mixed_derivatives(Val(orders))
    N = length(B)
    ul = SMatrix{Nc,0,T}()
    lr = copy(ul)
    RecombineMatrix(ops, N, ul, lr)
end

@generated function _sort_orders(::Val{orders}) where {orders}
    N = length(orders)
    v = typeof(orders)(sort([orders...]))
    :( $v )
end

# Verify that the B-spline order is compatible with the given differential
# operators.
function _check_bspline_order(ops::Tuple, B::BSplineBasis)
    n = max_order(ops...)
    k = order(B)
    if n >= k
        throw(ArgumentError(
            "cannot resolve operators $ops with B-splines of order $k"))
    end
    nothing
end

function _check_mixed_derivatives(::Val{orders}) where {orders}
    length(orders) === 1 && return  # single BC
    if orders !== ntuple(d -> d - 1, Val(length(orders)))  # mixed BCs
        throw(ArgumentError("unsupported case: derivatives = $orders"))
    end
    nothing
end

# Unsupported cases
RecombineMatrix(ops::Tuple{Vararg{AbstractDifferentialOp}}, args...) =
    throw(ArgumentError(
        "boundary condition combination is currently unsupported: $ops"))

Base.size(A::RecombineMatrix) = (A.N, A.M)
constraints(A::RecombineMatrix) = A.ops
num_constraints(A::RecombineMatrix) = length(constraints(A))
num_constraints(::UniformScaling) = 0  # case of non-recombined bases

DifferentialOps.max_order(A::RecombineMatrix) = max_order(constraints(A)...)

# Returns number of basis functions that are recombined from the original basis
# near each boundary.
# For instance, Neumann BCs have a single recombined function, ϕ_1 = b_1 + b_2;
# while mixed Dirichlet + Neumann have none, ϕ_1 = b_3.
num_recombined(A::RecombineMatrix) = max_order(A) + 1 - num_constraints(A)

# j: index in recombined basis
# M: length of recombined basis
@inline function which_recombine_block(A::RecombineMatrix, j)
    @boundscheck checkbounds(A, :, j)
    M = A.M
    n = num_recombined(A)
    j <= n && return 1
    j > M - n && return 3
    2
end

"""
    nzrows(A::RecombineMatrix, col::Integer) -> UnitRange{Int}

Returns the range of row indices `i` such that `A[i, col]` is non-zero.
"""
@propagate_inbounds function nzrows(A::RecombineMatrix,
                                    j::Integer) :: UnitRange{Int}
    block = which_recombine_block(A, j)
    j += num_constraints(A)
    if block == 1
        (j - 1):j
    elseif block == 2
        j:j  # shifted diagonal of ones
    else
        j:(j + 1)
    end
end

# Pretty-printing, adapted from BandedMatrices.jl code.
function Base.replace_in_print_matrix(
        A::RecombineMatrix, i::Integer, j::Integer, s::AbstractString)
    iszero(A[i, j]) ? Base.replace_with_centered_mark(s) : s
end

@inline function Base.getindex(A::RecombineMatrix, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    T = eltype(A)
    @inbounds block = which_recombine_block(A, j)

    if block == 2
        c = num_constraints(A)
        return T(i == j + c)  # δ_{i, j+c}
    end

    if block == 1
        C = A.ul
        if i <= size(C, 1)
            return @inbounds C[i, j] :: T
        end
        return zero(T)
    end

    # The lower-right corner starts at column h + 1.
    M = size(A, 2)
    n = num_recombined(A)
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

    n = num_recombined(A)
    c = num_constraints(A)
    n1 = n + c
    h = M - n

    @inbounds y[1:n1] = A.ul * @view x[1:n]

    for i = (n + 1):h
        @inbounds y[i + c] = x[i]
    end

    @inbounds y[(N - n - c + 1):N] = A.lr * @view x[(h + 1):M]

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

    n = num_recombined(A)
    c = num_constraints(A)
    n1 = n + c
    h = M - n

    @inbounds y[1:n1] = α * A.ul * view(x, 1:n) + β * SVector{n1}(view(y, 1:n1))

    for i = (n + 1):h
        @inbounds y[i + c] = α * x[i] + β * y[i + c]
    end

    js = (N - n - c + 1):N
    @inbounds y[js] =
        α * A.lr * view(x, (h + 1):M) + β * SVector{n1}(view(y, js))

    y
end

function LinearAlgebra.ldiv!(x::AbstractVector, A::RecombineMatrix,
                             y::AbstractVector)
    checkdims_mul(y, A, x)
    N, M = size(A)

    n = num_recombined(A)
    c = num_constraints(A)
    n1 = n + c
    h = M - n

    @inbounds x[1:n] = _ldiv_unique_solution(A.ul, view(y, 1:n1))

    for i = (n + 1):h
        @inbounds x[i] = y[i + c]
    end

    js = (N - n - c + 1):N
    @inbounds x[(h + 1):M] = _ldiv_unique_solution(A.lr, view(y, js))

    x
end

function LinearAlgebra.:\(A::RecombineMatrix, y::AbstractVector)
    x = similar(y, size(A, 2))
    ldiv!(x, A, y)
end

"""
    NoUniqueSolutionError <: Exception

Exception thrown when solving linear system using [`RecombineMatrix`](@ref),
when the system has no unique solution.
"""
struct NoUniqueSolutionError <: Exception end

function Base.showerror(io::IO, ::NoUniqueSolutionError)
    print(io,
          "overdetermined system has no unique solution.",
          " This means that the given function expanded in the parent basis",
          " has no exact representation (i.e. cannot be expanded) in the recombined basis."
         )
end

# Solve A \ y for overdetermined system with rectangular matrix A, assuming that
# the system has exactly one solution. Throws error if that's not the case.
function _ldiv_unique_solution(A::SMatrix, y::AbstractVector)
    N, M = size(A)
    @assert length(y) == N
    @assert N > M "system must be overdetermined"

    # Solve the first M equations.
    @inbounds Am = SMatrix{M,M}(view(A, 1:M, :))
    @inbounds ym = SVector{M}(view(y, 1:M))
    x = Am \ ym

    # Check that equations (M+1):N are satisfied.
    @inbounds for i = (M + 1):N
        res = zero(eltype(x))
        for j in axes(A, 2)
            res += A[i, j] * x[j]
        end
        res ≈ y[i] || throw(NoUniqueSolutionError())
    end

    x
end

function checkdims_mul(y, A, x)
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
