const DiffOpList = Tuple{Vararg{AbstractDifferentialOp}}

"""
    RecombineMatrix{T} <: AbstractMatrix{T}

Matrix for transformation from coefficients of the recombined basis, to the
corresponding B-spline basis coefficients.

# Extended help

The transformation matrix ``M`` is defined by
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
is the length of the B-spline basis, and ``M = N - δ`` is that of the
recombined basis (see [`RecombinedBSplineBasis`](@ref) for details).

Due to the local support of B-splines, basis recombination can be performed
by combining just a small set of B-splines near the boundaries (as discussed in
[`RecombinedBSplineBasis`](@ref)).
This leads to a recombination matrix which is almost a diagonal of ones, plus
a few extra super- and sub-diagonal elements in the upper left and lower right
corners, respectively.
The matrix is stored in a memory-efficient way that also allows fast access to
its elements.

Efficient implementations of matrix-vector products (using the `*` operator or
`LinearAlgebra.mul!`) and of left division of vectors (using `\\` or
`LinearAlgebra.ldiv!`) are included.
These two operations can be used to transform between coefficients in the
original and recombined bases.

Note that, since the recombined basis forms a subspace of the original basis
(which is related to the rectangular shape of the matrix),
it is generally not possible to obtain recombined coefficients from original
coefficients, unless the latter already satisfy the constraints encoded in the
recombined basis.
The left division operation will throw a [`NoUniqueSolutionError`](@ref) if that
is not the case.

---

    RecombineMatrix(ops::Tuple{Vararg{AbstractDifferentialOp}}, B::BSplineBasis, [T])
    RecombineMatrix(ops_left, ops_right, B::BSplineBasis, [T])

Construct recombination matrix describing a B-spline basis recombination.

In the first case, `ops` is the boundary condition (BC) to be applied on both boundaries.
The second case allows to set different BCs on each boundary.

The default element type `T` is generally `Float64`, except for specific
differential operators which yield a matrix of zeroes and ones, for which `Bool`
is the default.

See the [`RecombinedBSplineBasis`](@ref) constructor for details on the
`ops` argument.

"""
struct RecombineMatrix{
        T,
        LeftBC  <: Pair{<:DiffOpList, <:SMatrix},
        RightBC <: Pair{<:DiffOpList, <:SMatrix},
    } <: AbstractMatrix{T}
    left  :: LeftBC
    right :: RightBC
    M :: Int      # length of recombined basis (= N - nc_left - nc_right)
    N :: Int      # length of B-spline basis
    allowed_nonzeros_per_column :: NTuple{2, Int}

    global function _RecombineMatrix(
            left::Pair{<:DiffOpList, <:SMatrix},
            right::Pair{<:DiffOpList, <:SMatrix},
            N::Integer;
            dropped_bsplines::NTuple{2, Integer},
        )
        ops_l, ul = left    # left BC and corresponding upper-left submatrix
        ops_r, lr = right  # right BC and corresponding lower-right submatrix

        _check_BC_submatrix(left)
        _check_BC_submatrix(right)

        nc_l = length(ops_l)  # number of constraints on the left
        nc_r = length(ops_r)
        M = N - (nc_l + nc_r)
        T = promote_type(eltype(ul), eltype(lr))

        # Locality condition: make sure corner arrays are as close to banded as possible.
        nc = (nc_l, nc_r)
        allowed_nonzeros_per_column = 1 .+ nc .- dropped_bsplines

        _check_locality(ul, :upper_left, allowed_nonzeros_per_column[1])
        _check_locality(lr, :lower_right, allowed_nonzeros_per_column[2])

        LeftBC = typeof(left)
        RightBC = typeof(right)

        new{T, LeftBC, RightBC}(left, right, M, N, allowed_nonzeros_per_column)
    end
end

# For compatibility with previous versions (same BC on both sides)
function _RecombineMatrix(
        ops::DiffOpList, N::Integer,
        ul::SMatrix{n1, n}, lr::SMatrix{n1, n};
        kws...,
    ) where {n1, n}
    left = ops => ul
    right = ops => lr
    _RecombineMatrix(left, right, N; kws...)
end

function _check_BC_submatrix(bc::Pair)
    ops, A = bc
    m, n = size(A)
    m ≥ n || throw(ArgumentError("submatrix must have dimensions (m, n) with m ≥ n"))
    length(ops) == m - n || throw(ArgumentError("wrong dimensions of submatrix"))
    nothing
end

function _check_locality(A, corner, allowed_nonzeros_per_column)
    m, n = size(A)
    nc = m - n  # number of constraints
    is = axes(A, 1)
    js = axes(A, 2)

    @inbounds if corner == :upper_left
        dropped_bsplines = nc + 1 - allowed_nonzeros_per_column
        for j ∈ js
            istart = j + dropped_bsplines
            iend = istart + (allowed_nonzeros_per_column - 1)
            iallowed = istart:iend
            for i ∈ is
                @assert i ∈ iallowed || iszero(A[i, j])
            end
        end
    elseif corner == :lower_right
        # allowed_nonzeros_per_column = nc - dropped_bsplines + 1
        for j ∈ js
            iallowed = j:(j + allowed_nonzeros_per_column - 1)
            for i ∈ is
                @assert i ∈ iallowed || iszero(A[i, j])
            end
        end
    end

    nothing
end

# Default element type of recombination matrix.
# In some specific cases we can use Bool...
_default_eltype(::BoundaryCondition) = Float64
_default_eltype(::Derivative{0}) = Bool  # Dirichlet BCs
_default_eltype(::Derivative{1}) = Bool  # Neumann BCs
_default_eltype(::Vararg{Derivative}) = Float64

# Case (D(0), D(1), D(2), ...)
_default_eltype(::Derivative{0}, ::Derivative{1}, ::Vararg{Derivative}) = Bool  # TODO this isn't always right, is it?
_default_eltype(ops::DiffOpList) = _default_eltype(ops...)

# Same BCs on both sides.
RecombineMatrix(ops, B::BSplineBasis, args...) = RecombineMatrix(ops, ops, B, args...)

# No element type specified.
function RecombineMatrix(ops_l, ops_r, B::BSplineBasis)
    Tl = _default_eltype(ops_l)
    Tr = _default_eltype(ops_r)
    T = promote_type(Tl, Tr)
    RecombineMatrix(ops_l, ops_r, B::BSplineBasis, T)
end

function RecombineMatrix(left, right, B::BSplineBasis, ::Type{T}) where {T}
    ops_l = _normalise_ops(left, B)
    ops_r = _normalise_ops(right, B)
    _check_bspline_order(ops_l, B)
    _check_bspline_order(ops_r, B)
    N = length(B)
    ldrop, ul = _make_submatrix(ops_l, (B, Val(:left)), T)
    rdrop, lr = _make_submatrix(ops_r, (B, Val(:right)), T)
    _RecombineMatrix(ops_l => ul, ops_r => lr, N; dropped_bsplines = (ldrop, rdrop))
end

_normalise_ops(op::Derivative, B) = (op,)
_normalise_ops(op::Tuple, B) = op
_normalise_ops(op::BoundaryCondition, B) = op

RecombineMatrix(bc::BoundaryCondition, B::BSplineBasis) = RecombineMatrix(bc, B, _default_eltype(bc))
RecombineMatrix(ops::DiffOpList, B::BSplineBasis) = RecombineMatrix(ops, B, _default_eltype(ops))

RecombineMatrix(op::AbstractDifferentialOp, B::BSplineBasis, args...) =
    RecombineMatrix((op,), B, args...)

RecombineMatrix(r::DerivativeUnitRange, B::BSplineBasis, args...) =
    RecombineMatrix(Tuple(r), B, args...)

# Specialisation for Dirichlet BCs: we simply drop the first/last B-spline.
function _make_submatrix(::Tuple{Derivative{0}}, Bdata::Tuple, ::Type{T}) where {T}
    ndrop = 1
    A = SMatrix{1, 0, T}()
    ndrop, A
end

# Specialisation for Neumann BCs: we combine the first/last 2 B-splines, without dropping any of them.
function _make_submatrix(::Tuple{Derivative{1}}, Bdata::Tuple, ::Type{T}) where {T}
    ndrop = 0
    A = SMatrix{2, 1, T}(1, 1)
    ndrop, A
end

# Generalisation to higher orders and to more general differential operators
# (this includes Robin BCs, for instance).
#
# If more than one operator is passed, they must be of the form
#
#    (D(0), D(1), ..., D(m - 2), D(m - 1), [D(n)]),
#
# where the last D(n) is optional, and satisfies n ≥ m + 1.
# In this case, the first `m` B-splines are dropped, and the next `n - m + 1`
# B-splines are recombined into `q = n - m` functions.
#
# That last operator is allowed to be a linear combination of `Derivative`s.
# In that case, `n` is the maximum degree of the operator.
function _make_submatrix(ops::DiffOpList, Bdata::Tuple, ::Type{T}) where {T}
    B = first(Bdata) :: BSplineBasis
    h = order(B) ÷ 2
    # Identify operators corresponding to natural splines
    if ops === _natural_ops(Val(h))
        return _make_submatrix(Natural(), Bdata, T)
    end
    op = last(ops)
    n = max_order(op)
    ndrop = _bsplines_to_drop(ops...)
    A = _make_submatrix_manyops(Val(n + 1), Val(ndrop), ops, B, T)
    ndrop, A
end

function _make_submatrix_manyops(::Val{ndrop}, ::Val{ndrop}, ops, B, ::Type{T}) where {ndrop, T}
    # Case of mixed BCs (D(0), D(1), ..., D(n - 1)).
    # B-splines are dropped, and new functions are not created.
    # This is a generalisation of the Dirichlet case.
    n = ndrop - 1
    @assert ops === Tuple(Derivative(0:n))
    nc = length(ops)  # number of constraints
    SMatrix{nc, 0, T}()
end

function _make_submatrix_manyops(::Val{n1}, ::Val{ndrop}, ops, Bdata::Tuple, ::Type{T}) where {n1, ndrop, T}
    B, side = Bdata
    @assert B isa BSplineBasis
    @assert side isa Val  # either Val{:left} or Val{:right}

    nc = length(ops)

    # Check that operators look like:
    #
    #   (D{0}, D{1}, …, D{nc - 2}, some other operator of order ≥ nc - 1)
    #
    @assert _droplast(ops...) === Tuple(Derivative(0:(nc - 2)))
    @assert max_order(last(ops)) ≥ nc - 1

    n = n1 - 1
    q = n - ndrop
    @assert q ≥ 1

    A = zero(MMatrix{q + nc, q, T})
    op = last(ops)

    if side === Val(:left)
        x = first(boundaries(B))

        # Indices of B-splines to recombine.
        is = ntuple(d -> ndrop + d, Val(q + 1))

        # Coefficients of odd derivatives must change sign, since d/dn = -d/dx
        # on the left boundary.
        opn = dot(op, LeftNormal())

        # Evaluate n-th derivatives of bⱼ at the boundary.
        # TODO replace with analytical formula?
        bs = evaluate.(B, is, x, opn)
        for l = 1:q
            b0, b1 = bs[l], bs[l + 1]
            @assert !(b0 ≈ b1)
            r = 2 / (b0 - b1)  # normalisation factor
            A[l + nc - 1, l] = -b1 * r
            A[l + nc, l] = b0 * r
        end
    elseif side === Val(:right)
        x = last(boundaries(B))
        N = length(B)
        M = N - ndrop  # index of last undropped B-spline
        is = ntuple(d -> M - d + 1, Val(q + 1))
        opn = dot(op, RightNormal())
        @assert opn === op  # the right normal is in the x direction
        bs = evaluate.(B, is, x, opn)
        for l = 1:q
            b0, b1 = bs[q - l + 2], bs[q - l + 1]
            @assert !(b0 ≈ b1)
            r = 2 / (b0 - b1)
            A[l, l] = -b1 * r
            A[l + 1, l] = b0 * r
        end
    end

    SMatrix(A)
end

_droplast(a) = ()
_droplast(a, etc...) = (a, _droplast(etc...)...)

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

_bsplines_to_drop(ops::Tuple) = _bsplines_to_drop(ops...)

# Single BC: no B-splines are dropped (except if op = Derivative{0})
_bsplines_to_drop(op::AbstractDifferentialOp) = 0

_unval(::Val{n}) where {n} = n
_bsplines_to_drop(ops...) = _bsplines_to_drop(Val(0), ops...) |> _unval

function _bsplines_to_drop(::Val{n}, a, b, etc...) where {n}
    if a !== Derivative(n)
        throw(ArgumentError("unsupported combination of boundary conditions"))
    end
    _bsplines_to_drop(Val(n + 1), b, etc...)
end

# This is the case of the last operator, which is allowed to be something
# different than Derivative{n}.
function _bsplines_to_drop(::Val{n}, op) where {n}
    if op === Derivative(n)
        return Val(n + 1)
    end
    if max_order(op) ≤ n
        throw(ArgumentError("order of last operator is too low"))
    end
    Val(n)
end

Base.size(A::RecombineMatrix) = (A.N, A.M)

constraints(A::RecombineMatrix) = map(first, (A.left, A.right))
constraints(::UniformScaling) = ((), ())
submatrices(A::RecombineMatrix) = map(last, (A.left, A.right))

num_constraints(A::RecombineMatrix) = map(length, constraints(A))
num_constraints(::UniformScaling) = (0, 0)  # case of non-recombined bases

# Returns (left, right) tuple with maximum BC order on each boundary.
_max_order(A::RecombineMatrix) = map(ops -> max_order(ops...), constraints(A))

# Returns number of basis functions that are recombined from the original basis
# near each boundary.
# For instance, Neumann BCs have a single recombined function, ϕ_1 = b_1 + b_2;
# while mixed Dirichlet + Neumann have none, ϕ_1 = b_3.
num_recombined(A::RecombineMatrix) =
    map((m, c) -> m + 1 - c, _max_order(A), num_constraints(A))
num_recombined(::UniformScaling) = (0, 0)

# j: index in recombined basis
# M: length of recombined basis
@inline function which_recombine_block(A::RecombineMatrix, j)
    @boundscheck checkbounds(A, :, j)
    M = A.M
    nl, nr = num_recombined(A)  # left/right number of recombined bases
    j <= nl && return 1
    j > M - nr && return 3
    2
end

"""
    nzrows(A::RecombineMatrix, col::Integer) -> UnitRange{Int}

Returns the range of row indices `i` such that `A[i, col]` is non-zero.
"""
@propagate_inbounds function nzrows(
        A::RecombineMatrix, j::Integer;
        block = nothing,  # to avoid recomputing it if already known
    )
    blk = something(block, which_recombine_block(A, j))
    j += num_constraints(A)[1]
    if blk == 1
        # We take advantage of the locality condition imposed when constructing
        # the recombination matrix.
        δ = A.allowed_nonzeros_per_column[1]
        (j - δ + 1):j
    elseif blk == 2
        j:j  # shifted diagonal of ones
    else
        δ = A.allowed_nonzeros_per_column[2]
        j:(j + δ - 1)
    end :: UnitRange{Int}
end

# Pretty-printing, adapted from BandedMatrices.jl code.
function Base.replace_in_print_matrix(
        A::RecombineMatrix, i::Integer, j::Integer, s::AbstractString)
    iszero(A[i, j]) ? Base.replace_with_centered_mark(s) : s
end

@inline function Base.getindex(
        A::RecombineMatrix, i::Integer, j::Integer;
        block = nothing,
    )
    @boundscheck checkbounds(A, i, j)
    T = eltype(A)
    @inbounds blk = something(block, which_recombine_block(A, j))

    cl, cr = num_constraints(A)  # left/right constraints
    ul, lr = submatrices(A)

    if blk == 2
        return T(i == j + cl)  # δ_{i, j+c}
    end

    if blk == 1
        C_left = ul
        if i <= size(C_left, 1)
            return @inbounds C_left[i, j] :: T
        end
        return zero(T)
    end

    # The lower-right corner starts at column h + 1.
    M = size(A, 2)
    nl, nr = num_recombined(A)
    h = M - nr

    @assert j > h
    C_right = lr
    ii = i - h - cr
    if ii ∈ axes(C_right, 1)
        return @inbounds C_right[ii, j - h] :: T
    end

    zero(T)
end

# Efficient implementation of matrix-vector multiplication.
# Generally much faster than using a regular sparse array.
function LinearAlgebra.mul!(y::AbstractVector, A::RecombineMatrix,
                            x::AbstractVector)
    checkdims_mul(y, A, x)
    N, M = size(A)

    nl, nr = num_recombined(A)
    cl, cr = num_constraints(A)
    ul, lr = submatrices(A)
    n1 = nl + cl
    h = M - nr

    @inbounds y[1:n1] = ul * @view x[1:nl]

    for i = (nl + 1):h
        @inbounds y[i + cl] = x[i]
    end

    @inbounds y[(N - nr - cr + 1):N] = lr * @view x[(h + 1):M]

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

    nl, nr = num_recombined(A)
    cl, cr = num_constraints(A)
    ul, lr = submatrices(A)
    n1 = nl + cl
    h = M - nr

    @inbounds y[1:n1] = α * ul * view(x, 1:nl) + β * SVector{n1}(view(y, 1:n1))

    for i = (nl + 1):h
        @inbounds y[i + cl] = α * x[i] + β * y[i + cl]
    end

    js = (N - nr - cr + 1):N
    @inbounds y[js] =
        α * lr * view(x, (h + 1):M) + β * SVector{n1}(view(y, js))

    y
end

function LinearAlgebra.ldiv!(x::AbstractVector, A::RecombineMatrix,
                             y::AbstractVector)
    checkdims_mul(y, A, x)
    N, M = size(A)

    nl, nr = num_recombined(A)
    cl, cr = num_constraints(A)
    ul, lr = submatrices(A)
    n1 = nl + cl
    h = M - nr

    @inbounds x[1:nl] = _ldiv_unique_solution(ul, view(y, 1:n1))

    for i = (nl + 1):h
        @inbounds x[i] = y[i + cl]
    end

    js = (N - nr - cr + 1):N
    @inbounds x[(h + 1):M] = _ldiv_unique_solution(lr, view(y, js))

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

    # Solve the first non-trivial M equations, skipping rows of zeroes.
    n = 1
    if M != 0
        @inbounds while n ≤ N && all(iszero.(A[n, :]))
            n += 1
        end
    end
    m = n + M - 1
    @assert m <= N
    @inbounds Am = SMatrix{M,M}(view(A, n:m, :))
    @inbounds ym = SVector{M}(view(y, n:m))
    x = Am \ ym

    # Check that equations (m+1):N are satisfied.
    @inbounds for i = (m + 1):N
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
