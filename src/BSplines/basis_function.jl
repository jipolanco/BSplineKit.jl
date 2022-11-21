"""
    BasisFunction{B <: AbstractBSplineBasis, T}

Describes a single basis function.

The basis function may belong to a [`BSplineBasis`](@ref) (in which case it's
effectively a B-spline), or to a basis derived from a B-spline basis (such as a
`RecombinedBSplineBasis`).

---

    BasisFunction(basis::AbstractBSplineBasis, i::Int, [T = Float64])

Construct i-th basis function of the given basis.

The constructed function can be evaluated as `b(x)`, returning a value of type
`T`.

---

    (b::BasisFunction)(x, [op::AbstractDifferentialOp])

Evaluate basis function at coordinate `x`.

To evaluate a derivative, pass `Derivative(n)` as the `op` argument, with `n`
the derivative order.

To evaluate multiple derivatives, pass a derivative range `Derivative(m:n)`.
In particular, `Derivative(m:n)` evaluates the basis function itself and its
first `n` derivatives.

More general differential operators, such as `Derivative(n) + λ Derivative(m)`,
are also supported.
"""
struct BasisFunction{Basis <: AbstractBSplineBasis, T}
    basis :: Basis
    i     :: Int
    @inline function BasisFunction(
            b::AbstractBSplineBasis, i, ::Type{T} = Float64) where {T}
        Basis = typeof(b)
        new{Basis, T}(b, i)
    end
end

basis(b::BasisFunction) = b.basis
knots(b::BasisFunction) = knots(basis(b))
order(b::BasisFunction) = order(basis(b))
Base.eltype(::Type{BasisFunction{B,T}}) where {B,T} = T

function Base.show(io::IO, b::BasisFunction)
    print(io, "Basis function i = ", b.i, "\n")
    print(io, "  from ")
    summary(io, basis(b))
    ind = support(b)
    i, j = first(ind), last(ind)
    ts = knots(b)
    ti, tj = map(n -> ts[n], (i, j))
    print(io, "\n  support: [", ti, ", ", tj, ") (knots ", ind, ")")
end

"""
    support(b::BasisFunction) -> UnitRange{Int}

Get range of knots supported by the basis function.

Returns the knot range `i:j` such that the basis function support is
``t ∈ [t_i, t_j)``.
"""
support(b::BasisFunction) = support(basis(b), b.i)

"""
    support(B::AbstractBSplineBasis, i::Integer) -> UnitRange{Int}

Get range of knots supported by the ``i``-th basis function.
"""
support(B::AbstractBSplineBasis, i::Integer) = i:(i + order(B))

"""
    nonzero_in_segment(B::AbstractBSplineBasis, n::Int) -> UnitRange{Int}

Returns the range of basis functions that are non-zero in a given knot segment.

The ``n``-th knot segment is defined by ``Ω_n = [t_n, t_{n + 1}]``.

For [`BSplineBasis`](@ref) and `RecombinedBSplineBasis`, the number of non-zero
functions in any given segment is generally equal to the B-spline order ``k``.
This number decreases near the borders, but this is not significant when
B-spline knots have multiplicity ``k`` there (as is the default).

For B-spline bases, and excepting the borders, the non-zero B-splines are
``\\left\\{ b_i \\right\\}_{i = n - k + 1}^{n}``.
This function thus returns `(n - k + 1):N` when `B` is a `BSplineBasis`.

See also [`support`](@ref) for the inverse operation.
"""
function nonzero_in_segment(B::BSplineBasis, n::Int; N = length(B))
    a = clamp(n - order(B) + 1, 1, typemax(Int))
    b = clamp(n, typemin(Int), N)
    a:b
end

"""
    common_support(b1::BasisFunction, b2::BasisFunction, ...) -> UnitRange{Int}

Get range of knots commonly supported by different basis functions.

If the supports don't intersect, an empty range is returned (e.g. `6:5`),
following the behaviour of `intersect`. The lack of intersection can be checked
using `isempty`, which returns `true` for such a range.
"""
common_support(bs::Vararg{BasisFunction}) = ∩(support.(bs)...)

(b::BasisFunction)(x, op=Derivative(0)) = evaluate(basis(b), b.i, x, op, eltype(b))

"""
    evaluate(B::AbstractBSplineBasis, i::Integer, x,
             [op::AbstractDifferentialOp], [T=Float64])

Evaluate ``i``-th basis function in the given basis at `x` (can be a coordinate or a
vector of coordinates).

To evaluate a derivative, pass `Derivative(n)` as the `op` argument, with `n`
the derivative order.

More general differential operators, such as `Derivative(n) + λ Derivative(m)`,
are also supported.

See also [`evaluate!`](@ref).
"""
function evaluate end

evaluate(B::BSplineBasis, args...) = _evaluate(B, args...)

function _evaluate(B::AbstractBSplineBasis, i::Integer, x::Real,
                   op::AbstractDifferentialOp = Derivative(0),
                   ::Type{T} = Float64) where {T}
    if !checkbounds(Bool, B, i)
        throw(DomainError(i, lazy"Basis function index must be in $(eachindex(B))"))
    end
    k = order(B)
    t = knots(B)
    evaluate_diff(op, BSplineOrder(k), t, i, x, T)
end

_evaluate(B::AbstractBSplineBasis, i::Integer, x::Real, ::Type{T}) where {T} =
    _evaluate(B, i, x, Derivative(0), T)

# No derivative
evaluate_diff(::Derivative{0}, ::BSplineOrder{k}, t, i, x,
              ::Type{T}) where {k,T} = _evaluate(BSplineOrder(k), t, i, x, T)

# N-th derivative
function evaluate_diff(::Derivative{N}, ::BSplineOrder{k}, t, i, x,
                       ::Type{T}) where {N,k,T}
    @assert N > 0
    y = zero(T)
    dt = t[i + k - 1] - t[i]
    local dy::T  # make sure `y` doesn't change type, e.g. if T = Float32
    if !iszero(dt)
        # Recursively evaluate derivative `N - 1` of B-spline of order `k - 1`.
        dy = evaluate_diff(Derivative(N - 1), BSplineOrder(k - 1), t, i, x, T) / dt
        y += dy
    end
    dt = t[i + k] - t[i + 1]
    if !iszero(dt)
        dy = evaluate_diff(Derivative(N - 1), BSplineOrder(k - 1), t, i + 1, x, T) / dt
        y -= dy
    end
    (y * (k - 1)) :: T
end

# TODO this can be optimised... it should be possible to compute all derivatives
# at once.
@generated function evaluate_diff(
        ::DerivativeUnitRange{m, n}, args...,
    ) where {m, n}
    ex = :(())
    for i = m:n
        push!(ex.args, :(evaluate_diff(Derivative($i), args...)))
    end
    ex
end

# More general diff operators
evaluate_diff(S::ScaledDerivative, etc...) =
    S.α * evaluate_diff(S.D, etc...)

function evaluate_diff(S::DifferentialOpSum, etc...)
    vals = map(D -> evaluate_diff(D, etc...), (S.a, S.b))
    sum(vals)
end

# For evaluation at multiple locations (allocates a vector).
_evaluate(B, i, x::AbstractVector, args...) =
    _evaluate.(B, i, x, args...)

"""
    evaluate!(b::AbstractVector, B::BSplineBasis, i::Integer,
              x::AbstractVector, args...)

Evaluate i-th basis function at positions `x` and write result to `b`.

See also [`evaluate`](@ref).
"""
function evaluate!(b::AbstractVector{T}, B::AbstractBSplineBasis, i,
                   x::AbstractVector, args...) where {T}
    broadcast!(x -> evaluate(B, i, x, args..., T), b, x)
end

# Specialisation for first order B-splines.
function _evaluate(::BSplineOrder{1}, t::AbstractVector, i::Integer,
                   x::Real, ::Type{T}) where {T}
    # Local support of the B-spline.
    @inbounds ta = t[i]
    @inbounds tb = t[i + 1]

    in_local_support(x, t, ta, tb) ? one(T) : zero(T)
end

# General case of order k >= 2.
function _evaluate(::BSplineOrder{k}, t::AbstractVector, i::Integer,
                   x::Real, ::Type{T}) where {T,k}
    k::Int
    @assert k >= 2

    # Local support of the B-spline.
    @inbounds ta = t[i]
    @inbounds tb = t[i + k]

    if !in_local_support(x, t, ta, tb)
        return zero(T)
    end

    @inbounds ta1 = t[i + 1]
    @inbounds tb1 = t[i + k - 1]

    # Recursively evaluate lower-order B-splines.
    y = zero(T)
    local dy::T  # make sure `y` doesn't change type, e.g. if T = Float32

    if tb1 != ta
        dy = _evaluate(BSplineOrder(k - 1), t, i, x, T) *
            (x - ta) / (tb1 - ta)
        y += dy
    end

    if ta1 != tb
        dy = _evaluate(BSplineOrder(k - 1), t, i + 1, x, T) *
            (tb - x) / (tb - ta1)
        y += dy
    end

    y :: T
end

# In principle, the support of a B-spline is [ta, tb[.
# The exception is when tb is the last point in the domain, in which case we
# include it in the interval (otherwise the right boundary always has zero value).
@inline in_local_support(x, t, ta, tb) = (ta <= x < tb) || (x == tb == t[end])
