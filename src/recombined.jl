"""
    RecombinedBSplineBasis{n, k, T}

Functional basis defined from the recombination of a [`BSplineBasis`](@ref)
in order to satisfy certain homogeneous boundary conditions (BCs).

The basis recombination technique is a common way of applying BCs in Galerkin
methods. It is described for instance in Boyd 2000 (ch. 6), in the context of
a Chebyshev basis. In this approach, the original basis is "recombined" so that
each basis function individually satisfies the BCs.

The new basis, ``{ϕ_j(x), 1 ≤ j ≤ N-2}``, has two fewer functions than the original
B-spline basis, ``{b_j(x), 1 ≤ j ≤ N}``. Due to this, the number of collocation
points needed to obtain a square collocation matrix is `N - 2`. In particular,
for the matrix to be invertible, there must be **no** collocation points at the
boundaries.

Thanks to the local support of B-splines, basis recombination involves only a
little portion of the original B-spline basis. For instance, since there is only
one B-spline that is non-zero at each boundary, removing that function from the
basis is enough to apply homogeneous Dirichlet BCs. Imposing BCs for derivatives
is a bit more complex, but not much.

# Order of the boundary condition

The type parameter `n` represents the order of the BC. The recombined basis
requires the specification of a `Derivative` object determining the order of the
homogeneous BCs to be applied at the two boundaries.
Evidently, the order of the B-spline needs to be ``k ≥ n + 1``, since a B-spline
of order ``k`` is a ``C^{k - 1}``-continuous function (except on the knots
where it is ``C^{k - 1 - p}``, with ``p`` the knot multiplicity).

Some usual choices are:

- `Derivative(0)` sets homogeneous Dirichlet BCs (``u = 0`` at the
  boundaries) by removing the first and last B-splines, i.e. ``ϕ_1 = b_2``;

- `Derivative(1)` sets homogeneous Neumann BCs (``du/dx = 0`` at the
  boundaries) by adding the two first (and two last) B-splines,
  i.e. ``ϕ_1 = b_1 + b_2``.

Higher order BCs are also possible (although it's probably not very useful in
real applications!).
For instance, `Derivative(2)` recombines the first three B-splines into two
basis functions that satisfy ``ϕ_1'' = ϕ_2'' = 0`` at the left boundary, while
ensuring that lower and higher-order derivatives keep degrees of freedom at the
boundary.
Note that simply adding the first three B-splines, as in ``ϕ_1 = b_1 + b_2 +
b_3``, makes the first derivative vanish as well as the second one, which is
unwanted.
The chosen solution is to set ``ϕ_i = b_i - α_i b_3`` for ``i ∈ {1, 2}``,
with ``α_i = b_i'' / b_3''``. All these considerations apply similarly to the
right boundary.

This generalises easily to higher order BCs, and also applies to the lower order
BCs listed above.
To understand how this works, note that, due to the partition of unity property
of the B-spline basis:

```math
∑_j b_j(x) = 1 \\quad ⇒ \\quad ∑_j \\frac{d^n b_j}{dx^n}(x) = 0
\\text{ for } n ≥ 1.
```

Moreover, only the first `n + 1` B-splines have non-zero `n`-th derivative at
the left boundary. Hence, to enforce a derivative to be zero, the first `n + 1`
B-splines should be recombined linearly into `n` independent basis functions.

For now, the two boundaries are given the same BC (but this could be easily
extended...).

---

    RecombinedBSplineBasis(::Derivative{n}, B::BSplineBasis)

Construct `RecombinedBSplineBasis` from B-spline basis `B`, satisfying
homogeneous boundary conditions of order `n >= 0`.
"""
struct RecombinedBSplineBasis{
            n, k, T, Parent <: BSplineBasis{k,T},
            RMatrix <: RecombineMatrix{Q,n} where Q,
        } <: AbstractBSplineBasis{k,T}
    B :: Parent   # original B-spline basis
    M :: RMatrix  # basis recombination matrix

    function RecombinedBSplineBasis(
            ::Derivative{n}, B::BSplineBasis{k,T}) where {k,T,n}
        Parent = typeof(B)
        M = RecombineMatrix(Derivative(n), B)
        RMatrix = typeof(M)
        new{n,k,T,Parent,RMatrix}(B, M)
    end
end

"""
    RecombinedBSplineBasis(::Derivative{n}, args...; kwargs...)

Construct [`RecombinedBSplineBasis`](@ref) from B-spline basis, satisfying
homogeneous boundary conditions of order `n >= 0`.

This variant does not require a previously constructed [`BSplineBasis`](@ref).
Arguments are passed to the `BSplineBasis` constructor.
"""
RecombinedBSplineBasis(order::Derivative, args...; kwargs...) =
    RecombinedBSplineBasis(order, BSplineBasis(args...; kwargs...))

"""
    parent(R::RecombinedBSplineBasis)

Get original B-spline basis.
"""
Base.parent(R::RecombinedBSplineBasis) = R.B

"""
    recombination_matrix(R::RecombinedBSplineBasis)

Get [`RecombineMatrix`](@ref) associated to the recombined basis.
"""
recombination_matrix(R::RecombinedBSplineBasis) = R.M

"""
    length(R::RecombinedBSplineBasis)

Returns the number of functions in the recombined basis.
"""
Base.length(R::RecombinedBSplineBasis) = length(parent(R)) - 2

boundaries(R::RecombinedBSplineBasis) = boundaries(parent(R))

knots(R::RecombinedBSplineBasis) = knots(parent(R))
order(R::RecombinedBSplineBasis{D,k}) where {D,k} = k
Base.eltype(::Type{RecombinedBSplineBasis{D,k,T}}) where {D,k,T} = T

"""
    order_bc(B::AbstractBSplineBasis) -> Union{Int,Nothing}

Get order of homogeneous boundary conditions satisfied by the basis.

For bases that don't satisfy any particular boundary conditions (like
[`BSplineBasis`](@ref)), this returns `nothing`.
"""
order_bc(::AbstractBSplineBasis) = nothing
order_bc(::RecombinedBSplineBasis{D}) where {D} = D

# Support is shifted by +1 wrt BSplineBasis.
support(R::RecombinedBSplineBasis, i::Integer) = support(parent(R), i) .+ 1

# For homogeneous Dirichlet BCs: just shift the B-spline basis (removing b₁).
evaluate_bspline(R::RecombinedBSplineBasis{0}, j, args...) =
    evaluate_bspline(parent(R), j + 1, args...)

# Generalisation for D >= 1
function evaluate_bspline(R::RecombinedBSplineBasis{n}, i, args...) where {n}
    B = parent(R)
    A = recombination_matrix(R)
    M, N = size(A)

    block = which_recombine_block(Derivative(n), i, M)

    i1 = i + 1
    ϕ = evaluate_bspline(B, i1, args...)  # this B-spline is always needed
    T = typeof(ϕ)

    if block == 2
        # A[i, i + 1] should be 1
        return ϕ
    end

    ϕ::T *= A[i, i1]

    js = if block == 1
        1:(n + 1)
    else
        (N - n):N
    end

    for j ∈ js
        j == i1 && continue  # already added
        α = A[i, j]
        iszero(α) && continue
        ϕ::T += α * evaluate_bspline(B, j, args...)
    end

    ϕ::T
end