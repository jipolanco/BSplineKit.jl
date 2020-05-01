"""
    RecombinedBSplineBasis{n, k, T}

Functional basis defined from the recombination of a [`BSplineBasis`](@ref)
in order to satisfy certain homogeneous boundary conditions (BCs).

The basis recombination technique is a common way of applying BCs in Galerkin
methods. It is described for instance in Boyd 2000 (ch. 6), in the context of
a Chebyshev basis. In this approach, the original basis is "recombined" so that
each basis function individually satisfies the BCs.

The new basis, ``{ϕⱼ(x), 1 ≤ j ≤ N-2}``, has two fewer functions than the original
B-spline basis, ``{bⱼ(x), 1 ≤ j ≤ N}``. Due to this, the number of collocation
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
  boundaries) by removing the first and last B-splines, i.e. ``ϕ₁ = b₂``;

- `Derivative(1)` sets homogeneous Neumann BCs (``du/dx = 0`` at the
  boundaries) by adding the two first (and two last) B-splines,
  i.e. ``ϕ₁ = b₁ + b₂``.

Higher order BCs are also possible (although it's probably not very useful in
real applications!).
For instance, `Derivative(2)` recombines the first three B-splines into two
basis functions that satisfy ``ϕ₁'' = ϕ₂'' = 0`` at the left boundary, while
ensuring that lower and higher-order derivatives keep degrees of freedom at the
boundary.
Note that simply adding the first three B-splines, as in ``ϕ₁ = b₁ + b₂ +
b₃``, makes the first derivative vanish as well as the second one, which is
unwanted.
The chosen solution is to set ``ϕᵢ = bᵢ - αᵢ b₃`` for ``i ∈ {1, 2}``,
with ``αᵢ = bᵢ'' / b₃''``. All these considerations apply similarly to the
right boundary.

This generalises easily to higher order BCs, and also applies to the lower order
BCs listed above.
To understand how this works, note that, due to the partition of unity property
of the B-spline basis:

``∑ⱼ bⱼ(x) = 1 ⇒ ∑ⱼ \\frac{d^n b_j}{dx^n}(x) = 0 \\text{ for } n ≥ 1.``

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
            D, k, T, Parent <: BSplineBasis{k,T},
        } <: AbstractBSplineBasis{k,T}
    B :: Parent  # original B-spline basis

    function RecombinedBSplineBasis(
            ::Derivative{D}, B::BSplineBasis{k,T}) where {k,T,D}
        Parent = typeof(B)
        new{D,k,T,Parent}(B)
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

# Homogeneous Neumann BCs.
function evaluate_bspline(R::RecombinedBSplineBasis{1}, j, args...)
    B = parent(R)
    N = length(R)
    if j == 1
        evaluate_bspline(B, 1, args...) + evaluate_bspline(B, 2, args...)
    elseif j == N
        evaluate_bspline(B, N + 1, args...) + evaluate_bspline(B, N + 2, args...)
    else
        # Same as for Dirichlet.
        evaluate_bspline(B, j + 1, args...)
    end
end

# Generalisation for D >= 2
function evaluate_bspline(R::RecombinedBSplineBasis{D}, j, args...) where {D}
    @assert D >= 2
    N = length(R)
    B = parent(R)

    ja = ntuple(identity, Val(D))  # = (1, 2, ..., D)
    jb = (N + 1) .- ja  # = (N, N - 1, ..., N - D + 1)

    if j ∉ ja && j ∉ jb
        return evaluate_bspline(B, j + 1, args...)
    end

    # At each border, we want 2 independent linear combinations of the first 3
    # B-splines such that ϕ₁'' = ϕ₂'' = 0 at x = a, and such that lower-order
    # derivatives keep at least one degree of freedom (i.e. they do *not*
    # vanish). A simple solution is to choose:
    #
    #     ϕ₁ = b₁ - α₁ b₃,
    #     ϕ₂ = b₂ - α₂ b₃,
    #
    # with αᵢ = bᵢ'' / b₃.

    a, b = boundaries(B)

    if j ∈ ja
        x = a
        js = (j, D + 1)
    elseif j ∈ jb
        x = b
        js = (j + 2, N + 2 - D)
    end

    # First, evaluate D-th derivatives of bⱼ at the boundary.
    # TODO replace with analytical formula?
    pj, p3 = evaluate_bspline.(B, js, x, Derivative(D))
    α = pj / p3

    bj, b3 = evaluate_bspline.(B, js, args...)
    bj - α * b3
end
