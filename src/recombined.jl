"""
    RecombinedBSplineBasis{D, k, T}

Functional basis defined from the recombination of a [`BSplineBasis`](@ref)
in order to satisfy certain homogeneous boundary conditions (BCs).

The basis recombination technique is a common way of applying BCs in Galerkin
methods. It is described for instance in Boyd 2000 (ch. 6), in the context of
a Chebyshev basis. In this approach, the original basis is "recombined" so that
each basis function individually satisfies the BCs.

The new basis, `{ϕⱼ(x), j ∈ 1:(N-2)}`, has two fewer functions than the original
B-spline basis, `{bⱼ(x), j ∈ 1:N}`. Due to this, the number of collocation
points needed to obtain a square collocation matrix is `N - 2`. In particular,
for the matrix to be invertible, there must be **no** collocation points at the
boundaries.

Thanks to the local support of B-splines, basis recombination involves only a
little portion of the original B-spline basis. For instance, since there is only
one B-spline that is non-zero at each boundary, removing that function from the
basis is enough to apply homogeneous Dirichlet BCs. Imposing BCs for derivatives
is a bit more complex, but not much.

# Order of boundary condition

The type parameter `D` represents the order of the BC. The recombined basis
requires the specification of a `Derivative` object determining the order of the
homogeneous BCs to be applied at the two boundaries.

Some typical choices are:

- `Derivative(0)` sets homogeneous Dirichlet BCs (`u = 0` at the
  boundaries) by removing the first and last B-splines, i.e. ϕ₁ = b₂;

- `Derivative(1)` sets homogeneous Neumann BCs (`du/dx = 0` at the
  boundaries) by adding the two first (and two last) B-splines,
  i.e. ϕ₁ = b₁ + b₂.

For now, the two boundaries are given the same BC (but this could be easily
extended...). And actually, only the two above choices are available for now.

---

    RecombinedBSplineBasis(::Derivative{D}, B::BSplineBasis)

Construct `RecombinedBSplineBasis` from B-spline basis `B`, satisfying
homogeneous boundary conditions of order `D >= 0`.
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
    RecombinedBSplineBasis(::Derivative{D}, args...; kwargs...)

Construct [`RecombinedBSplineBasis`](@ref) from B-spline basis, satisfying
homogeneous boundary conditions of order `D >= 0`.

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

knots(R::RecombinedBSplineBasis) = knots(parent(R))
order(R::RecombinedBSplineBasis{D,k}) where {D,k} = k
Base.eltype(::Type{RecombinedBSplineBasis{D,k,T}}) where {D,k,T} = T

# For homogeneous Dirichlet BCs: just shift the B-spline basis (removing b₁).
evaluate_bspline(R::RecombinedBSplineBasis{0}, j, args...) =
    evaluate_bspline(parent(R), j + 1, args...)

# Homogeneous Neumann BCs.
function evaluate_bspline(R::RecombinedBSplineBasis, j, args...)
    Nb = length(R)  # length of recombined basis
    B = parent(R)
    if j == 1
        evaluate_bspline(B, 1, args...) + evaluate_bspline(B, 2, args...)
    elseif j == Nb
        evaluate_bspline(B, Nb + 1, args...) + evaluate_bspline(B, Nb + 2, args...)
    else
        # Same as for Dirichlet.
        evaluate_bspline(B, j + 1, args...)
    end
end
