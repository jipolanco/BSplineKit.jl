"""
    RecombinedBSplineBasis{orders, k, T}

Functional basis defined from the recombination of a [`BSplineBasis`](@ref)
in order to satisfy certain homogeneous boundary conditions (BCs).

# Extended help

The basis recombination technique is a common way of applying BCs in Galerkin
methods. It is described for instance in Boyd 2000 (ch. 6), in the context of
a Chebyshev basis. In this approach, the original basis is "recombined" so that
each basis function individually satisfies the BCs.

The new basis, ``\\{ϕ_j(x), 1 ≤ j ≤ N-2\\}``, has two fewer functions
than the original B-spline basis, ``\\{b_j(x), 1 ≤ j ≤ N\\}``.
Due to this, the number of collocation points needed to obtain a square
collocation matrix is ``N - 2``. In particular, for the matrix to be invertible,
there must be **no** collocation points at the boundaries.
More generally, as described further below, it is possible to simultaneously
impose different boundary conditions, which further decreases the number of
degrees of freedom.

Thanks to the local support of B-splines, basis recombination involves only a
little portion of the original B-spline basis. For instance, since there is only
one B-spline that is non-zero at each boundary, removing that function from the
basis is enough to apply homogeneous Dirichlet BCs. Imposing BCs for derivatives
is a bit more complex, but not much.

## Order of the boundary condition

In this section, we consider the simplest case where a single homogeneous
boundary condition of derivative order ``n`` is to be satisfied by the basis.

The recombined basis requires the specification of a `Derivative` object
determining the order of the homogeneous BCs to be applied at the two
boundaries.
Evidently, the order of the B-spline needs to be ``k ≥ n + 1``, since a B-spline
of order ``k`` is a ``C^{k - 1}``-continuous function (except on the knots
where it is ``C^{k - 1 - p}``, with ``p`` the knot multiplicity).

Some usual choices are:

- `Derivative(0)` sets homogeneous Dirichlet BCs (``u = 0`` at the
  boundaries) by removing the first and last B-splines, i.e. ``ϕ_1 = b_2``;

- `Derivative(1)` sets homogeneous Neumann BCs (``u' = 0`` at the
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
The chosen solution is to set ``ϕ_i = b_i - α_i b_3`` for ``i ∈ \\{1, 2\\}``,
with ``α_i = b_i'' / b_3''``. All these considerations apply similarly to the
right boundary.

This generalises easily to higher order BCs.
First note that, due to the partition of unity property of the B-spline basis,
we have

```math
∑_j b_j(x) = 1 \\quad ⇒
\\quad ∑_j \\frac{\\mathrm{d}^n b_j}{\\mathrm{d}x^n}(x) = 0
\\text{ for } n ≥ 1.
```

Secondly, only the first ``n + 1`` B-splines have non-zero ``n``-th derivative on
the left boundary. Hence, to enforce a derivative to be zero, the first ``n + 1``
B-splines should be recombined linearly into ``n`` independent basis functions.

Finally, note that in the current implementation, it is not possible to impose
different boundary conditions on both boundaries.

## Multiple boundary conditions

As an option, the recombined basis may simultaneously satisfy homogeneous BCs of
different orders. In this case, a list of `Derivative`s must be passed.
The list must be sorted in increasing order.

Presently, the only supported case is where all orders from ``0`` to ``n`` are
present.
In this case, the resulting basis is simply obtained by removing the first (and
last) ``n + 1`` functions from the original B-spline basis.
This can be seen as a generalisation of the Dirichlet case described in the
previous section.

For instance, if `(Derivative(0), Derivative(1))` is passed, then the basis
simultaneously satisfies homogeneous Dirichlet and Neumann BCs at the two
boundaries.
The resulting basis is ``ϕ_1 = b_3, ϕ_2 = b_4, …, ϕ_{N - 4} = b_{N - 2}``.

---

    RecombinedBSplineBasis(::Derivative{n}, B::BSplineBasis)

Construct `RecombinedBSplineBasis` from B-spline basis `B`, satisfying
homogeneous boundary conditions of order ``n ≥ 0``.

---

    RecombinedBSplineBasis((::Derivative{n1}, ::Derivative{n2}, ...),
                           B::BSplineBasis)

Construct `RecombinedBSplineBasis` simultaneously satisfying homogeneous BCs of
all the given derivative orders.
The list of derivative orders must be sorted in increasing order.
"""
struct RecombinedBSplineBasis{
            k, T, Parent <: BSplineBasis{k,T},
            DiffOps <: Tuple{Vararg{AbstractDifferentialOp}},
            RMatrix <: RecombineMatrix{Q,DiffOps} where Q,
        } <: AbstractBSplineBasis{k,T}
    B :: Parent   # original B-spline basis
    ops :: DiffOps  # list of differential operators for BCs
    M :: RMatrix  # basis recombination matrix

    function RecombinedBSplineBasis(ops::Tuple{Vararg{AbstractDifferentialOp}},
                                    B::BSplineBasis{k,T}) where {k,T}
        Parent = typeof(B)
        M = RecombineMatrix(ops, B)
        RMatrix = typeof(M)
        Ops = typeof(ops)
        new{k,T,Parent,Ops,RMatrix}(B, ops, M)
    end

    RecombinedBSplineBasis(deriv::Derivative, args...) =
        RecombinedBSplineBasis((deriv, ), args...)
end

function Base.show(io::IO, R::RecombinedBSplineBasis)
    # This is somewhat consistent with the output of the BSplines package.
    println(io, length(R), "-element ", typeof(R), ':')
    println(io, " boundary conditions: ", constraints(R))
    println(io, " order: ", order(R))
    print(io, " knots: ", knots(R))
    nothing
end

"""
    RecombinedBSplineBasis(order, args...; kwargs...)

Construct [`RecombinedBSplineBasis`](@ref) from B-spline basis, satisfying
homogeneous boundary conditions of order ``n ≥ 0``.

This variant does not require a previously constructed [`BSplineBasis`](@ref).
Arguments are passed to the `BSplineBasis` constructor.
"""
RecombinedBSplineBasis(order, args...; kwargs...) =
    RecombinedBSplineBasis(order, BSplineBasis(args...; kwargs...))

"""
    parent(R::RecombinedBSplineBasis)

Get original B-spline basis.
"""
Base.parent(R::RecombinedBSplineBasis) = R.B

"""
    recombination_matrix(R::AbstractBSplineBasis)

Get [`RecombineMatrix`](@ref) associated to the recombined basis.

For non-recombined bases such as [`BSplineBasis`](@ref), this returns the
identity matrix (`LinearAlgebra.I`).
"""
recombination_matrix(R::RecombinedBSplineBasis) = R.M
recombination_matrix(B::AbstractBSplineBasis) = LinearAlgebra.I

"""
    length(R::RecombinedBSplineBasis)

Returns the number of functions in the recombined basis.
"""
@inline Base.length(R::RecombinedBSplineBasis) =
    length(parent(R)) - 2 * num_constraints(R)

boundaries(R::RecombinedBSplineBasis) = boundaries(parent(R))

knots(R::RecombinedBSplineBasis) = knots(parent(R))
order(R::RecombinedBSplineBasis{k}) where {k} = k
Base.eltype(::Type{RecombinedBSplineBasis{k,T}}) where {k,T} = T

"""
    constraints(R::AbstractBSplineBasis)
    constraints(A::RecombineMatrix)

Return the constraints (homogeneous boundary conditions) that the basis
satisfies on each boundary.

For non-recombined bases such as [`BSplineBasis`](@ref), this returns an empty
tuple.
"""
constraints(R::RecombinedBSplineBasis) = R.ops
constraints(B::BSplineBasis) = ()

"""
    num_constraints(R::AbstractBSplineBasis) -> Int
    num_constraints(A::RecombineMatrix) -> Int

Returns the number of constraints (i.e., number of BCs to satisfy) on each
boundary.

For instance, if `R` simultaneously satisfies Dirichlet and Neumann boundary
conditions on each boundary, this returns 2.

Note that for non-recombined bases such as [`BSplineBasis`](@ref), the number of
constraints is zero.
"""
num_constraints(B) = length(constraints(B))

"""
    num_recombined(R::AbstractBSplineBasis) -> Int
    num_recombined(A::RecombineMatrix) -> Int

Returns the number of recombined functions in the recombined basis for each
boundary.

For instance, if `R` satisfies Neumann boundary conditions, then only the first
and last basis functions are different from the original B-spline basis, e.g.
``ϕ_1 = b_1 + b_2``, and this returns 1.

For non-recombined bases such as [`BSplineBasis`](@ref), this returns zero.
"""
num_recombined(R::RecombinedBSplineBasis) = num_recombined(recombination_matrix(R))
num_recombined(B::AbstractBSplineBasis) = 0

# Support is shifted wrt BSplineBasis.
# TODO this is not very general: it will fail for general recombinations
# (but it will always work for the specific way we're recombining B-splines).
# A more correct way of doing this is using the recombination matrix,
# checking whether each knot is in the support of at least one B-spline.
# This would be slower though... but it would be good to have this verification
# in the tests.
support(R::RecombinedBSplineBasis, i::Integer) =
    support(parent(R), i) .+ num_constraints(R)

# For homogeneous Dirichlet BCs: just shift the B-spline basis (removing b₁).
# TODO check that this variant is actually being called...
evaluate_bspline(
        R::RecombinedBSplineBasis{k,T,P,Tuple{Derivative{0}}} where {k,T,P},
        j, args...,
    ) = evaluate_bspline(parent(R), j + 1, args...)

# Generalisation for D >= 1
function evaluate_bspline(R::RecombinedBSplineBasis, j, args...)
    B = parent(R)
    A = recombination_matrix(R)
    n = num_recombined(A)
    c = num_constraints(A)
    N = size(A, 1)

    block = which_recombine_block(A, j)

    j1 = j + c
    ϕ = evaluate_bspline(B, j1, args...)  # this B-spline is always needed
    T = typeof(ϕ)

    if block == 2
        # A[j + 1, j] should be 1
        return ϕ
    end

    ϕ::T *= A[j1, j]

    js = if block == 1
        1:(n + 1)
    else
        (N - n):N
    end

    for i ∈ js
        i == j1 && continue  # already added
        α = A[i, j]
        iszero(α) && continue
        ϕ::T += α * evaluate_bspline(B, i, args...)
    end

    ϕ::T
end
