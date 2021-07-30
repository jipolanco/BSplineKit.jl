@doc raw"""
    galerkin_projection(
        f, B::AbstractBSplineBasis,
        [deriv = Derivative(0)], [VectorType = Vector{Float64}],
    )

Perform Galerkin projection of a function `f` onto the given basis.

By default, returns a vector with values

```math
φ_i = ⟨ b_i, f ⟩
= ∫_a^b b_i(x) \, f(x) \, \mathrm{d}x,
```

where ``a`` and ``b`` are the boundaries of the B-spline basis
``\{ b_i \}_{i = 1}^N``.

The integrations are performed using Gauss--Legendre quadrature.
The number of quadrature nodes is chosen so that the result is exact when ``f``
is a polynomial of degree ``k - 1`` (or, more generally, a spline belonging to
the space spanned by the basis `B`).
Here ``k`` is the order of the B-spline basis.
In the more general case, this function returns a quadrature approximation of
the projection.

See also [`galerkin_projection!`](@ref) for the in-place operation, and
[`galerkin_matrix`](@ref) for more details.
"""
function galerkin_projection(
        f, B::AbstractBSplineBasis,
        deriv = Derivative(0),
        ::Type{V} = Vector{Float64},
    ) where {V <: AbstractVector}
    N = length(B)
    φ = V(undef, N)
    galerkin_projection!(f, φ, B, deriv)
end

"""
    galerkin_projection!(
        f, φ::AbstractVector, B::AbstractBSplineBasis, [deriv = Derivative(0)],
    )

Compute Galerkin projection ``φ_i = ⟨ b_i, f ⟩``.

See [`galerkin_projection`](@ref) for details.
"""
function galerkin_projection!(
        f, φ::AbstractVector, B::AbstractBSplineBasis, deriv = Derivative(0),
    )
    N = length(B)
    if length(φ) != N
        throw(DimensionMismatch(
            "output vector must have length $N (got length $(length(φ)))"
        ))
    end
    Base.require_one_based_indexing(φ)

    k = order(B)
    ts = knots(B)

    # Quadrature information (nodes, weights).
    quadx, quadw = _quadrature_prod(Val(2k - 2))
    @assert length(quadx) == k  # we need k quadrature points per knot segment

    fill!(φ, 0)

    nlast = last(eachindex(ts))

    # We loop over all knot segments Ω[n] = (ts[n], ts[n + 1]).
    # For all B-splines with support in this segment, we integrate the product
    # B[i] * f over this segment, adding the result to φ[i].
    @inbounds for n in eachindex(ts)
        n == nlast && break
        tn, tn1 = ts[n], ts[n + 1]
        tn1 == tn && continue  # interval of length = 0

        metric = QuadratureMetric(tn, tn1)

        # Unnormalise quadrature nodes, such that xs ∈ [tn, tn1]
        xs = metric .* quadx
        # @assert all(x -> tn ≤ x ≤ tn1, xs)

        is = nonzero_in_segment(B, n)

        # This is a property of B-spline bases, which should be preserved by
        # derived (recombined) bases.
        @assert 0 < length(is) ≤ k

        # Evaluate all required basis functions on quadrature nodes.
        bis = eval_basis_functions(B, is, xs, deriv)
        fs = map(f, xs)

        for (ni, i) in enumerate(is)
            φ[i] += metric.α * ((bis[ni] .* fs) ⋅ quadw)
        end
    end

    φ
end
