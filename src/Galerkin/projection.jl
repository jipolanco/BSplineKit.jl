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
        f::F, B::AbstractBSplineBasis,
        deriv = Derivative(0),
        ::Type{V} = Vector{Float64},
    ) where {F, V <: AbstractVector}
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
        f::F, φ::AbstractVector, B::AbstractBSplineBasis, deriv = Derivative(0),
    ) where {F}
    N = length(B)
    length(φ) == N || throw(DimensionMismatch("incorrect length of output vector φ"))
    Base.require_one_based_indexing(φ)

    k = order(B)
    ts = knots(B)

    # Quadrature information (nodes, weights).
    quadx, quadw = _quadrature_prod(Val(2k - 2))
    @assert length(quadx) == k  # we need k quadrature points per knot segment

    T = eltype(φ)
    fill!(φ, 0)
    nlast = last(eachindex(ts))
    ioff = first(num_constraints(B))

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

        for (x, w) ∈ zip(xs, quadw)
            ilast = n - ioff
            _, bis = evaluate_all(B, x, deriv, T; ileft = ilast)
            y = metric.α * w * f(x)
            for (δi, bi) ∈ pairs(bis)
                i = ilast + 1 - δi
                if i ∉ axes(φ, 1)  # this can be true for recombined bases
                    continue
                end
                @inbounds φ[i] += y * bi
            end
        end
    end

    φ
end

function galerkin_projection!(
        f::F, φ::AbstractArray{T, N},
        Bs::Tuple{Vararg{AbstractBSplineBasis, N}},
        derivs::Tuple{Vararg{Derivative, N}} = ntuple(_ -> Derivative(0), Val(N)),
    ) where {F, T, N}
    dims = map(length, Bs)
    size(φ) == dims || throw(DimensionMismatch("incorrect length of output vector φ"))
    Base.require_one_based_indexing(φ)

    ks_all = map(order, Bs)
    ts_all = map(knots, Bs)

    quads = map(ks_all) do k
        nodes, weights = _quadrature_prod(Val(2k - 2))
        @assert length(nodes) == k
        (; nodes, weights)
    end

    fill!(φ, 0)
    inds = CartesianIndices(map(ts -> axes(ts, 1)[begin:end-1], ts_all))
    Ioff = CartesianIndex(map(B -> first(num_constraints(B)), Bs))

    # We loop over all knot segments Ω[I] = t₁[I₁:I₁+1] × t₂[I₂:I₂+1] × …
    @inbounds for I ∈ inds
        knot_intervals = map(ts_all, Tuple(I)) do ts, n
            ts[n], ts[n + 1]
        end

        if any(interv -> ==(interv...), knot_intervals)  # interval has zero "volume"
            continue
        end

        metrics = map(interv -> QuadratureMetric(interv...), knot_intervals)

        # Unnormalise quadrature nodes, such that xs ∈ [tn, tn1]
        nodes = map(metrics, quads) do m, q
            m .* q.nodes
        end
        weights = map(metrics, quads) do m, q
            m.α * q.weights
        end

        Ilast = I - Ioff

        # Map over all dimensions (1:N)
        bs_at_nodes = map(Bs, nodes, derivs, Tuple(Ilast)) do B, xs, deriv, ileft
            # Map over all quadrature nodes in the current dimension (1:k)
            map(xs) do x
                _, bs = evaluate_all(B, x, deriv, T; ileft = ileft)
                bs
            end
        end

        _projection_kernel!(f, φ, nodes, weights, bs_at_nodes, Ilast)
    end

    φ
end

# This is used in N-dimensional projections
@inline function _projection_kernel!(
        f::F, φ::AbstractArray{<:Any, N},
        nodes::Tuple{Vararg{Any, N}},
        weights::Tuple{Vararg{Any, N}},
        bs_at_nodes::Tuple{Vararg{Any, N}},
        Ilast::CartesianIndex{N},
    ) where {F, N}
    # Loop over all nodes (x, y, …) within the knot interval
    node_inds = CartesianIndices(map(eachindex, nodes))
    @inbounds for K ∈ node_inds
        node = getindex.(nodes, Tuple(K))  # quadrature node (x, y, …)
        ws = getindex.(weights, Tuple(K))
        bs_all = getindex.(bs_at_nodes, Tuple(K))  # all B-splines evaluated at node (x, y, …)
        fval = prod(ws) * f(node...)
        subinds = CartesianIndices(map(eachindex, bs_all))
        for δJ ∈ subinds
            bs = getindex.(bs_all, Tuple(δJ))
            J = Ilast + oneunit(Ilast) - δJ
            if !checkbounds(Bool, φ, J)  # this can be the case for recombined bases
                continue
            end
            φ[J] += fval * prod(bs)
        end
    end
    nothing
end
