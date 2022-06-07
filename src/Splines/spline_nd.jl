# N-dimensional tensor-product splines.

(S::Spline)(xs...) = _evaluate_tensor_product(bases(S), S, xs)

# TODO optimise!
# Note that the evaluation of B-splines is by far the most expensive, so I'm not
# sure we can do much better.
function _evaluate_tensor_product(
        Bs::BasisTuple{N}, S::Spline{T, N}, xs::Tuple{Vararg{Any,N}},
    ) where {T, N}
    @assert N ≥ 2  # there's a separate function for the case N = 1
    @assert Bs === bases(S)
    coefs = coefficients(S)
    ks = orders(S)
    # Evaluate all B-splines: ((i, bxs), (j, bys), …)
    bsp = map((B, x) -> B(x), Bs, xs)
    inds_base = CartesianIndex(map(first, bsp) .+ 1)  # (i + 1, j + 1, …)
    bsp_values = map(last, bsp)  # (bxs, bys, …)
    val = zero(T)
    inds = map(k -> Base.OneTo(k), ks)
    @inbounds for δs ∈ CartesianIndices(inds)
        I = inds_base - δs
        coef = coefs[I]
        bs = getindex.(bsp_values, Tuple(δs))
        val += coef * prod(bs)
    end
    val
end
