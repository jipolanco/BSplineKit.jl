using ..Collocation: GrevilleSiteIterator

"""
    VariationDiminishing <: AbstractApproxMethod

Schoenberg's variation diminishing spline approximation.

Approximates a function ``f`` by setting the B-spline coefficients as
``c_i = f(x_i)``, where the locations ``x_i`` are chosen as the Greville sites:

```math
x_i = \\frac{1}{k - 1} ∑_{j = 1}^{k - 1} t_{i + j}.
```

This method is expected to preserve the shape of the function.
However, it may be very inaccurate as an actual approximation of it.

For details, see e.g. de Boor 2001, chapter XI.

!!! warning

    Currently, this method is not guaranteed to work well with recombined
    B-spline bases (of type [`RecombinedBSplineBasis`](@ref)).

"""
struct VariationDiminishing <: AbstractApproxMethod end

function approximate(f, B::AbstractBSplineBasis, m::VariationDiminishing)
    T = typeof(f(first(knots(B))))
    S = Spline{T}(undef, B)
    A = SplineApproximation(m, S, nothing)
    approximate!(f, A)
end

function _approximate!(f, A, m::VariationDiminishing)
    @assert method(A) === m
    B = basis(A)
    S = spline(A)
    cs = coefficients(S)
    for (i, x) in zip(eachindex(cs), GrevilleSiteIterator(B))
        @inbounds cs[i] = f(x)
    end
    A
end
