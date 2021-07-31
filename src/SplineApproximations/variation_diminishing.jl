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
    # TODO
    # - optimise window averaging operation
    # - reuse code here and in collocation_points: define lazy iterator over
    #   Greville sites?
    # - does this work properly with recombined bases (-> shift index of first
    #   knot)
    S = spline(A)
    N = length(S)
    ts = knots(S)
    k = order(S)
    cs = coefficients(S)
    degree = k - 1
    T = float(eltype(ts))
    @inbounds for i = 1:N
        # Compute Greville site
        x = zero(T)
        for j = 1:degree
            x += ts[i + j]
        end
        x /= degree
        cs[i] = f(x)
    end
    A
end
