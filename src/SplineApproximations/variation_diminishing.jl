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

function approximate(
        f::F, Bs::Tuple{Vararg{AbstractBSplineBasis}},
        m::VariationDiminishing,
    ) where {F}
    xtest = map(B -> first(knots(B)), Bs)  # test coordinate for determining types
    T = typeof(f(xtest...))
    S = Spline(undef, Bs, T)
    A = SplineApproximation(m, S, nothing)
    approximate!(f, A)
end

function _approximate!(f::F, A, m::VariationDiminishing) where {F}
    @assert method(A) === m
    Bs = Splines.bases(A)
    S = spline(A)
    cs = coefficients(S)
    _approximate_VD!(f, cs, Bs)
    A
end

# 1D case
function _approximate_VD!(
        f::F, cs::AbstractVector, Bs::NTuple{1},
    ) where {F}
    B = first(Bs)
    for (i, x) in zip(eachindex(cs), GrevilleSiteIterator(B))
        @inbounds cs[i] = f(x)
    end
    cs
end

# N-D case
function _approximate_VD!(
        f::F, cs::AbstractArray{T, N},
        Bs::Tuple{Vararg{AbstractBSplineBasis, N}},
    ) where {F, T, N}
    @assert N ≥ 2
    iter = Iterators.product(map(GrevilleSiteIterator, Bs)...)
    @assert axes(iter) == axes(cs)
    @inbounds for (n, xs) ∈ zip(eachindex(cs), iter)
        cs[n] = f(xs...)
    end
    cs
end
