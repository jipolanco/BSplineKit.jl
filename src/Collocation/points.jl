"""
    SelectionMethod

Abstract type defining a method for choosing collocation points from spline
knots.
"""
abstract type SelectionMethod end

"""
    AvgKnots <: SelectionMethod

Each collocation point is chosen as a sliding average over `k - 1` knots:

```math
x_i = \\frac{1}{k - 1} ∑_{j = 1}^{k - 1} t_{i + j}
```

The resulting collocation points are sometimes called Greville sites
(de Boor 2001).
"""
struct AvgKnots <: SelectionMethod end

struct GrevilleSiteIterator{Basis <: AbstractBSplineBasis}
    basis :: Basis
end

BSplines.basis(it::GrevilleSiteIterator) = it.basis
Base.length(it::GrevilleSiteIterator) = length(basis(it))
Base.eltype(it::GrevilleSiteIterator) = float(eltype(knots(basis(it))))

# TODO
# - optimise window averaging?
function Base.iterate(it::GrevilleSiteIterator, i = 0)
    B = basis(it)
    i === length(B) && return nothing
    # Account for basis recombination / boundary conditions
    ii = i + first(num_constraints(B))
    k = order(B)
    ts = knots(B)
    T = eltype(it)
    x = zero(T)
    @inbounds for j = 2:k
        x += ts[ii + j]
    end
    x /= k - 1
    x, i + 1
end

"""
    collocation_points(B::AbstractBSplineBasis; method=Collocation.AvgKnots())

Define and return adapted collocation points for evaluation of splines.

The number of returned collocation points is equal to the number of functions in
the basis.

Note that if `B` is a [`RecombinedBSplineBasis`](@ref) (adapted for boundary value
problems), collocation points are not included at the boundaries, since the
boundary conditions are implicitly satisfied by the basis.

In principle, the choice of collocation points is not unique.
The selection method can be chosen via the `method` argument.
For now, only a single method is accepted:

- [`Collocation.AvgKnots()`](@ref)

See also [`collocation_points!`](@ref).
"""
function collocation_points(
        B::AbstractBSplineBasis; method::SelectionMethod=AvgKnots())
    x = similar(knots(B), length(B))
    collocation_points!(x, B, method=method)
end

"""
    collocation_points!(x::AbstractVector, B::AbstractBSplineBasis;
                        method::SelectionMethod=AvgKnots())

Fill vector with collocation points for evaluation of splines.

See [`collocation_points`](@ref) for details.
"""
function collocation_points!(
        x::AbstractVector, B::AbstractBSplineBasis;
        method::SelectionMethod=AvgKnots())
    N = length(B)
    if N != length(x)
        throw(ArgumentError(
            "number of collocation points must match number of B-splines"))
    end
    collocation_points!(method, x, B)
end

function collocation_points!(::AvgKnots, x, B::AbstractBSplineBasis)
    N = length(B)           # number of functions in basis
    @assert length(x) == N
    k = order(B)
    t = knots(B)
    T = eltype(x)

    # For recombined bases, skip points at the boundaries.
    # Note that j = 0 for non-recombined bases, i.e. boundaries are included.
    j, _ = num_constraints(B)

    v::T = inv(k - 1)
    a::T, b::T = boundaries(B)

    for i in eachindex(x)
        j += 1  # generally i = j, unless x has weird indexation
        xi = zero(T)

        for n = 1:k-1
            xi += T(t[j + n])
        end
        xi *= v

        # Make sure that the point is inside the domain.
        # This may not be the case if end knots have multiplicity less than k.
        x[i] = clamp(xi, a, b)
    end

    x
end
