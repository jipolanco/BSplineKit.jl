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

# These are needed for `collect(it)`
Base.length(it::GrevilleSiteIterator) = length(basis(it))
Base.eltype(it::GrevilleSiteIterator) = float(eltype(knots(basis(it))))

@inline function Base.iterate(it::GrevilleSiteIterator, state = nothing)
    B = basis(it)
    N = length(B)
    T = eltype(it)

    if state === nothing
        i = 0
        compensation = zero(T)
    else
        i, xprev, compensation = state
        i == N && return nothing
    end

    # For recombined bases, skip points at the boundaries.
    cl, cr = num_constraints(B)  # left/right constraints
    ii = i + cl

    k = order(B)
    ts = knots(B)
    lims = boundaries(B)

    if state === nothing
        x = zero(T)
        for j = 2:k
            x += @inbounds ts[ii + j]
        end
        x /= k - 1
    else
        # Optimised window averaging: reuse result from previous iteration.
        # We use Kahan summation to avoid roundoff errors.
        # https://en.wikipedia.org/wiki/Kahan_summation_algorithm
        y = @inbounds (ts[ii + k] - ts[ii + 1]) / (k - 1) - compensation
        x = xprev + y
        compensation = (x - xprev) - y
    end

    # Make sure that the point is inside the domain.
    # This may not be the case if end knots have multiplicity less than k.
    x = clamp(x, lims...)

    x, (i + 1, x, compensation)
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
    _collocation_points!(x, B, method)
end

_collocation_points!(xs, B::AbstractBSplineBasis, ::AvgKnots) =
    copyto!(xs, GrevilleSiteIterator(B))
