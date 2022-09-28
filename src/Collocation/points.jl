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

"""
    SameAsKnots <: SelectionMethod

Collocation points are chosen to match B-spline knots.

Note that this only makes sense when the number of degrees of freedom of the
B-spline basis (i.e. the `length` of the basis) matches the number of knots,
which is generally not the case.

Some examples of bases that satisfy this condition are:

- recombined B-spline bases ([`RecombinedBSplineBasis`](@ref)) with
  [`Natural`](@ref) boundary conditions;

- periodic B-spline bases ([`PeriodicBSplineBasis`](@ref)).

"""
struct SameAsKnots <: SelectionMethod end

default_method(::AbstractBSplineBasis) = AvgKnots()
default_method(::PeriodicBSplineBasis{k}) where {k} =
    iseven(k) ? SameAsKnots() : AvgKnots()

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
        i, tsum_prev, compensation = state
        i == N && return nothing
    end

    # For recombined bases, skip points at the boundaries.
    cl, _ = num_constraints(B)  # left/right constraints
    ii = i + cl

    k = order(B)
    ts = knots(B)

    if state === nothing
        tsum = zero(eltype(ts))
        for j = 2:k
            tsum += @inbounds ts[ii + j]
        end
    else
        # Optimised window averaging: reuse result from previous iteration.
        #
        # We use a two-step Kahan summation to avoid roundoff errors.
        # https://en.wikipedia.org/wiki/Kahan_summation_algorithm
        #
        # The non-compensated summation would be written as:
        # tsum = tsum_prev + ts[ii + k] - ts[ii + 1]

        # 1. Add next knot
        y = @inbounds ts[ii + k] - compensation
        tsum = tsum_prev + y
        compensation = (tsum - tsum_prev) - y

        tsum_prev = tsum

        # 2. Remove previous knot
        y = @inbounds -ts[ii + 1] - compensation
        tsum = tsum_prev + y
        compensation = (tsum - tsum_prev) - y
    end

    x = T(tsum / (k - 1))
    x = ensure_points_at_boundaries(B, i, x)

    # Make sure that the point is inside the domain.
    # This may not be the case if end knots have multiplicity less than k.
    x = clamp(x, boundaries(B)...)

    x, (i + 1, tsum, compensation)
end

# Ensure that there are collocation points at the boundaries.
# This is only relevant for regular BSplineBasis, and is not needed for
# recombined or periodic bases.
function ensure_points_at_boundaries(B::BSplineBasis, i, x)
    a, b = boundaries(B)
    N = length(B)
    if i == 0
        oftype(x, a)
    elseif i == N - 1
        oftype(x, b)
    else
        x
    end
end

ensure_points_at_boundaries(::AbstractBSplineBasis, i, x) = x

"""
    collocation_points(
        B::AbstractBSplineBasis,
        method::SelectionMethod = default_method(B),
    )

Define and return adapted collocation points for evaluation of splines.

The number of returned collocation points is equal to the number of functions in
the basis.

Note that if `B` is a [`RecombinedBSplineBasis`](@ref) (adapted for boundary value
problems), collocation points are not included at the boundaries, since the
boundary conditions are implicitly satisfied by the basis.

In principle, the choice of collocation points is not unique.
The selection method can be chosen via the `method` argument.
For now, the following  methods are accepted:

- [`Collocation.AvgKnots()`](@ref);
- [`Collocation.SameAsKnots()`](@ref), which requires the length of the basis to
  be equal to the number of knots.

The former is the default, except for periodic B-spline bases
([`PeriodicBSplineBasis`](@ref)) of *even* order ``k``, for which `SameAsKnots`
is the default.
(Note that for odd-order B-splines, this can lead to non-invertible collocation
matrices.)

See also [`collocation_points!`](@ref).
"""
function collocation_points end

# This is left for compatibility with older versions.
collocation_points(B::AbstractBSplineBasis; method = default_method(B)) =
    collocation_points(B, method)

# Returns the set of knots that are inside the domain.
# This is mainly relevant for periodic bases, where the number of knots is
# theoretically infinite.
# In that case, we return the set of knots in the "main" period, which is
# usually the set of knots that the user passed when creating the basis.
_knots_in_domain(B::AbstractBSplineBasis) = _knots_in_domain(knots(B))
_knots_in_domain(ts::AbstractVector) = ts
_knots_in_domain(ts::PeriodicKnots) = parent(ts)

function collocation_points(B::AbstractBSplineBasis, method::SelectionMethod)
    x = similar(knots(B), length(B))
    collocation_points!(x, B, method)
end

# Avoid allocation in the SameAsKnots case.
function collocation_points(B::AbstractBSplineBasis, ::SameAsKnots)
    ts = _knots_in_domain(B)
    length(ts) == length(B) ||
        throw(ArgumentError(
            "number of knots must match number of B-splines"
        ))
    ts
end

"""
    collocation_points!(
        x::AbstractVector, B::AbstractBSplineBasis,
        method::SelectionMethod = default_method(B),
    )

Fill vector with collocation points for evaluation of splines.

See [`collocation_points`](@ref) for details.
"""
function collocation_points! end

# This is left for compatibility with older versions.
function collocation_points!(
        x::AbstractVector, B::AbstractBSplineBasis;
        method = default_method(B),
    )
    collocation_points!(x, B, method)
end

function collocation_points!(
        x::AbstractVector, B::AbstractBSplineBasis, method::SelectionMethod,
    )
    N = length(B)
    if N != length(x)
        throw(ArgumentError(
            "number of collocation points must match number of B-splines"
        ))
    end
    _collocation_points!(x, B, method)
end

_collocation_points!(xs, B::AbstractBSplineBasis, ::AvgKnots) =
    copyto!(xs, GrevilleSiteIterator(B))

_collocation_points!(xs, B::AbstractBSplineBasis, ::SameAsKnots) =
    copyto!(xs, _knots_in_domain(B))
