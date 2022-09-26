"""
    BSplineKit.BoundaryConditions

Contains some boundary condition definitions.
"""
module BoundaryConditions

export Natural,
       Periodic,
       period

abstract type BoundaryCondition end

"""
    Natural <: BoundaryCondition

Generalised natural boundary condition.

This boundary condition is convenient for spline interpolations, as it
provides extra constraints enabling to equate the number of unique B-spline
knots to the number of data points.

For cubic splines (order ``k = 4``), this corresponds to [natural cubic
splines](https://en.wikipedia.org/wiki/Spline_(mathematics)#Examples), imposing
the second derivatives to be zero at the boundaries (``S''(a) = S''(b) = 0``).

For higher-order splines, this boundary condition generalises the standard
natural cubic splines, by setting derivatives of order ``2, 3, …, k/2`` to
be zero at the boundaries.
For instance, for ``k = 6`` (quintic splines), this imposes ``S'' = S''' = 0``.
In practice, BSplineKit.jl achieves this by using [basis recombination](@ref
basis-recombination-api).

Note that, for symmetry reasons, only even-order splines are supported by this BC.
"""
struct Natural <: BoundaryCondition end

"""
    Periodic <: BoundaryCondition

Represents periodic boundary conditions with a given period `L`.

---

    Periodic(L::Real)

Constructs periodic boundary conditions with period `L`.
"""
struct Periodic{T <: Real} <: BoundaryCondition
    period :: T
end

"""
    period(bc::Periodic) -> Real
    period(B::PeriodicBSplineBasis) -> Real
    period(ts::PeriodicKnots) -> Real

Returns the period `L` associated to a periodic boundary condition or B-spline
basis.
"""
period(bc::Periodic) = bc.period

end
