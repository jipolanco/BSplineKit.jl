"""
    SplineApproximations

Module for function approximation using splines.

The general idea is to find the spline ``g(x)`` in a given spline space that
best approximates a known function ``f(x)``.
In other words, given a predefined [`BSplineBasis`](@ref), the objective is to
find some appropriate B-spline coefficients such that the resulting spline
appropriately approximates ``f``.
Different approximation approaches are implemented, trading accuracy and speed.
"""
module SplineApproximations

using ..BSplines
using ..SplineInterpolations
using ..Splines

# For documentation
using ..Galerkin: galerkin_matrix
using ..Recombinations: RecombinedBSplineBasis

export
    VariationDiminishing,
    ApproxByInterpolation,
    MinimiseL2Error,
    approximate,
    approximate!

"""
    AbstractApproxMethod

Abstract type describing a type of approximation method.
"""
abstract type AbstractApproxMethod end

struct SplineApproximation{
        ApproxSpline <: Spline,
        Method <: AbstractApproxMethod,
        Data,
    } <: SplineWrapper{ApproxSpline}
    method :: Method
    spline :: ApproxSpline  # approximating spline
    data   :: Data
end

method(a::SplineApproximation) = a.method
SplineInterpolations.spline(a::SplineApproximation) = a.spline
data(a::SplineApproximation) = a.data

function Base.show(io::IO, A::SplineApproximation)
    print(io, nameof(typeof(A)), " containing the ", spline(A))
    print(io, "\n approximation method: ", method(A))
    nothing
end

"""
    approximate(f, B::AbstractBSplineBasis, [method = ApproxByInterpolation(B)])

Approximate function `f` in the given basis, using the chosen method.

From lower to higher accuracy (and cost), the possible approximation methods are:

- [`VariationDiminishing`](@ref),
- [`ApproxByInterpolation`](@ref),
- [`MinimiseL2Error`](@ref).

See their respective documentations for details.

Note that, once an approximation has been performed, it's possible to
efficiently perform additional approximations (of other functions `f`) by
calling the in-place [`interpolate!`](@ref).
This completely avoids allocations and strongly reduces computation time.

# Examples

```jldoctest
julia> B = BSplineBasis(BSplineOrder(3), -1:0.4:1);

julia> S_interp = approximate(sin, B)
SplineApproximation containing the 7-element Spline{Float64}:
 order: 3
 knots: [-1.0, -1.0, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.0, 1.0]
 coefficients: [-0.8414709848078965, -0.7317273726556252, -0.39726989430226317, 0.0, 0.39726989430226317, 0.7317273726556252, 0.8414709848078965]
 approximation method: interpolation at [-1.0, -0.8, -0.4, 0.0, 0.4, 0.8, 1.0]

julia> sin(0.3), S_interp(0.3)
(0.29552020666133955, 0.2959895327282942)

julia> approximate!(exp, S_interp)
SplineApproximation containing the 7-element Spline{Float64}:
 order: 3
 knots: [-1.0, -1.0, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.0, 1.0]
 coefficients: [0.36787944117144233, 0.4403725424224455, 0.6570101184826601, 0.9801271149667085, 1.4622271917170906, 2.181107315860912, 2.718281828459045]
 approximation method: interpolation at [-1.0, -0.8, -0.4, 0.0, 0.4, 0.8, 1.0]

julia> exp(0.3), S_interp(0.3)
(1.3498588075760032, 1.3491015490105398)

julia> S_fast = approximate(exp, B, VariationDiminishing())
SplineApproximation containing the 7-element Spline{Float64}:
 order: 3
 knots: [-1.0, -1.0, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.0, 1.0]
 coefficients: [0.36787944117144233, 0.44932896411722156, 0.6703200460356393, 1.0, 1.4918246976412703, 2.225540928492468, 2.718281828459045]
 approximation method: VariationDiminishing()

julia> S_opt = approximate(exp, B, MinimiseL2Error())
SplineApproximation containing the 7-element Spline{Float64}:
 order: 3
 knots: [-1.0, -1.0, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.0, 1.0]
 coefficients: [0.368073806165329, 0.4403423586370571, 0.6570767565347361, 0.9802789580270397, 1.4621592525088214, 2.182012185989042, 2.7166900724062018]
 approximation method: MinimiseL2Error()

julia> x = 0.34; exp(x), S_opt(x), S_interp(x), S_fast(x)
(1.4049475905635938, 1.4044530324752085, 1.4044149581073815, 1.4328668494041878)
```
"""
approximate(f, B::AbstractBSplineBasis) =
    approximate(f, B, ApproxByInterpolation(B))

"""
    approximate!(f, A::SplineApproximation)

Approximate function `f` reusing a previous Spline approximation in a given
basis.

See [`approximate`](@ref) for details.
"""
approximate!(f, A::SplineApproximation) = _approximate!(f, A, method(A))

include("variation_diminishing.jl")
include("by_interpolation.jl")
include("minimiseL2.jl")

end
