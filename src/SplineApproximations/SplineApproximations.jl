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
SplineApproximation containing the 7-element Spline{1, Float64}:
 basis: 7-element BSplineBasis of order 3, domain [-1.0, 1.0]
 order: 3
 knots: [-1.0, -1.0, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.0, 1.0]
 coefficients: [-0.841471, -0.731727, -0.39727, 2.85767e-17, 0.39727, 0.731727, 0.841471]
 approximation method: interpolation at [-1.0, -0.8, -0.4, 0.0, 0.4, 0.8, 1.0]

julia> sin(0.3), S_interp(0.3)
(0.29552020666133955, 0.2959895327282942)

julia> approximate!(exp, S_interp)
SplineApproximation containing the 7-element Spline{1, Float64}:
 basis: 7-element BSplineBasis of order 3, domain [-1.0, 1.0]
 order: 3
 knots: [-1.0, -1.0, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.0, 1.0]
 coefficients: [0.367879, 0.440373, 0.65701, 0.980127, 1.46223, 2.18111, 2.71828]
 approximation method: interpolation at [-1.0, -0.8, -0.4, 0.0, 0.4, 0.8, 1.0]

julia> exp(0.3), S_interp(0.3)
(1.3498588075760032, 1.3491015490105396)

julia> S_fast = approximate(exp, B, VariationDiminishing())
SplineApproximation containing the 7-element Spline{1, Float64}:
 basis: 7-element BSplineBasis of order 3, domain [-1.0, 1.0]
 order: 3
 knots: [-1.0, -1.0, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.0, 1.0]
 coefficients: [0.367879, 0.449329, 0.67032, 1.0, 1.49182, 2.22554, 2.71828]
 approximation method: VariationDiminishing()

julia> S_opt = approximate(exp, B, MinimiseL2Error())
SplineApproximation containing the 7-element Spline{1, Float64}:
 basis: 7-element BSplineBasis of order 3, domain [-1.0, 1.0]
 order: 3
 knots: [-1.0, -1.0, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.0, 1.0]
 coefficients: [0.368074, 0.440342, 0.657077, 0.980279, 1.46216, 2.18201, 2.71669]
 approximation method: MinimiseL2Error()

julia> x = 0.34; exp(x), S_opt(x), S_interp(x), S_fast(x)
(1.4049475905635938, 1.4044530324752076, 1.4044149581073813, 1.4328668494041878)
```
"""
approximate(f::F, Bs::Tuple{Vararg{AbstractBSplineBasis}}) where {F} =
    approximate(f, Bs, ApproxByInterpolation(Bs))

approximate(f::F, B::AbstractBSplineBasis, args...) where {F} =
    approximate(f, (B,), args...)

"""
    approximate!(f, A::SplineApproximation)

Approximate function `f` reusing a previous Spline approximation in a given
basis.

See [`approximate`](@ref) for details.
"""
approximate!(f::F, A::SplineApproximation) where {F} = _approximate!(f, A, method(A))

include("variation_diminishing.jl")
include("by_interpolation.jl")
include("minimiseL2.jl")

end
