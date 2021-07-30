"""
    SplineApproximations

Module for function approximation using splines.

The general idea is to find the spline ``g(x)`` in a given spline space that
best approximates a known function ``f(x)``.
In other words, given a predefined [`BSplineBasis`](@ref), the objective is to
find some appropriate B-spline coefficients such that the resulting spline
appropriately approximates ``f``.
Different approximation approaches are proposed, trading accuracy and speed.
"""
module SplineApproximations

using ..BSplines
using ..Collocation: collocation_points
using ..SplineInterpolations
using ..Splines

# For documentation
using ..Galerkin: galerkin_matrix
using ..Recombinations: RecombinedBSplineBasis

export
    VariationDiminishing,
    ApproxByInterpolation,
    MinimiseL2Error,
    MinimizeL2Error,
    approximate,
    approximate!

"""
    AbstractApproxMethod

Abstract type describing a type of approximation method.
"""
abstract type AbstractApproxMethod end

Base.show(io::IO, m::AbstractApproxMethod) = print(io, nameof(typeof(m)))

"""
    VariationDiminishing <: AbstractApproxMethod

Schoenberg's variation diminishing spline approximation.

Approximates a function ``f`` by setting the B-spline coefficients as
``c_i = f(x_i)``, where the locations ``x_i`` are chosen as the Greville sites:

```math
x_i = \\frac{1}{k - 1} ∑_{j = 1}^{k - 1} t_{i + j}
```

This method is expected to provide a decent low-accuracy approximation of the
function ``f`` .
For details, see e.g. de Boor 2001, chapter XI.

!!! warning

    Currently, this method is not guaranteed to work well with recombined
    B-spline bases (of type [`RecombinedBSplineBasis`](@ref)).

"""
struct VariationDiminishing <: AbstractApproxMethod end

Base.show(io::IO, ::VariationDiminishing) = print(io, "variation diminishing")

"""
    ApproxByInterpolation <: AbstractApproxMethod

Approximate function by a spline that passes through the given set of points.

The number of points must be equal to the length of the B-spline basis defining
the approximation space.

See belows for different ways of specifying the interpolation points.

---

    ApproxByInterpolation(xs::AbstractVector)

Specifies an approximation by interpolation at the given points `xs`.

---

    ApproxByInterpolation(B::AbstractBSplineBasis)

Specifies an approximation by interpolation at an automatically-determined set
of points, which are expected to be appropriate for the given basis.

The interpolation points are determined by calling [`collocation_points`](@ref).
"""
struct ApproxByInterpolation{Points <: AbstractVector} <: AbstractApproxMethod
    xs :: Points
end

Base.show(io::IO, m::ApproxByInterpolation) =
    print(io, "interpolation at ", m.xs)

ApproxByInterpolation(B::AbstractBSplineBasis) =
    ApproxByInterpolation(collocation_points(B))

@doc raw"""
    MinimiseL2Error <: AbstractApproxMethod

Approximate a given function ``f(x)`` by minimisation of the ``L^2`` distance
between ``f`` and its spline approximation ``g(x)``.

# Extended help

Minimises the ``L^2`` distance between the two functions:

```math
{\left\lVert f - g \right\rVert}^2 = \left< f - g, f - g \right>
```

where

```math
\left< u, v \right> = ∫_a^b u(x) \, v(x) \, \mathrm{d}x
```

is the inner product between two functions, and ``a`` and ``b`` are the
boundaries of the prescribed B-spline basis.
Here, ``g`` is the spline ``g(x) = ∑_{i = 1}^N c_i \, b_i(x)``, and
``\{ b_i \}_{i = 1}^N`` is a prescribed B-spline basis.

One can show that the optimal coefficients ``c_i`` minimising the ``L^2`` error
are the solution to the linear system ``\bm{M} \bm{c} = \bm{φ}``,
where ``M_{ij} = \left< b_i, b_j \right>`` (computed by [`galerkin_matrix`](@ref))
and ``φ_i = \left< b_i, f \right>``.

The integrals associated to ``\bm{M}`` and ``\bm{φ}`` are computed via
Gauss--Legendre quadrature.
The number of quadrature nodes is chosen as a function of the order ``k`` of the
prescribed B-spline basis, ensuring that ``\bm{M}`` is computed exactly (see
also [`galerkin_matrix`](@ref)).
In the particular case where ``f`` is a polynomial of degree ``k - 1``, this
also results in an exact computation of ``\bm{φ}``.
In more general cases, as long as ``f`` is smooth enough, this is still expected
to yield a very good approximation of the integral, and thus of the optimal coefficients ``c_i``.

"""
struct MinimiseL2Error <: AbstractApproxMethod end

const MinimizeL2Error = MinimiseL2Error

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
 approximation method: variation diminishing

julia> x = 0.3; exp(x), S_interp(x), S_fast(x)
(1.3498588075760032, 1.3491015490105398, 1.3764276336437629)
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

function approximate(f, B::AbstractBSplineBasis, m::ApproxByInterpolation)
    S = SplineInterpolation(undef, B, m.xs)
    A = SplineApproximation(m, spline(S), S)
    approximate!(f, A)
end

function _approximate!(f, A::SplineApproximation, m::ApproxByInterpolation)
    @assert method(A) === m
    S = data(A)
    xs = SplineInterpolations.interpolation_points(S)
    @assert xs === method(A).xs
    ys = coefficients(S)  # just to avoid allocating extra vector
    map!(f, ys, xs)  # equivalent to ys .= f.(xs)
    interpolate!(S, ys)
    A
end

function approximate(f, B::AbstractBSplineBasis, m::VariationDiminishing)
    S = Spline(undef, B)
    A = SplineApproximation(m, S, nothing)
    approximate!(f, A)
end

function _approximate!(f, A::SplineApproximation, m::VariationDiminishing)
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

end
