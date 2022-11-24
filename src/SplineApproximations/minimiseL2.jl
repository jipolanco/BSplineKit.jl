using ..Galerkin:
    galerkin_matrix,
    galerkin_projection!,
    galerkin_projection  # for Documenter only...

using LinearAlgebra:
    cholesky!,
    Cholesky,
    ldiv!

@doc raw"""
    MinimiseL2Error <: AbstractApproxMethod

Approximate a given function ``f(x)`` by minimisation of the ``L^2`` distance
between ``f`` and its spline approximation ``g(x)``.

# Extended help

Minimises the ``L^2`` distance between the two functions:

```math
{\left\lVert f - g \right\rVert}^2 = \left< f - g, f - g \right>,
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
where ``M_{ij} = \left< b_i, b_j \right>`` and ``φ_i = \left< b_i, f \right>``.
These two terms are respectively computed by [`galerkin_matrix`](@ref) and
[`galerkin_projection`](@ref).

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

function approximate(f, B::AbstractBSplineBasis, m::MinimiseL2Error)
    T = typeof(f(first(knots(B))))
    S = Spline{T}(undef, B)
    M = galerkin_matrix(B)  # by default it's a BandedMatrix

    # We annotate the return type to avoid inference issue in ArrayLayouts...
    # https://github.com/JuliaLinearAlgebra/ArrayLayouts.jl/issues/66
    Mfact = cholesky!(M) :: Cholesky{eltype(M), typeof(parent(M))}

    data = (; Mfact,)
    A = SplineApproximation(m, S, data)
    approximate!(f, A)
end

function _approximate!(f, A, m::MinimiseL2Error)
    @assert method(A) === m
    S = spline(A)
    cs = coefficients(S)
    galerkin_projection!(f, cs, basis(S))  # computes rhs onto cs
    cs_data = Splines.unwrap_coefficients(S)  # same as cs, except for periodic splines
    ldiv!(data(A).Mfact, cs_data)  # now cs = M \ rhs
    A
end
