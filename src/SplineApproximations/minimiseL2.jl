using ..Galerkin:
    galerkin_matrix,
    galerkin_projection!,
    galerkin_projection  # for Documenter only...

using LinearAlgebra:
    cholesky!,
    Cholesky,
    ldiv!

using Base.Cartesian: @ntuple

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

function approximate(
        f::F, Bs::Tuple{Vararg{AbstractBSplineBasis}},
        m::MinimiseL2Error,
    ) where {F}
    xtest = map(B -> first(knots(B)), Bs)  # test coordinate for determining types
    T = typeof(f(xtest...))
    S = Spline(undef, Bs, T)

    Mfact = map(Bs) do B
        M = galerkin_matrix(B)  # by default this is a BandedMatrix
        cholesky!(M)
    end

    if length(Bs) ≥ 2
        coefs = coefficients(S)
        buf = similar(coefs, max(size(S)...))  # temporary buffer for linear systems
    else
        buf = nothing
    end

    data = (; Ms = Mfact, buf)
    A = SplineApproximation(m, S, data)
    approximate!(f, A)
end

function _approximate!(f::F, A, m::MinimiseL2Error) where {F}
    @assert method(A) === m
    S = spline(A)
    cs = coefficients(S)
    Bs = Splines.bases(S)
    _approximate_L2!(f, cs, Bs, data(A).Ms, data(A).buf)
    A
end

# 1D case
function _approximate_L2!(
        f::F, cs::AbstractVector, Bs::NTuple{1}, Ms::NTuple{1},
        ::Nothing,
    ) where {F}
    galerkin_projection!(f, cs, Bs)  # computes rhs onto cs
    ldiv!(first(Ms), cs)  # now cs = M \ rhs
    cs
end

# N-D case
@inline function _approximate_L2!(
        f::F, coefs::AbstractArray{T, N},
        Bs::Tuple{Vararg{AbstractBSplineBasis, N}},
        Ms::Tuple{Vararg{Cholesky, N}},
        buf::AbstractVector{T},
    ) where {F, T, N}
    @assert N ≥ 2

    # We first project the function `f` onto the tensor-product B-spline basis.
    galerkin_projection!(f, coefs, Bs)  # computes rhs onto coefs

    # The rest is quite similar to multidimensional interpolations.
    _approximate_L2_dim!(coefs, buf, Ms...)
end

# This is really similar to `_interpolate_dim!`.
# There are small differences due to Cj being the Cholesky factorisation of a
# BandedMatrix (and thus we need a contiguous buffer).
@inline function _approximate_L2_dim!(coefs, buf, Cj, Cnext...)
    N = ndims(coefs)
    R = length(Cnext)
    j = N - R
    L = j - 1
    inds = axes(coefs)
    inds_l = CartesianIndices(ntuple(d -> @inbounds(inds[d]), Val(L)))
    inds_r = CartesianIndices(ntuple(d -> @inbounds(inds[j + d]), Val(R)))

    Base.require_one_based_indexing(buf)
    @assert length(buf) ≥ length(inds[j])
    utmp = view(buf, inds[j])

    @inbounds for J ∈ inds_r, I ∈ inds_l
        # NOTE: the following line gives a wrong result when A is a Cholesky
        # decomposition of a BandedMatrix. The problem seems to be that
        # BandedMatrices.jl wrongly calls a LAPACK function which assumes that
        # the output is contiguous, which is not the case here.
        #
        # @views ldiv!(A, coefs[I, :, J])

        # The alternative is to copy to a contiguous buffer...
        coefs_ij = @view coefs[I, :, J]
        copy!(utmp, coefs_ij)
        ldiv!(Cj, utmp)
        copy!(coefs_ij, utmp)
    end

    _approximate_L2_dim!(coefs, buf, Cnext...)
end

@inline _approximate_L2_dim!(coefs, buf) = coefs
