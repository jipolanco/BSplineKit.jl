"""
    galerkin_matrix(
        B::AbstractBSplineBasis,
        [deriv = (Derivative(0), Derivative(0))],
        [MatrixType = BandedMatrix{Float64}],
    )

Compute Galerkin mass or stiffness matrix, as well as more general variants
of these.

# Extended help

The Galerkin mass matrix is defined as

```math
M_{ij} = ⟨ ϕ_i, ϕ_j ⟩ \\quad \\text{for} \\quad
i ∈ [1, N] \\text{ and } j ∈ [1, N],
```

where ``ϕ_i(x)`` is the ``i``-th basis function and `N = length(B)` is the
number of functions in the basis `B`.
Here, ``⟨f, g⟩`` is the [``L^2`` inner
product](https://en.wikipedia.org/wiki/Square-integrable_function#Properties)
between functions ``f`` and ``g``.

Since products of B-splines are themselves piecewise polynomials, integrals can
be computed exactly using [Gaussian quadrature
rules](https://en.wikipedia.org/wiki/Gaussian_quadrature).
To do this, we use Gauss--Legendre quadratures via the
[FastGaussQuadrature](https://github.com/JuliaApproximation/FastGaussQuadrature.jl)
package.

## Matrix layout and types

The mass matrix is banded with ``2k - 1`` bands.
Moreover, the matrix is symmetric and positive definite, and only ``k`` bands are
needed to fully describe the matrix.
Hence, a
[`Hermitian`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/index.html#LinearAlgebra.Hermitian)
view of an underlying matrix is returned.

By default, the underlying matrix holding the data is a `BandedMatrix` that
defines the upper part of the symmetric matrix.
Other types of container are also supported, including regular sparse matrices
(`SparseMatrixCSC`) and dense arrays (`Matrix`).
See [`collocation_matrix`](@ref) for a discussion on matrix types.

## Derivatives of basis functions

Galerkin matrices associated to the derivatives of basis functions may be
constructed using the optional `deriv` parameter.
For instance, if `deriv = (Derivative(0), Derivative(2))`, the matrix
``⟨ ϕ_i, ϕ_j'' ⟩`` is constructed, where primes denote derivatives.
Note that, if the derivative orders are different, the resulting matrix is not
symmetric, and a `Hermitian` view is not returned in those cases.

## Combining different bases

More generally, it is possible to compute matrices of the form
``⟨ ψ_i^{(n)}, ϕ_j^{(m)} ⟩``, where `n` and `m` are derivative orders, and ``ψ_i``
and ``ϕ_j`` belong to two *different* (but related) bases `B₁` and `B₂`.
For this, instead of the `B` parameter, one must pass a tuple of bases
`(B₁, B₂)`.
The restriction is that the bases must have the same parent B-spline basis.
That is, they must share the same set of B-spline knots and be of equal
polynomial order.

Note that, if both bases are different, the matrix will not be symmetric, and
will not even be square if the bases have different lengths.

In practice, this feature may be used to combine a B-spline basis `B`, with a
recombined basis `R` generated from `B` (see [Basis recombination](@ref)).
"""
function galerkin_matrix(
        Bs::NTuple{2,AbstractBSplineBasis},
        deriv::DerivativeCombination{2} = Derivative.((0, 0)),
        ::Type{M} = BandedMatrix{Float64},
    ) where {M <: AbstractMatrix}
    B1, B2 = Bs
    symmetry = M <: Hermitian
    if symmetry && (deriv[1] !== deriv[2] || B1 !== B2)
        throw(ArgumentError(
            "input matrix type incorrectly assumes symmetry"))
    end
    A = allocate_galerkin_matrix(M, Bs)
    galerkin_matrix!(A, Bs, deriv)
end

function galerkin_matrix(
        B::AbstractBSplineBasis,
        deriv::DerivativeCombination{2} = Derivative.((0, 0)),
        ::Type{Min} = BandedMatrix{Float64},
    ) where {Min <: AbstractMatrix}
    symmetry = deriv[1] === deriv[2]
    T = eltype(Min)
    M = symmetry ? Hermitian{T,Min} : Min
    galerkin_matrix((B, B), deriv, M)
end

galerkin_matrix(B, ::Type{M}) where {M <: AbstractMatrix} =
    galerkin_matrix(B, Derivative.((0, 0)), M)

function _check_bases(Bs::Tuple{Vararg{AbstractBSplineBasis}})
    ps = parent.(Bs)
    if any(ps .!== ps[1])
        throw(ArgumentError(
            "all bases must share the same parent B-spline basis"
        ))
    end
    nothing
end

allocate_galerkin_matrix(::Type{Hermitian{T,M}},
                         Bs) where {T, M <: AbstractMatrix} =
    Hermitian(allocate_galerkin_matrix(M, Bs, true))

allocate_galerkin_matrix(::Type{M}, Bs) where {M} =
    allocate_galerkin_matrix(M, Bs, false)

allocate_galerkin_matrix(::Type{M}, Bs, _symmetry) where {M <: AbstractMatrix} =
    M(undef, length.(Bs)...)

allocate_galerkin_matrix(::Type{SparseMatrixCSC{T}}, Bs, _symmetry) where {T} =
    spzeros(T, length.(Bs)...)

function allocate_galerkin_matrix(::Type{M}, Bs, symmetry) where {M <: BandedMatrix}
    _check_bases(Bs)
    B1, B2 = Bs
    k = order(B1)
    @assert k == order(B2)  # verified in _check_bases
    # When the bases are the same, the upper/lower bandwidths are Nb = k - 1
    # (total = 2k - 1 bands).
    # If the matrix is symmetric, we only store the upper band.
    Nb = k - 1
    N1, N2 = length.(Bs)
    bands = if symmetry
        (0, Nb)
    else
        # If the bases have different number of constraints (BCs) on the left
        # boundary (⇔ δl ≠ 0), then the bands are shifted.
        δl, δr = num_constraints(B2) .- num_constraints(B1)
        @assert N1 == N2 + δl + δr
        (Nb + δl, Nb - δl)
    end
    M(undef, (N1, N2), bands)
end

"""
    galerkin_matrix!(A::AbstractMatrix, B::AbstractBSplineBasis,
                     deriv = (Derivative(0), Derivative(0)))

Fill preallocated Galerkin matrix.

The matrix may be a `Hermitian` view, in which case only one half of the matrix
will be filled. Note that, for the matrix to be symmetric, both derivative orders
in `deriv` must be the same.

More generally, it is possible to combine different functional bases by passing
a tuple of `AbstractBSplineBasis` as `B`.

See [`galerkin_matrix`](@ref) for details.
"""
function galerkin_matrix! end

galerkin_matrix!(M, B::AbstractBSplineBasis, args...) =
    galerkin_matrix!(M, (B, B), args...)

function galerkin_matrix!(
        S::AbstractMatrix, Bs::Tuple{Vararg{<:AbstractBSplineBasis,2}},
        deriv = Derivative.((0, 0)),
    )
    _check_bases(Bs)
    B1, B2 = Bs
    same_ij = B1 == B2 && deriv[1] == deriv[2]

    if size(S) != length.(Bs)
        throw(DimensionMismatch("wrong dimensions of Galerkin matrix"))
    end

    # Orders and knots are assumed to be the same (see _check_bases).
    k = order(B1)
    ts = knots(B1)
    @assert k == order(B2) && ts === knots(B2)

    # Quadrature information (nodes, weights).
    quadx, quadw = _quadrature_prod(Val(2k - 2))
    @assert length(quadx) == k  # we need k quadrature points per knot segment

    A = if S isa Hermitian
        deriv[1] === deriv[2] ||
            throw(ArgumentError("matrix will not be symmetric with deriv = $deriv"))
        B1 === B2 ||
            throw(ArgumentError("matrix will not be symmetric if bases are different"))
        fill_upper = S.uplo === 'U'
        fill_lower = S.uplo === 'L'
        parent(S)
    else
        fill_upper = true
        fill_lower = true
        S
    end

    fill!(S, 0)
    nlast = last(eachindex(ts))

    # We loop over all knot segments Ω[n] = (ts[n], ts[n + 1]).
    # We integrate all supported B-spline products B1[i] * B2[j] over this
    # segment, adding the result to A[i, j].
    @inbounds for n in eachindex(ts)
        n == nlast && break
        tn, tn1 = ts[n], ts[n + 1]
        tn1 == tn && continue  # interval of length = 0

        metric = QuadratureMetric(tn, tn1)

        # Unnormalise quadrature nodes, such that xs ∈ [tn, tn1]
        xs = metric .* quadx
        # @assert all(x -> tn ≤ x ≤ tn1, xs)

        is = nonzero_in_segment(B1, n)
        js = nonzero_in_segment(B2, n)

        # This is a property of B-spline bases, which should be preserved by
        # derived (recombined) bases.
        @assert 0 < length(is) ≤ k && 0 < length(js) ≤ k

        # Evaluate all required basis functions on quadrature nodes.
        bis = eval_basis_functions(B1, is, xs, deriv[1])
        bjs = same_ij ? bis : eval_basis_functions(B2, js, xs, deriv[2])

        for (nj, j) in enumerate(js), (ni, i) in enumerate(is)
            if !fill_upper && i < j
                continue
            elseif !fill_lower && i > j
                continue
            end
            A[i, j] += metric.α * ((bis[ni] .* bjs[nj]) ⋅ quadw)
        end
    end

    S
end
