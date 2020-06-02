const DerivativeCombination{N} = Tuple{Vararg{Derivative,N}}

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
[`Symmetric`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/index.html#LinearAlgebra.Symmetric)
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
Note that the returned matrix will only be symmetric if the two derivative
orders are the same.

## Combining different bases

More generally, it is possible to combine different bases.
For this, instead of the `B` parameter, one must pass a tuple of bases
`(B₁, B₂)`.
The bases must have the same parent B-spline basis.
That is, they must share the same set of B-spline knots and be of equal
polynomial order.

This feature may be combined with a specification of different derivative orders
for both bases.
Note that, if both bases are different, the matrix will not be symmetric, and
will not even be square if the respective basis lengths differ.
"""
function galerkin_matrix(
        Bs::NTuple{2,AnyBSplineBasis},
        deriv::DerivativeCombination{2} = Derivative.((0, 0)),
        ::Type{M} = BandedMatrix{Float64},
    ) where {M <: AbstractMatrix}
    B1, B2 = Bs
    symmetry = deriv[1] === deriv[2] && B1 === B2
    A = allocate_galerkin_matrix(M, Bs, symmetry)

    # Make the matrix symmetric if possible.
    S = symmetry ? Symmetric(A) : A

    galerkin_matrix!(S, Bs, deriv)
end

galerkin_matrix(B::AnyBSplineBasis, args...) = galerkin_matrix((B, B), args...)

galerkin_matrix(B, ::Type{M}) where {M <: AbstractMatrix} =
    galerkin_matrix(B, Derivative.((0, 0)), M)

function _check_bases(Bs::Tuple{Vararg{AnyBSplineBasis}})
    ps = parent.(Bs)
    if any(ps .!== ps[1])
        throw(ArgumentError(
            "all bases be constructed from the same parent B-spline basis"
        ))
    end
    nothing
end

allocate_galerkin_matrix(::Type{M}, Bs, etc...) where {M <: AbstractMatrix} =
    M(undef, length.(Bs)...)

allocate_galerkin_matrix(::Type{SparseMatrixCSC{T}}, Bs, etc...) where {T} =
    spzeros(T, length.(Bs)...)

function allocate_galerkin_matrix(
        ::Type{M}, Bs, symmetry) where {M <: BandedMatrix}
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
        # If the bases have different lengths (⇔ δ ≠ 0), then the bands are
        # shifted.
        δ = num_constraints(B2) - num_constraints(B1)
        @assert N1 == N2 + 2δ
        (Nb + δ, Nb - δ)
    end
    M(undef, (N1, N2), bands)
end

"""
    galerkin_tensor(
        B::AbstractBSplineBasis,
        (D1::Derivative, D2::Derivative, D3::Derivative),
        [T = Float64],
    )

Compute 3D banded tensor appearing from quadratic terms in Galerkin method.

As with [`galerkin_matrix`](@ref), it is also possible to combine different
functional bases by passing, instead of `B`, a tuple `(B₁, B₂, B₃)` of three
`AbstractBSplineBasis`.
For now, the first two bases, `B₁` and `B₂`, must have the same length.

The tensor is efficiently stored in a [`BandedTensor3D`](@ref) object.
"""
function galerkin_tensor end

function galerkin_tensor(
        Bs::NTuple{3,AbstractBSplineBasis},
        deriv::DerivativeCombination{3},
        ::Type{T} = Float64,
    ) where {T}
    _check_bases(Bs)
    dims = length.(Bs)
    b = order(first(Bs)) - 1   # band width
    if length(Bs[1]) != length(Bs[2])
        throw(ArgumentError("the first two bases must have the same lengths"))
    end
    δ = num_constraints(Bs[3]) - num_constraints(Bs[1])
    A = BandedTensor3D{T}(undef, dims, b, bandshift=(0, 0, δ))
    galerkin_tensor!(A, Bs, deriv)
end

galerkin_tensor(B::AbstractBSplineBasis, args...) =
    galerkin_tensor((B, B, B), args...)

"""
    galerkin_matrix!(A::AbstractMatrix, B::AbstractBSplineBasis,
                     deriv = (Derivative(0), Derivative(0)))

Fill preallocated Galerkin matrix.

The matrix may be a `Symmetric` view, in which case only one half of the matrix
will be filled. Note that, for the matrix to be symmetric, both derivative orders
in `deriv` must be the same.

More generally, it is possible to combine different functional bases by passing
a tuple of `AbstractBSplineBasis` as `B`.

See [`galerkin_matrix`](@ref) for details.
"""
function galerkin_matrix! end

galerkin_matrix!(M, B::AnyBSplineBasis, args...) =
    galerkin_matrix!(M, (B, B), args...)

function galerkin_matrix!(S::AbstractMatrix, Bs::NTuple{2,AbstractBSplineBasis},
                          deriv = Derivative.((0, 0)))
    _check_bases(Bs)
    N, M = size(S)
    B1, B2 = Bs

    if (N, M) != length.(Bs)
        throw(ArgumentError("wrong dimensions of Galerkin matrix"))
    end

    fill!(S, 0)

    # Orders and knots are assumed to be the same (see _check_bases).
    k = order(B1)
    t = knots(B1)
    @assert k == order(B2) && t === knots(B2)
    h = k - 1
    T = eltype(S)

    # Quadrature information (weights, nodes).
    quad = _quadrature_prod(2k - 2)
    @assert length(quad[1]) == k  # this should correspond to k quadrature points

    if S isa Symmetric
        deriv[1] === deriv[2] || error("matrix will not be symmetric with deriv = $deriv")
        B1 === B2 || error("matrix will not be symmetric if bases are different")
        fill_upper = S.uplo === 'U'
        fill_lower = S.uplo === 'L'
        A = parent(S)
    else
        fill_upper = true
        fill_lower = true
        A = S
    end

    # Number of BCs on each boundary
    δ = num_constraints(B2) - num_constraints(B1)
    @assert M + 2δ == N

    for j = 1:M
        i0 = j + δ
        tj = support(B2, j)
        fj = x -> evaluate_bspline(B2, j, x, deriv[2])
        istart = fill_upper ? 1 : i0
        iend = fill_lower ? N : i0
        for i = istart:iend
            ti = support(B1, i)
            t_inds = intersect(ti, tj)  # common support of b_i and b_j
            isempty(t_inds) && continue
            fi = x -> evaluate_bspline(B1, i, x, deriv[1])
            f = x -> fi(x) * fj(x)
            A[i, j] = _integrate(f, t, t_inds, quad)
        end
    end

    S
end

# Alternative using the BSplineBasis type defined in the BSplines package.
# TODO merge this with other variant!
function galerkin_matrix!(
        S::AbstractMatrix, Bs::NTuple{2,BSplines.BSplineBasis},
        deriv = Derivative.((0, 0)),
    )
    if Bs[1] !== Bs[2]
        throw(ArgumentError(
            "for now, this variant of galerkin_matrix! doesn't support combinations of different B-spline bases"
        ))
    end
    B = first(Bs)
    N = size(S, 1)

    if N != length(B)
        throw(ArgumentError("wrong dimensions of Galerkin matrix"))
    end

    fill!(S, 0)

    k = order(B)
    t = knots(B)
    T = eltype(S)

    # Quadrature information (weights, nodes).
    quad = _quadrature_prod(2k - 2)
    @assert length(quad[1]) == k

    same_deriv = deriv[1] === deriv[2]

    if S isa Symmetric
        same_deriv || error("matrix will not be symmetric with deriv = $deriv")
        fill_upper = S.uplo === 'U'
        fill_lower = S.uplo === 'L'
        A = parent(S)
    else
        fill_upper = true
        fill_lower = true
        A = S
    end

    # B-splines evaluated at a given point.
    bi = zeros(T, k)
    bj = copy(bi)

    # Integrate over each segment between knots.
    for l = k:N
        a, b = t[l], t[l + 1]  # segment (t[l], t[l + 1])
        α = (b - a) / 2
        β = (a + b) / 2

        x, w = quad
        for n in eachindex(w)
            y = α * x[n] + β

            # Evaluate B-splines at quadrature point.
            # This evaluates B-splines (i, j) ∈ [(off + 1):(off + k)].
            off = BSplines.bsplines!(bi, B, y, deriv[1], leftknot=l) :: Int
            if same_deriv
                copy!(bj, bi)
            else
                δ = BSplines.bsplines!(bj, B, y, deriv[2], leftknot=l) :: Int
                @assert δ === off
            end

            @assert off === l - k

            for j = 1:k
                istart = fill_upper ? 1 : j
                iend = fill_lower ? k : j
                for i = istart:iend
                    A[off + i, off + j] += α * w[n] * bi[i] * bj[j]
                end
            end
        end
    end

    S
end

function galerkin_tensor!(A::BandedTensor3D,
                          Bs::NTuple{3,AbstractBSplineBasis},
                          deriv::DerivativeCombination{3})
    _check_bases(Bs)

    Ns = size(A)
    if any(Ns .!= length.(Bs))
        throw(ArgumentError("wrong dimensions of Galerkin tensor"))
    end

    Ni, Nj, Nl = Ns
    @assert Ni == Nj  # verified earlier...

    Bi, Bj, Bl = Bs

    k = order(Bi)  # same for all bases (see _check_bases)
    t = knots(Bi)
    h = k - 1
    T = eltype(A)

    # Quadrature information (weights, nodes).
    quad = _quadrature_prod(3k - 3)

    Al = Matrix{T}(undef, 2k - 1, 2k - 1)
    δ = num_constraints(Bl) - num_constraints(Bi)
    @assert Ni == Nj == Nl + 2δ

    if bandwidth(A) != k - 1
        throw(ArgumentError("BandedTensor3D must have bandwidth = $(k - 1)"))
    end
    if bandshift(A) != (0, 0, δ)
        throw(ArgumentError("BandedTensor3D must have bandshift = (0, 0, $δ)"))
    end

    for l = 1:Nl
        # TODO
        # - verify this for non-cubic tensors
        # - add tests!
        ll = l + δ
        istart = clamp(ll - h, 1, Ni)
        iend = clamp(ll + h, 1, Nj)
        is = istart:iend
        js = is

        band_ind = BandedTensors.band_indices(A, l)
        @assert issubset(is, band_ind) && issubset(js, band_ind)

        i0 = first(band_ind) - 1
        j0 = i0
        @assert i0 == ll - k

        fill!(Al, 0)

        tl = support(Bl, l)
        fl = x -> evaluate_bspline(Bl, l, x, deriv[3])

        for j in js
            tj = support(Bj, j)
            fj = x -> evaluate_bspline(Bj, j, x, deriv[2])
            for i in is
                ti = support(Bi, i)
                fi = x -> evaluate_bspline(Bi, i, x, deriv[1])
                t_inds = intersect(ti, tj, tl)
                isempty(t_inds) && continue
                f = x -> fi(x) * fj(x) * fl(x)
                Al[i - i0, j - j0] = _integrate(f, t, t_inds, quad)
            end
        end

        A[:, :, l] = Al
    end

    A
end

# Generate quadrature information for B-spline product.
# Returns weights and nodes for integration in [-1, 1].
#
# See https://en.wikipedia.org/wiki/Gaussian_quadrature.
#
# Some notes:
#
# - On each interval between two neighbouring knots, each B-spline is a
#   polynomial of degree (k - 1). Hence, the product of two B-splines has degree
#   (2k - 2).
#
# - The Gauss--Legendre quadrature rule is exact for polynomials of degree
#   <= (2n - 1), where n is the number of nodes and weights.
#
# - Conclusion: on each knot interval, `k` nodes should be enough to get an
#   exact integral. (I verified that results don't change when using more than
#   `k` nodes.)
#
# Here, p is the polynomial order (p = 2k - 2 for the product of two B-splines).
_quadrature_prod(p) = gausslegendre(cld(p + 1, 2))

# Integrate function over the subintervals t[inds].
function _integrate(f::Function, t, inds, (x, w))
    int = 0.0  # compute stuff in Float64, regardless of type wanted by the caller
    N = length(w)  # number of weights / nodes
    for i in inds[2:end]
        # Integrate in [t[i - 1], t[i]].
        # See https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
        a, b = t[i - 1], t[i]
        α = (b - a) / 2
        β = (a + b) / 2
        for n = 1:N
            y = α * x[n] + β
            int += α * w[n] * f(y)
        end
    end
    int
end
