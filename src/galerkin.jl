"""
    galerkin_matrix(
        B::BSplineBasis,
        [deriv = (Derivative(0), Derivative(0))],
        [MatrixType = BandedMatrix{Float64}]
    )

Compute Galerkin mass or stiffness matrix.

Definition of mass matrix:

    M[i, j] = ⟨ bᵢ, bⱼ ⟩  for  i = 1:N and j = 1:N,

where `bᵢ` is the i-th B-spline and `N = length(B)` is the number of B-splines
in the basis `B`.
Here, ⟨⋅,⋅⟩ is the [L² inner
product](https://en.wikipedia.org/wiki/Square-integrable_function#Properties)
between functions.

To obtain a matrix associated to the B-spline derivatives, set the `deriv`
argument to the order of the derivatives.
For instance, if `deriv = (Derivative(0), Derivative(2))`, this returns the
matrix `⟨ bᵢ, bⱼ'' ⟩`.

Note that the Galerkin matrix is banded,
with `k + 1` and `k + 2` for `k` even and odd, respectively.
Moreover, if both derivative orders are the same, the matrix is
symmetric and positive definite.
In those cases, a
[`Symmetric`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/index.html#LinearAlgebra.Symmetric)
view of an underlying matrix is returned.

By default, the underlying matrix holding the data is a `BandedMatrix` that
defines the upper part of the symmetric matrix.
Other types of container are also supported, including regular sparse matrices
(`SparseMatrixCSC`) and dense arrays (`Matrix`).
"""
function galerkin_matrix(
        B::BSplineBasis,
        deriv = Derivative.((0, 0)),
        ::Type{M} = BandedMatrix{Float64},
    ) where {M <: AbstractMatrix}
    N = length(B)
    symmetry = deriv[1] === deriv[2]

    A = allocate_galerkin_matrix(M, N, order(B), symmetry)

    # Make the matrix symmetric if possible.
    S = symmetry ? Symmetric(A) : A

    galerkin_matrix!(S, B, deriv)
end

galerkin_matrix(B, ::Type{M}) where {M} =
    galerkin_matrix(B, Derivative.((0, 0)), M)

allocate_galerkin_matrix(::Type{M}, N, etc...) where {M <: AbstractMatrix} =
    M(undef, N, N)

allocate_galerkin_matrix(::Type{SparseMatrixCSC{T}}, N, etc...) where {T} =
    spzeros(T, N, N)

function allocate_galerkin_matrix(::Type{M}, N, k,
                                  symmetry) where {M <: BandedMatrix}
    # The upper/lower bandwidths are:
    # - for even k: Nb = k / 2       (total = k + 1 bands)
    # - for odd  k: Nb = (k + 1) / 2 (total = k + 2 bands)
    # Note that if the matrix is also symmetric, then we only need the upper
    # band.
    Nb = (k + 1) >> 1
    bands = symmetry ? (0, Nb) : (Nb, Nb)
    M(undef, (N, N), bands)
end

"""
    galerkin_matrix!(A::AbstractMatrix, B::BSplineBasis,
                     deriv = (Derivative(0), Derivative(0)))

Fill preallocated Galerkin matrix.

The matrix may be a `Symmetric` view, in which case only one half of the matrix
will be filled. Note that, for the matrix to be symmetric, both derivative orders
in `deriv` must be the same.

See also [`galerkin_matrix`](@ref).
"""
function galerkin_matrix!(S::AbstractMatrix, B::BSplineBasis,
                          deriv = Derivative.((0, 0)))
    N = size(S, 1)

    if N != length(B)
        throw(ArgumentError("wrong dimensions of Galerkin matrix"))
    end

    fill!(S, 0)

    k = order(B)
    t = knots(B)
    h = (k + 1) ÷ 2  # k/2 if k is even
    T = eltype(S)

    # Quadrature information (weights, nodes).
    quad = _quadrature_prod(k)

    if S isa Symmetric
        deriv[1] === deriv[2] || error("matrix will not be symmetric with deriv = $deriv")
        fill_upper = S.uplo === 'U'
        fill_lower = S.uplo === 'L'
        A = parent(S)
    else
        fill_upper = true
        fill_lower = true
        A = S
    end

    for j = 1:N
        # We're only visiting the elements that have non-zero values.
        # In other words, we know that S[i, j] = 0 outside the chosen interval.
        istart = fill_upper ? clamp(j - h, 1, N) : j
        iend = fill_lower ? clamp(j + h, 1, N) : j
        bj = BSpline(B, j)
        tj = support(bj)
        fj = x -> bj(x, deriv[2])
        for i = istart:iend
            bi = BSpline(B, i)
            ti = support(bi)
            fi = x -> bi(x, deriv[1])
            t_inds = intersect(ti, tj)  # common support of b_i and b_j
            @assert !isempty(t_inds)    # there is a common support (the B-splines see each other)
            @assert length(t_inds) == k + 1 - abs(j - i)
            A[i, j] = _integrate_prod(fi, fj, t, t_inds, quad)
        end
    end

    S
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
_quadrature_prod(k) = gausslegendre(k)

# Integrate product of functions over the subintervals t[inds].
function _integrate_prod(f, g, t, inds, (x, w))
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
            int += α * w[n] * f(y) * g(y)
        end
    end
    int
end
