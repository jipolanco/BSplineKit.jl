const DerivativeCombination{N} = Tuple{Vararg{Derivative,N}}

"""
    galerkin_matrix(
        B::AbstractBSplineBasis,
        [deriv = (Derivative(0), Derivative(0))],
        [MatrixType = BandedMatrix{Float64}]
    )

Compute Galerkin mass or stiffness matrix.

# Extended help

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

Note that the Galerkin matrix is banded with `2k - 1` bands.
Moreover, if both derivative orders are the same, the matrix is
symmetric and positive definite, and only `k` bands are needed to fully describe
the matrix.
In those cases, a
[`Symmetric`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/index.html#LinearAlgebra.Symmetric)
view of an underlying matrix is returned.

By default, the underlying matrix holding the data is a `BandedMatrix` that
defines the upper part of the symmetric matrix.
Other types of container are also supported, including regular sparse matrices
(`SparseMatrixCSC`) and dense arrays (`Matrix`).
"""
function galerkin_matrix(
        B::AnyBSplineBasis,
        deriv::DerivativeCombination{2} = Derivative.((0, 0)),
        ::Type{M} = BandedMatrix{Float64},
    ) where {M <: AbstractMatrix}
    N = length(B)
    symmetry = deriv[1] === deriv[2]

    A = allocate_galerkin_matrix(M, N, order(B), symmetry)

    # Make the matrix symmetric if possible.
    S = symmetry ? Symmetric(A) : A

    galerkin_matrix!(S, B, deriv)
end

galerkin_matrix(B::AnyBSplineBasis, ::Type{M}) where {M <: AbstractMatrix} =
    galerkin_matrix(B, Derivative.((0, 0)), M)

allocate_galerkin_matrix(::Type{M}, N, etc...) where {M <: AbstractMatrix} =
    M(undef, N, N)

allocate_galerkin_matrix(::Type{SparseMatrixCSC{T}}, N, etc...) where {T} =
    spzeros(T, N, N)

function allocate_galerkin_matrix(::Type{M}, N, k,
                                  symmetry) where {M <: BandedMatrix}
    # The upper/lower bandwidths are Nb = k - 1 (total = 2k - 1 bands).
    # Note that if the matrix is also symmetric, then we only need the upper
    # band.
    Nb = k - 1
    bands = symmetry ? (0, Nb) : (Nb, Nb)
    M(undef, (N, N), bands)
end

"""
    galerkin_tensor(
        B::AbstractBSplineBasis,
        (D1::Derivative, D2::Derivative, D3::Derivative),
        [T = Float64],
    )

Compute 3D banded tensor appearing from quadratic terms in Galerkin method.

The tensor is efficiently stored [`BandedTensor3D`](@ref) object.
"""
function galerkin_tensor(
        B::AnyBSplineBasis,
        deriv::DerivativeCombination{3},
        ::Type{T} = Float64,
    ) where {T}
    N = length(B)
    b = order(B) - 1   # band width
    A = BandedTensor3D{T}(undef, N, b)
    galerkin_tensor!(A, B, deriv)
end

"""
    galerkin_matrix!(A::AbstractMatrix, B::AbstractBSplineBasis,
                     deriv = (Derivative(0), Derivative(0)))

Fill preallocated Galerkin matrix.

The matrix may be a `Symmetric` view, in which case only one half of the matrix
will be filled. Note that, for the matrix to be symmetric, both derivative orders
in `deriv` must be the same.

See [`galerkin_matrix`](@ref) for details.
"""
function galerkin_matrix!(S::AbstractMatrix, B::AbstractBSplineBasis,
                          deriv = Derivative.((0, 0)))
    N = size(S, 1)

    if N != length(B)
        throw(ArgumentError("wrong dimensions of Galerkin matrix"))
    end

    fill!(S, 0)

    k = order(B)
    t = knots(B)
    h = k - 1
    T = eltype(S)

    # Quadrature information (weights, nodes).
    quad = _quadrature_prod(2k - 2)
    @assert length(quad[1]) == k  # this should correspond to k quadrature points

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
        tj = support(B, j)
        fj = x -> evaluate_bspline(B, j, x, deriv[2])
        for i = istart:iend
            ti = support(B, i)
            fi = x -> evaluate_bspline(B, i, x, deriv[1])
            t_inds = intersect(ti, tj)  # common support of b_i and b_j
            @assert !isempty(t_inds)    # there is a common support (the B-splines see each other)
            @assert length(t_inds) == k + 1 - abs(j - i)
            f = x -> fi(x) * fj(x)
            A[i, j] = _integrate(f, t, t_inds, quad)
        end
    end

    S
end

# TODO merge this with other variant!
function galerkin_matrix!(S::AbstractMatrix, B::BSplines.BSplineBasis,
                          deriv = Derivative.((0, 0)))
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

function galerkin_tensor!(A::BandedTensor3D, B::AbstractBSplineBasis,
                          deriv::DerivativeCombination{3})
    N = size(A, 1)

    if N != length(B)
        throw(ArgumentError("wrong dimensions of Galerkin tensor"))
    end

    k = order(B)
    t = knots(B)
    h = k - 1
    T = eltype(A)

    # Quadrature information (weights, nodes).
    quad = _quadrature_prod(3k - 3)
    @assert BandedTensors.bandwidth(A) == k - 1

    Al = Matrix{T}(undef, 2k - 1, 2k - 1)

    for l = 1:N
        istart = clamp(l - h, 1, N)
        iend = clamp(l + h, 1, N)
        is = istart:iend
        js = is

        band_ind = BandedTensors.band_indices(A, l)
        @assert issubset(is, band_ind) && issubset(js, band_ind)

        i0 = first(band_ind) - 1
        j0 = i0
        @assert i0 == l - k

        fill!(Al, 0)

        tl = support(B, l)
        fl = x -> evaluate_bspline(B, l, x, deriv[3])

        for j in js
            tj = support(B, j)
            fj = x -> evaluate_bspline(B, j, x, deriv[2])
            for i in is
                ti = support(B, i)
                fi = x -> evaluate_bspline(B, i, x, deriv[1])
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
