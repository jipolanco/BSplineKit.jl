function galerkin_matrix(
        B::BSplineBasis,
        ::Type{M} = BandedMatrix{Float64}
    ) where {M <: AbstractMatrix}
    N = length(B)
    A = allocate_galerkin_matrix(M, N, order(B))
    galerkin_matrix!(A, B)
end

allocate_galerkin_matrix(::Type{M}, N, k) where {M <: AbstractMatrix} =
    Symmetric(M(undef, N, N))

function allocate_galerkin_matrix(::Type{M}, N, k) where {M <: BandedMatrix}
    # The upper/lower bandwidths are:
    # - for even k: Nb = k / 2       (total = k + 1 bands)
    # - for odd  k: Nb = (k + 1) / 2 (total = k + 2 bands)
    # Note that the matrix is also symmetric, so we only need the upper band.
    Nb = (k + 1) >> 1
    A = M(undef, (N, N), (0, Nb))
    Symmetric(A)
end

function galerkin_matrix!(S::Symmetric, B::BSplineBasis)
    N = size(S, 1)

    if N != length(B)
        throw(ArgumentError("wrong dimensions of Galerkin matrix"))
    end

    fill!(S, 0)

    # The matrix is symmetric, so we fill only the upper part.
    # For now we assume that S uses the upper part of its parent.
    @assert S.uplo === 'U'
    A = parent(S)

    k = order(B)
    t = knots(B)
    h = (k + 1) ÷ 2  # k/2 if k is even
    T = eltype(S)

    # Quadrature information (weights, nodes).
    quad = _quadrature_prod(k)

    # Upper part: j >= i
    for j = 1:N
        # We're only visiting the elements that have non-zero values.
        # In other words, we know that S[i, j] = 0 outside the chosen interval.
        istart = clamp(j - h, 1, N)
        bj = BSpline(B, j)
        tj = j:(j + k)  # support of b_j (knot indices)
        for i = istart:j
            ti = i:(i + k)  # support of b_i
            t_inds = intersect(ti, tj)  # common support of b_i and b_j
            @assert !isempty(t_inds)    # there is a common support (the B-splines see each other)
            @assert length(t_inds) == k + 1 - (j - i)
            bi = BSpline(B, i)
            A[i, j] = _integrate_prod(bi, bj, t, t_inds, quad)
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
