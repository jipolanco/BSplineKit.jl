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
function _quadrature_prod(::Val{p}) where {p}
    n = cld(p + 1, 2)
    _gausslegendre(Val(n))
end

# Precomputed Gauss--Legendre nodes and weights, for low numbers of nodes.
# Copied from FastGaussQuadrature.jl, with the difference that the
# implementation below directly returns SVector's (avoiding small allocations).
_gausslegendre(::Val{1}) = (
    SVector(0.0),
    SVector(2.0),
)

_gausslegendre(::Val{2}) = (
    SVector(-1 / sqrt(3), 1 / sqrt(3)),
    SVector(1.0, 1.0),
)

_gausslegendre(::Val{3}) = (
    SVector(-sqrt(3 / 5), 0.0, sqrt(3 / 5)),
    SVector(5 / 9, 8 / 9, 5 / 9),
)

function _gausslegendre(::Val{4})
    a = 2 / 7 * sqrt(6 / 5)
    (
        SVector(-sqrt(3 / 7 + a), -sqrt(3/7-a), sqrt(3/7-a), sqrt(3/7+a)),
        SVector((18 - sqrt(30)) / 36, (18 + sqrt(30)) / 36,
                (18 + sqrt(30)) / 36, (18 - sqrt(30)) / 36),
    )
end

function _gausslegendre(::Val{5})
    b = 2 * sqrt(10 / 7)
    (
        SVector(-sqrt(5 + b) / 3, -sqrt(5 - b) / 3, 0.0,
                 sqrt(5 - b) / 3,  sqrt(5 + b) / 3),
        SVector((322 - 13 * sqrt(70)) / 900, (322 + 13 * sqrt(70)) / 900, 128 / 225,
                (322 + 13 * sqrt(70)) / 900, (322 - 13 * sqrt(70)) / 900),
    )
end

# Fallback: call `gausslegendre` from FastGaussQuadrature.jl
function _gausslegendre(::Val{n}) where {n}
    x, w = gausslegendre(n)
    SVector{n}(x), SVector{n}(w)
end

# Metric for integration on interval [a, b].
# This is to transform from integration on the interval [-1, 1], in which
# quadratures are defined.
# See https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
struct QuadratureMetric{T}
    α :: T
    β :: T

    function QuadratureMetric(a::T, b::T) where {T}
        α = (b - a) / 2
        β = (a + b) / 2
        new{T}(α, β)
    end
end

# Apply metric to normalised coordinate (x ∈ [-1, 1]).
Base.:*(M::QuadratureMetric, x::Real) = M.α * x + M.β
Broadcast.broadcastable(M::QuadratureMetric) = Ref(M)

# Evaluate elements of basis `B` (given by indices `is`) at points `xs`.
# The length of `xs` is assumed static.
# The length of `is` is generally equal to the B-spline order, but may me
# smaller near the boundaries (this is not significant if knots are "augmented",
# as is the default).
# TODO implement evaluation of all B-splines at once (should be much faster...)
function eval_basis_functions(B, is, xs, args...)
    N = length(xs)
    k = order(B)
    @assert length(is) ≤ k
    bis = ntuple(Val(k)) do n
        # In general, `is` will have length equal `k`.
        # If its length is less than `k` (may happen near boundaries, if knots
        # are not "augmented"), we repeat values for the first B-spline, just
        # for type stability concerns. These values are never used.
        i = n > length(is) ? is[1] : is[n]
        map(xs) do x
            B[i](x, args...)
        end
    end
end
