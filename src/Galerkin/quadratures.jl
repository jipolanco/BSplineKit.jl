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
    gausslegendre(Val(n))
end

function default_quadrature(Bs::Tuple{Vararg{AbstractBSplineBasis}})
    # Polynomial order of integrand (the value is usually inferred).
    polynomial_order = sum(B -> order(B) - 1, Bs)
    _quadrature_prod(Val(polynomial_order))
end

# Compile-time computation of Gauss--Legendre nodes and weights.
# If the @generated branch is taken, then the computation is done at compile
# time, making sure that no runtime allocations are performed.
function gausslegendre(::Val{n}) where {n}
    if @generated
        vecs = _gausslegendre_impl(Val(n))
        :( $vecs )
    else
        _gausslegendre_impl(Val(n))
    end
end

function _gausslegendre_impl(::Val{n}) where {n}
    data = FastGaussQuadrature.gausslegendre(n)
    map(SVector{n}, data)
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
