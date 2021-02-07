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
@inline function _integrate(f::Function, t, inds, (x, w))
    int = 0.0  # compute stuff in Float64, regardless of type wanted by the caller
    N = length(w)  # number of weights / nodes
    @inbounds for i in inds[2:end]
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
