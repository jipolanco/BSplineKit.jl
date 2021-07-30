using BSplineKit
using Test

function test_approximation(B)
    k = order(B)

    fpoly = let poly_coefs = ntuple(d -> (-d)^d, Val(k))
        x -> @eval_poly(x, poly_coefs...)
    end

    f_fast = approximate(fpoly, B, VariationDiminishing())

    # @test repr(f_fast)

    # With these two methods, the approximation should exactly match a
    # polynomial of degree k - 1.
    f_interp = approximate(fpoly, B, ApproxByInterpolation(B))

    # TODO test this method...
    # f_opt = approximate(fpoly, B, MinimiseL2Error(B))

    # xfine = range(first(x), last(x); length = 2 * length(x))
    # @test fapprox.(xfine) ≈ fpoly.(xfine)
end

# TODO
# - test a couple of orders
# - test the 3 approximation methods
# - test in-place approximation
# - test show(::SplineApproximation)
# - test recombined basis (test function should satisfy BCs...)
@testset "Approximation" begin
    N = 16
    breaks = [-cos(π * n / N) for n = 0:N]
    @testset "Order $k" for k = 4:5
        B = BSplineBasis(BSplineOrder(k), copy(breaks))
        # R = RecombinedBSplineBasis(B, Derivative(1))  # Neumann BCs
        test_approximation(B)
    end
end
