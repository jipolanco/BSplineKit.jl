using BSplineKit
using Test

# TODO
# - test recombined basis (test function should satisfy BCs...)

# We make sure that the function is specialised on `f`, to fully avoid allocations.
# See https://docs.julialang.org/en/v1/manual/performance-tips/#Be-aware-of-when-Julia-avoids-specializing
function check_approximate_in_place!(f::F, fapprox) where {F}
    S = copy(spline(fapprox))  # assumed to already approximate `f`
    # @test 0 == @allocated approximate!(f, fapprox)  # no allocations / fails when using SnoopPrecompile?
    @test spline(fapprox) == S  # check that nothing changed
    nothing
end

function test_approximation(B)
    k = order(B)

    ftest = sin

    f_fast = @inferred approximate(ftest, B, VariationDiminishing())
    @test startswith(repr(f_fast), "SplineApproximation containing")
    check_approximate_in_place!(ftest, f_fast)

    # With these two methods, the approximation should exactly match a
    # polynomial of degree k - 1.
    f_interp = @inferred approximate(ftest, B, ApproxByInterpolation(B))
    @test startswith(repr(f_interp), "SplineApproximation containing")
    check_approximate_in_place!(ftest, f_interp)

    f_opt = @inferred approximate(ftest, B, MinimiseL2Error())
    @test startswith(repr(f_opt), "SplineApproximation containing")
    check_approximate_in_place!(ftest, f_opt)

    # Polynomial of degree k - 1.
    # The interpolation and minimisation methods should exactly approximate such
    # a polynomial.
    @testset "Polynomial" begin
        poly_coefs = ntuple(d -> (-d)^d, Val(k))
        fpoly(x) = evalpoly(x, poly_coefs)

        xfine = range(boundaries(B)...; length = 3 * length(B))

        approximate!(fpoly, f_interp)
        @test fpoly.(xfine) ≈ f_interp.(xfine)

        approximate!(fpoly, f_opt)
        @test fpoly.(xfine) ≈ f_opt.(xfine)
    end

    nothing
end

@testset "Approximation" begin
    N = 16
    breaks = [-cos(π * n / N) for n = 0:N]
    @testset "Order $k" for k = 4:5
        B = BSplineBasis(BSplineOrder(k), copy(breaks))
        # R = RecombinedBSplineBasis(B, Derivative(1))  # Neumann BCs
        test_approximation(B)
    end
end
