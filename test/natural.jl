# Test generalised natural splines (which are the "standard" natural cubic
# splines when k = 4).

using BSplineKit
using Random
using Test

import LinearAlgebra

function test_natural(ord::BSplineOrder)
    B = BSplineBasis(ord, -1:0.1:1)
    k = order(B)
    if isodd(k)
        # `Natural` boundary condition only supported for even-order splines (got k = $k)
        @test_throws ArgumentError RecombinedBSplineBasis(B, Natural())
        return
    end

    rng = MersenneTwister(42)

    R = @inferred RecombinedBSplineBasis(B, Natural())
    S = Spline(R, 1 .+ rand(rng, length(R)))  # coefficients in [1, 2]

    M = recombination_matrix(R)
    if k == 2
        # In this case, basis functions are not really recombined, and the
        # resulting basis is identical to the original one.
        @test M == LinearAlgebra.I
    end

    for x ∈ boundaries(R)
        Sx = @inferred S(x)
        @test Sx > 1
        ϵ = 2e-8 * Sx  # threshold
        @test abs((Derivative(1) * S)(x)) > ϵ  # different from zero
        for n = 2:(k ÷ 2)
            Sder = Derivative(n) * S
            @test abs(Sder(x)) < ϵ  # approximately zero
        end
    end

    nothing
end

# Note that the case k = 2 is also supported, even if it's not really useful
# (in this case the recombination is equivalent to the original B-spline basis).
@testset "Natural splines" begin
    @testset "k = $k" for k ∈ (2, 4, 5, 6, 8)
        test_natural(BSplineOrder(k))
    end
end
