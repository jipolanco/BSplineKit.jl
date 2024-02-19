# Test generalised natural splines (which are the "standard" natural cubic
# splines when k = 4).

using BSplineKit
using Random
using Test

import LinearAlgebra

function test_natural(::Type{T}, ord::BSplineOrder) where {T}
    ts = range(T(-1), T(1); length = 21)
    B = @inferred BSplineBasis(ord, ts)
    k = order(B)
    @test B isa AbstractBSplineBasis{k, T}

    if isodd(k)
        # `Natural` boundary condition only supported for even-order splines (got k = $k)
        @test_throws ArgumentError RecombinedBSplineBasis(B, Natural())
        return
    end

    rng = MersenneTwister(42)

    R = @inferred RecombinedBSplineBasis(B, Natural())
    @test R isa AbstractBSplineBasis{k, T}
    S = Spline(R, 1 .+ rand(rng, T, length(R)))  # coefficients in [1, 2]

    M = recombination_matrix(R)
    @test M isa AbstractMatrix{T}

    if k == 2
        # In this case, basis functions are not really recombined, and the
        # resulting basis is identical to the original one.
        @test M == LinearAlgebra.I
    end

    for x ∈ boundaries(R)
        Sx = @inferred S(x)
        @test Sx > 1
        rtol = if T === Float64
            2e-8
        elseif T === Float32
            2f-4
        end
        ϵ = rtol * Sx  # threshold
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
        test_natural(Float64, BSplineOrder(k))
    end
    @testset "Float32" begin
        # Note: results are very wrong with high-order splines (k ≥ 6) when using Float32.
        # The zero derivative condition is not well verified.
        test_natural(Float32, BSplineOrder(2))
        test_natural(Float32, BSplineOrder(4))
    end
end
