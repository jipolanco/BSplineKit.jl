using BSplineKit
using Test
using Random

@testset "Extrapolation" begin
    xdata = -1:0.2:1
    ydata = randn(length(xdata))
    itp = interpolate(xdata, ydata, BSplineOrder(4))

    @testset "Flat" begin
        ext = @inferred extrapolate(itp, Flat())
        @test startswith(repr(ext), "SplineExtrapolation containing the")
        @test itp(0.42) == @inferred ext(0.42)
        @test itp(-1.0) == @inferred ext(-2.3)
        @test itp(1.0) == @inferred ext(2.4)
    end

    @testset "Smooth" begin
        itp_nat = interpolate(xdata, ydata, BSplineOrder(4), Natural())
        ext = @inferred extrapolate(itp_nat, Smooth())
        @test itp_nat(0.42) == @inferred ext(0.42)
        @test itp_nat(-1.0) == @inferred ext(-1.0)
        @test itp_nat(1.0) == @inferred ext(1.0)

        # When order = 4 (cubic splines), the smooth extrapolation ensures that
        # the first and second derivatives is continuous at the boundaries.
        # Moreover, with natural BCs, the second derivative is zero, so that
        # the slope is locally linear.
        @test ext(-1.01) - ext(-1.0) ≈ ext(-1.0) - ext(-0.99)
        @test ext(1.01) - ext(1.0) ≈ ext(1.0) - ext(0.99)
    end

    @testset "Linear" begin
        ext = @inferred extrapolate(itp, Linear())
        @test itp(0.42) == @inferred ext(0.42)
        @test itp(-1.0) == @inferred ext(-1.0)
        @test itp(1.0) == @inferred ext(1.0)

        S′ = Derivative() * itp
        @test ext(-1.1) ≈ itp(-1.0) - 0.1 * S′(-1.0)
        @test ext(+1.1) ≈ itp(+1.0) + 0.1 * S′(+1.0)
    end
end
