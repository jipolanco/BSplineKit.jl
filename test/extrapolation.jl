using BSplineKit
using Test
using Random

@testset "Extrapolation" begin
    xdata = -1:0.2:1
    ydata = randn(length(xdata))
    itp = interpolate(xdata, ydata, BSplineOrder(4))

    @testset "Flat" begin
        ext = @inferred extrapolate(itp, Flat())
        @test itp(0.42) == @inferred ext(0.42)
        @test itp(-1.0) == @inferred ext(-2.3)
        @test itp(1.0) == @inferred ext(2.4)
    end
end
