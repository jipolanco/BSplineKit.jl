using BSplineKit.DifferentialOps
using LinearAlgebra: ⋅

@testset "DifferentialOps" begin
    D2 = Derivative(2)
    D3 = Derivative(3)
    @test max_order(D2) == 2
    @test max_order(D3) == 3

    @test D2^2 === Derivative(4)
    @test D2^3 === Derivative(6)
    @test Derivative() === Derivative(1)
    @test Derivative()^3 === Derivative(3)

    @inferred (() -> Derivative())()
    @inferred (() -> Derivative()^3)()
    @inferred (() -> Derivative(3))()

    S = D2 + 5 * D3
    S′ = 5 * D3 + D2
    @test S == S′
    @test max_order(S) == max_order(S′) == 3

    @testset "Projections" begin
        @test LeftNormal() ⋅ D2 == D2 == 1 * D2
        @test RightNormal() ⋅ D2 == D2 == 1 * D2

        # Odd-order derivatives are flipped on the left boundary.
        @test LeftNormal() ⋅ D3 == -D3 == -1 * D3
        @test RightNormal() ⋅ D3 == D3 == 1 * D3

        @test LeftNormal() ⋅ S == D2 - 5 * D3
        @test RightNormal() ⋅ S == D2 + 5 * D3
    end
end
