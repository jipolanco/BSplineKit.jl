using BSplineKit
using StaticArrays

@testset "Static length" begin
    breaks = SVector(0.0, 0.1, 0.4, 0.7, 1.0)
    B = @inferred BSplineBasis(BSplineOrder(4), breaks)
    @test B === @inferred BSplineBasis(BSplineOrder(4), Tuple(breaks))  # test constructor based on tuple
    Bdyn = BSplineBasis(BSplineOrder(4), collect(breaks))  # non-static knots
    @assert knots(Bdyn) isa Vector
    @test static_length(B) === length(B)
    @test static_length(Bdyn) === nothing
    k = order(B)
    ts = knots(B)
    @test ts isa SVector
    @test length(ts) == length(breaks) + 2 * (k - 1)
    @inferred (B -> Val(length(B)))(B)  # length is statically known
    N = @inferred length(B)
    @test length(ts) == N + k
    @test B(0.32) == Bdyn(0.32)

    @testset "Collocation points ($method)" for method ∈ (Collocation.AvgKnots(), Collocation.SameAsKnots())
        xs = @inferred collocation_points(B, method)
        @test xs isa SVector
        @test xs == collocation_points(Bdyn, method)
    end

    @testset "Collocation matrix" begin
        xs = @inferred collocation_points(B)  # uses the default method ensuring a square collocation matrix
    end
end
