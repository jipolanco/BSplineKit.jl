using BSplineKit: BSplineOrder
using BSplineKit.BSplines: multiplicity

function test_splines(B::BSplineBasis, knots_in)
    k = order(B)
    t = knots(B)

    @testset "Knots (k = $k)" begin
        let (ka, kb) = multiplicity.(Ref(t), (1, length(t)))
            @test ka == kb == k
        end

        @test @views all(t[1:k] .== knots_in[1])
        @test @views all(t[(end - k + 1):end] .== knots_in[end])
        @test @views t[(k + 1):(end - k)] == knots_in[2:(end - 1)]
    end

    @testset "B-splines (k = $k)" begin
        N = length(B)
        @test_throws DomainError evaluate(B, 0, 0.2)
        @test_throws DomainError evaluate(B, N + 1, 0.2)

        # Verify values at the boundaries.
        @test evaluate(B, 1, t[1]) == 1.0
        @test evaluate(B, N, t[end]) == 1.0
    end

    xcol = collocation_points(B, method=Collocation.AvgKnots())
    @test xcol[1] == knots_in[1]
    @test xcol[end] == knots_in[end]

    C = collocation_matrix(B, xcol)

    @testset "Spline (k = $k)" begin
        # Generate data at collocation points and get B-spline coefficients.
        ucol = cos.(xcol)
        coefs = C \ ucol

        @inferred Spline(B, coefs)
        S = Spline(B, coefs)
        @test all(S.(xcol) .≈ ucol)
    end

    nothing
end

function test_splines(::BSplineOrder{k}) where {k}
    knots_in = gauss_lobatto_points(10 + k)

    @inferred BSplineBasis(BSplineOrder(k), knots_in)
    @inferred (() -> BSplineBasis(k, knots_in))()

    g = BSplineBasis(k, knots_in)
    @test order(g) == k
    test_splines(g, knots_in)

    nothing
end

@testset "Splines" begin
    test_splines(BSplineOrder(4))
    test_splines(BSplineOrder(5))
end
