using BSplineKit
using Test

@testset "Periodic splines" begin
    ts = 0:0.1:0.9
    L = 1  # period
    B = @inferred PeriodicBSplineBasis(BSplineOrder(4), ts, L)
    N = length(B)

    @test N == length(ts)
    @test @inferred(period(B)) == L
    @test typeof(period(B)) === eltype(ts)
    @test @inferred(boundaries(B)) == (0, 1)
    @test B == B
    let B′ = PeriodicBSplineBasis(BSplineOrder(3), ts, L)
        @test B ≠ B′
    end

    @test startswith(
        repr(B),
        "10-element PeriodicBSplineBasis of order 4, domain [0.0, 1.0), period 1.0"
    )

    # The last knot must be strictly less than `first(ts) + period`
    @test_throws ArgumentError PeriodicBSplineBasis(BSplineOrder(4), ts, last(ts))

    # Order must be ≥ 1
    @test_throws ArgumentError PeriodicBSplineBasis(BSplineOrder(0), ts, L)

    @test PeriodicBSplineBasis(4, ts, L) == B

    @testset "Evaluate x = $x" for x ∈ (0.02, 0.42, 0.89)
        ilast, bs = @inferred B(x)
        @test (ilast, bs) == @inferred B(x; ileft = ilast)
        @test 0 == @allocated B(x)  # no allocations!
        @test all(>(0), bs)
        @test sum(bs) ≈ 1  # partition of unity

        # Check periodicity
        for n ∈ (-2, 1)
            ilast′, bs′ = B(x + n * L)
            @test ilast′ == ilast + n * N
            @test all(bs .≈ bs′)
        end
    end
end
