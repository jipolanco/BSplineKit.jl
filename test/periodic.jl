using BSplineKit
using Test

@testset "Periodic splines" begin
    breaks = 0:0.1:0.9
    L = 1  # period
    k = 4
    B = @inferred PeriodicBSplineBasis(BSplineOrder(k), breaks, L)
    N = length(B)

    ts = @inferred knots(B)
    @test length(ts) == N + k  # consistency with the regular BSplineBasis

    @test N == length(breaks)
    @test @inferred(period(B)) == L
    @test typeof(period(B)) === eltype(breaks)
    @test @inferred(boundaries(B)) == (0, 1)
    @test B == B
    let B′ = PeriodicBSplineBasis(BSplineOrder(3), breaks, L)
        @test B ≠ B′
    end

    @test startswith(
        repr(B),
        "10-element PeriodicBSplineBasis of order $k, domain [0.0, 1.0), period 1.0"
    )

    # The last knot must be strictly less than `first(breaks) + period`
    @test_throws ArgumentError PeriodicBSplineBasis(BSplineOrder(k), breaks, last(breaks))

    # Order must be ≥ 1
    @test_throws ArgumentError PeriodicBSplineBasis(BSplineOrder(0), breaks, L)

    @test PeriodicBSplineBasis(k, breaks, L) == B

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

    # TODO
    # - test splines
    # - test collocation and Galerkin
    # - test interpolations

    # Far from the boundaries, the result should match a regular BSplineBasis
    # (except for the knot index, which can be different).
    let Bn = BSplineBasis(BSplineOrder(k), breaks)
        x = (2 * breaks[k + 1] + breaks[k + 2]) / 3
        @test Bn(x)[2] == B(x)[2]
    end
end
