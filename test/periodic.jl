using BSplineKit
using LinearAlgebra
using Random
using Test

function test_periodic_splines(ord::BSplineOrder)
    k = order(ord)
    L = 1  # period
    breaks = range(0, L; step = 0.05)[begin:end-1]
    B = @inferred PeriodicBSplineBasis(ord, breaks, L)
    N = length(B)

    ts = @inferred knots(B)
    @test length(ts) == N + k  # consistency with the regular BSplineBasis

    @test N == length(breaks)
    @test @inferred(period(B)) == L
    @test typeof(period(B)) === eltype(breaks)
    @test @inferred(boundaries(B)) == (0, 1)
    @test B == B
    let B′ = PeriodicBSplineBasis(BSplineOrder(k - 1), breaks, L)
        @test B ≠ B′
    end

    @test startswith(
        repr(B),
        "$N-element PeriodicBSplineBasis of order $k, domain [0.0, 1.0), period 1.0"
    )

    # The last knot must be strictly less than `first(breaks) + period`
    @test_throws ArgumentError PeriodicBSplineBasis(ord, breaks, last(breaks))

    # Order must be ≥ 1
    @test_throws ArgumentError PeriodicBSplineBasis(BSplineOrder(0), breaks, L)

    @test PeriodicBSplineBasis(k, breaks, L) == B

    rng = MersenneTwister(42)
    S = @inferred Spline(B, randn(rng, length(B)))

    # This is mainly to check that the coefficients are not re-wrapped in a
    # PeriodicVector.
    let Salt = @inferred Spline(B, coefficients(S))
        @test S == Salt
        @test typeof(S) === typeof(Salt)
    end

    @testset "Evaluate x = $x" for x ∈ (0.02, 0.42, 0.89)
        ilast, bs = @inferred B(x)
        @test (ilast, bs) == @inferred B(x; ileft = ilast)
        @test all(>(0), bs)
        @test sum(bs) ≈ 1  # partition of unity

        # Check for zero allocations
        eval_alloc(B, x) = (B(x); @allocated(B(x)))
        @test 0 == eval_alloc(B, x)

        # Check spline evaluation
        coefs = coefficients(S)
        Sx = sum(enumerate(bs)) do (δi, bi)
            i = mod1(ilast + 1 - δi, length(coefs))
            coefs[i] * bi
        end
        @test Sx ≈ S(x)
        @test 0 == eval_alloc(S, x)

        # Check periodicity
        for n ∈ (-2, 1)
            ilast′, bs′ = B(x + n * L)
            @test ilast′ == ilast + n * N
            @test all(bs .≈ bs′)
        end
    end

    # Far from the boundaries, the result should match a regular BSplineBasis
    # (except for the knot index, which is shifted).
    @testset "Compare to regular" begin
        Bn = BSplineBasis(ord, breaks)
        x = (2 * breaks[k + 2] + breaks[k + 3]) / 3

        # Index offset for matching regular to periodic basis
        δ = (order(B) - 1) - BSplines.index_offset(knots(B))

        @testset "B-spline evaluation" begin
            iₙ, bsₙ = Bn(x)  # B-splines from regular ("normal") basis
            iₚ, bsₚ = B(x)   # B-splines from periodic basis
            @test iₙ == iₚ + δ  # knot indices are shifted
            @test bsₙ == bsₚ    # b-spline values are the same
        end

        @testset "Spline evaluation" begin
            coefs = randn(rng, length(Bn))
            Sn = Spline(Bn, coefs)
            Sp = Spline{Float64}(undef, B)
            coefficients(Sp) .= @view coefs[(δ + 1):(δ + length(Sp))]
            @test Sn(x) == Sp(x)
        end
    end

    ftest(x) = sinpi(2x)  # should be L-periodic (L = 1)

    @testset "Collocation + interpolation" begin
        xs = @inferred collocation_points(B)
        @test xs == breaks  # this is the default
        C = @inferred collocation_matrix(B, xs)

        # Interpolate manually
        ys = ftest.(xs)
        cs = C \ ys
        S = @inferred Spline(B, cs)
        @test S.(xs) ≈ ys

        # Compare with interpolation interface
        I = @inferred interpolate(xs, ys, ord, Periodic(L))
        @test I.(xs) ≈ ys
    end

    # TODO
    # - test Galerkin + approximations
    # - test derivatives
end

@testset "Periodic splines (k = $k)" for k ∈ (3, 4, 6)
    test_periodic_splines(BSplineOrder(k))
end
