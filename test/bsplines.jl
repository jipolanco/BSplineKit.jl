using Test
using BSplineKit
using Random
using LinearAlgebra: dot

using BSplineKit.BSplines:
    find_knot_interval

function test_bsplines(ord::BSplineOrder)
    rng = MersenneTwister(42)
    ξs = sort!(rand(rng, 20))
    ξs[begin] = 0; ξs[end] = 1;
    B = BSplineBasis(ord, copy(ξs))
    N = length(B)
    ts = knots(B)
    a = first(ts)
    b = last(ts)
    @test (a, b) == boundaries(B)
    k = order(B)

    @testset "Find interval" begin
        @inferred find_knot_interval(ts, 0.3)
        @test find_knot_interval(ts, a - 1) == (1, -1)
        @test find_knot_interval(ts, a) == (k, 0)
        @test find_knot_interval(ts, b) == (N, 0)
        @test find_knot_interval(ts, b + 1) == (N, 1)
        let x = (a + b) / 3  # test on a "normal" location
            @test find_knot_interval(ts, x) == (searchsortedlast(ts, x), 0)
        end
        let i = length(ts) ÷ 2  # test on a knot location
            @test find_knot_interval(ts, ts[i]) == (i, 0)
        end
    end

    xs_eval = [
        ts[begin],                                          # at left boundary
        ts[begin + k + 2],                                  # on a knot
        0.2 * ts[begin + k + 2] + 0.8 * ts[begin + k + 3],  # between two knots
        ts[end],                                            # at right boundary
    ]

    S = Spline(B, randn(rng, length(B)))

    @testset "Evaluate" for x ∈ xs_eval
        i, bs = @inferred evaluate_all(B, x)
        @test length(bs) == k
        @test eltype(bs) === Float64

        let
            i′, bs′ = @inferred evaluate_all(B, x, Float32)
            @test eltype(bs′) === Float32
            @test i === i′
            @test all(bs .≈ bs′)
        end

        # Test reusing knot interval (ileft)
        @test (i, bs) == @inferred evaluate_all(B, x; ileft = i)

        @test sum(bs) ≈ 1  # partition of unity property

        # 1. Compare with evaluation of single B-spline
        for j ∈ eachindex(bs)
            @test bs[j] ≈ B[i - j + 1](x)
        end

        # 2. Compare with full evaluation of a spline
        cs = view(coefficients(S), i:-1:(i - k + 1))
        @test S(x) ≈ dot(cs, bs)

        # 3. Compare to non-generated version (assuming the @generated version
        #    is called)
        @test evaluate_all(B, x) == @inferred BSplines._evaluate_all_alt(
            knots(B), x, BSplineOrder(k), Derivative(0), eltype(bs),
        )
        @test evaluate_all(B, x, Float32) == @inferred BSplines._evaluate_all_alt(
            knots(B), x, BSplineOrder(k), Derivative(0), Float32,
        )
    end

    nothing
end

@testset "B-splines (k = $k)" for k ∈ (2, 4, 5, 6, 8)
    test_bsplines(BSplineOrder(k))
end
