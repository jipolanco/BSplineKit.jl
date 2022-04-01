using Random

function test_interpolation(ord::BSplineOrder)
    rng = MersenneTwister(42)
    ndata = 40
    xs = sort(randn(rng, ndata))
    ys = randn(rng, ndata)
    k = order(ord)

    @testset "No BCs" begin
        S = @inferred interpolate(xs, ys, ord)
        @test S.(xs) ≈ ys
    end

    if iseven(k)
        @testset "Natural BCs" begin
            S = @inferred interpolate(xs, ys, ord, Natural())
            @test S.(xs) ≈ ys
            ts = @inferred knots(S)
            Nt = length(ts)
            ts_unique = view(ts, k:(Nt - k + 1))
            @test allunique(ts_unique)

            # Check that unique knots are the same as data locations (this is not
            # the case without natural BCs)
            @test length(ts_unique) == length(xs)
            @test ts_unique == xs

            # Check that some derivatives are zero.
            for n = 2:(k ÷ 2)
                Sder = Derivative(n) * S
                for x ∈ boundaries(basis(S))
                    @test abs(Sder(x)) < abs(S(x)) * 1e-7
                end
            end
        end
    end

    nothing
end

@testset "Interpolation" begin
    @testset "k = $k" for k ∈ (3, 4, 6, 8)
        test_interpolation(BSplineOrder(k))
    end
end
