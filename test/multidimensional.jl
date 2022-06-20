# Test tensor-product splines.

using BSplineKit
using Random
using StaticArrays
using Test

# Compare with separable N-dimensional polynomial P(x, y, …) = P₁(x) * P₂(y) * …
function test_polynomial_multidim(::Val{N}, ord::BSplineOrder) where {N}
    k = order(ord)

    # Polynomial coefficients
    # P₁(x) = -1 + 2x - 3x^2 +  4x^3 - ...
    # P₂(y) = -2 + 4x - 6x^2 +  8x^3 - ...
    # P₃(z) = -3 + 6x - 9x^2 + 12x^3 - ...
    Ps = ntuple(Val(N)) do n
        ntuple(d -> n * d * sign((-1)^d), Val(k))
    end

    # Ps′ = map(Ps) do P
    #     ntuple(d -> d * P[d + 1], Val(k - 1))
    # end

    polyeval(Ps::NTuple, xs::NTuple) = prod(zip(Ps, xs)) do (P, x)
        @evalpoly(x, P...)
    end

    limits = ntuple(n -> (0, n), Val(N))

    randpoint(rng, limits) = map(limits) do (a, b)
        a + rand(rng) * (b - a)
    end

    @testset "Interpolation" begin
        points = ntuple(n -> range(limits[n]...; step = 0.1), Val(N))
        xiter = Iterators.product(points...)
        fdata = map(xs -> polyeval(Ps, xs), xiter)
        I = @inferred interpolate(points, fdata, ord)
        @inferred I(first(xiter)...)

        # The interpolation must exactly match the original polynomial (up to
        # roundoff error).
        rng = MersenneTwister(42)
        @test all(1:100) do _  # evaluate at 100 random points
            xs = randpoint(rng, limits)
            polyeval(Ps, xs) ≈ I(xs...)
        end

        # TODO implement derivatives
        noderivs = ntuple(_ -> Derivative(0), Val(N - 1))
        @test_skip derivative(I, (Derivative(1), noderivs...))
    end

    @testset "Approximation" begin
        f(xs...) = polyeval(Ps, xs)  # function to approximate
        Bs = map(limits) do lims
            BSplineBasis(ord, range(lims...; step = 0.2))
        end

        methods = (
            ApproxByInterpolation(Bs),
            # MinimiseL2Error(),
        )

        # The approximation must exactly match the original polynomial (up to
        # roundoff error).
        @testset "$(nameof(typeof(m)))" for m ∈ methods
            g = approximate(f, Bs, m)
            rng = MersenneTwister(43)
            @test all(1:100) do _  # evaluate at 100 random points
                xs = randpoint(rng, limits)
                f(xs...) ≈ g(xs...)
            end
        end
    end
end

@testset "Tensor-product splines ($N-D)" for N ∈ (2, 3)
    @testset "Compare with polynomial" begin
        test_polynomial_multidim(Val(N), BSplineOrder(4))
    end
end
