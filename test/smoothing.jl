# Test smoothing splines

using BSplineKit
using QuadGK: quadgk
using Test

# Returns the integral of S''(x) (the "curvature") over the whole spline.
function total_curvature(S::Spline)
    ts = knots(S)
    k = order(S)
    xs = @view ts[(begin - 1 + k):(end + 1 - k)]  # exclude repeated knots at the boundaries
    @assert length(xs) == length(S)
    S″ = Derivative(2) * S
    curv, err = quadgk(xs) do x
        abs2(S″(x))
    end
    curv
end

function distance_from_data(S::Spline, xs, ys)
    dist = zero(eltype(ys))
    for i in eachindex(xs, ys)
        dist += abs2(ys[i] - S(xs[i]))
    end
    sqrt(dist / length(xs))
end

@testset "Smoothing cubic splines" begin
    xs = (0:0.01:1).^2
    ys = @. cospi(2 * xs) + 0.1 * sinpi(200 * xs)  # smooth + highly fluctuating components

    # Check that smoothing with λ = 0 is equivalent to interpolation
    @testset "Fit λ = 0" begin
        λ = 0
        S = @inferred fit(xs, ys, λ)
        S_interp = spline(interpolate(xs, ys, BSplineOrder(4), Natural()))
        @test coefficients(S) ≈ coefficients(S_interp)
    end

    @testset "Smoothing" begin
        λs = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        curvatures = similar(λs)
        distances = similar(λs)
        for i in eachindex(λs)
            S = @inferred fit(xs, ys, λs[i])
            curvatures[i] = total_curvature(S)
            distances[i] = distance_from_data(S, xs, ys)
        end
        @test issorted(curvatures; rev = true)  # in decreasing order (small λ => large curvature)
        @test issorted(distances)  # in increasing order (large λ => large distance from data)
    end

    @testset "Weights" begin
        λ = 1e-3
        weights = fill!(similar(xs), 1)
        S = fit(xs, ys, λ)
        Sw = @inferred fit(xs, ys, λ; weights)  # equivalent to the default (all weights are 1)
        @test coefficients(S) == coefficients(Sw)
        # Now give more weight to point i = 3
        weights[3] = 1000
        Sw = fit(xs, ys, λ; weights)
        @test abs(Sw(xs[3]) - ys[3]) < abs(S(xs[3]) - ys[3])  # the new curve is closer to the data point i = 3
        @test total_curvature(Sw) > total_curvature(S)  # since we give more importance to data fitting (basically, the sum of weights is larger)
        @test distance_from_data(Sw, xs, ys) < distance_from_data(S, xs, ys)
    end
end
