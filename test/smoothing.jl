# Test smoothing splines

using BSplineKit
using QuadGK: quadgk
using ReverseDiff
using Test

# This is the objective function that `fit` is supposed to minimise.
# We can verify this using automatic differentiation: the gradient wrt the spline
# coefficients should be zero.
function smoothing_objective(cs, R::AbstractBSplineBasis, xs, ys; weights = nothing, λ)
    # Construct spline from coefficients and knots
    S = Spline(R, cs)
    S″ = Derivative(2) * S

    # Compute first term of objective (loss) function
    T = eltype(cs)
    loss = zero(T)
    for i in eachindex(xs, ys)
        w = weights === nothing ? 1 : weights[i]
        loss += w * abs2(ys[i] - S(xs[i]))
    end

    # Integrate roughness term interval by interval
    for i in eachindex(xs)[1:end-1]
        # Note: S″(x) is linear within each interval, and thus the integrand is quadratic.
        # Therefore, a two-point GL quadrature is exact (weights = 1 and locations = ±1/√3).
        a = xs[i]
        b = xs[i + 1]
        Δ = (b - a) / 2
        xc = (a + b) / 2
        gl_weight = 1
        gl_ξ = 1 / sqrt(T(3))
        for ξ in (-gl_ξ, +gl_ξ)
            x = Δ * ξ + xc
            loss += λ * Δ * gl_weight * abs2(S″(x))
        end
    end

    loss
end

function check_zero_gradient(S::Spline, xs, ys; weights = nothing, λ)
    cs = coefficients(S)
    R = basis(S)  # usually a RecombinedBSplineBasis

    # Not sure how useful this is...
    ∇f = similar(cs)  # gradient wrt coefficients
    inputs = (cs,)
    results = (∇f,)
    # all_results = map(DiffResults.GradientResult, results)
    cfg = ReverseDiff.GradientConfig(inputs)

    # Compute gradient
    ReverseDiff.gradient!(results, cs -> smoothing_objective(cs, R, xs, ys; weights, λ), inputs, cfg)

    # Verify that |∇f|² is negligible. Note that is has the same units as |y_i|² ≡ Y², since
    # f ~ Y² and therefore ∂f/∂cⱼ ~ Y. So we compare it with the sum of |y_i|².
    reference = sum(abs2, ys)
    err = sum(abs2, ∇f)
    @test err / reference < 1e-12

    nothing
end

# Returns the integral of |S''(x)| (the "curvature") over the whole spline.
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
        for (i, λ) in pairs(λs)
            S = @inferred fit(xs, ys, λ)
            curvatures[i] = total_curvature(S)
            distances[i] = distance_from_data(S, xs, ys)
            check_zero_gradient(S, xs, ys; λ)
        end
        @test issorted(curvatures; rev = true)  # in decreasing order (small λ => large curvature)
        @test issorted(distances)  # in increasing order (large λ => large distance from data)
    end

    @testset "Weights" begin
        λ = 1e-3
        weights = fill!(similar(xs), 1)
        S = fit(xs, ys, λ)
        Sw = @inferred fit(xs, ys, λ; weights)  # equivalent to the default (all weights are 1)
        check_zero_gradient(Sw, xs, ys; λ, weights)
        @test coefficients(S) == coefficients(Sw)
        # Now give more weight to point i = 3
        weights[3] = 1000
        Sw = fit(xs, ys, λ; weights)
        check_zero_gradient(Sw, xs, ys; λ, weights)
        @test abs(Sw(xs[3]) - ys[3]) < abs(S(xs[3]) - ys[3])  # the new curve is closer to the data point i = 3
        @test total_curvature(Sw) > total_curvature(S)  # since we give more importance to data fitting (basically, the sum of weights is larger)
        @test distance_from_data(Sw, xs, ys) < distance_from_data(S, xs, ys)
    end
end
