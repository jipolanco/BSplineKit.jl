using BSplines

using LinearAlgebra
using Test

function test_splines(B::BSplineBasis, knots_in)
    k = order(B)
    t = knots(B)

    let (ka, kb) = BSplines.multiplicity.(Ref(t), (1, length(t)))
        @test ka == kb == k
    end

    @test @views all(t[1:k] .== knots_in[1])
    @test @views all(t[(end - k + 1):end] .== knots_in[end])
    @test @views t[(k + 1):(end - k)] == knots_in[2:(end - 1)]

    N = length(B)
    @test_throws DomainError evaluate_bspline(B, 0, 0.2)
    @test_throws DomainError evaluate_bspline(B, N + 1, 0.2)

    # Verify values at the boundaries.
    @test evaluate_bspline(B, 1, t[1]) == 1.0
    @test evaluate_bspline(B, N, t[end]) == 1.0

    @inferred collocation_points(B)
    xcol = collocation_points(B)

    @inferred collocation_matrix(B, xcol)
    C = collocation_matrix(B, xcol)

    # Generate data at collocation points and get B-spline coefficients.
    ucol = cos.(xcol)
    coefs = C \ ucol

    @inferred Spline(B, coefs)
    S = Spline(B, coefs)
    @test all(S.(xcol) .≈ ucol)

    nothing
end

function test_splines(::Val{k}) where {k}
    knots_in = let N = 10 + k
        [-cos(n * π / N) for n = 0:N]
    end

    @inferred BSplineBasis(Val(k), knots_in)
    @inferred (() -> BSplineBasis(k, knots_in))()

    g = BSplineBasis(k, knots_in)
    @test order(g) == k
    test_splines(g, knots_in)

    nothing
end

function main()
    test_splines(Val(3))
    test_splines(Val(4))
end

main()
