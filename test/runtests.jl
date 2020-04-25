using BasisSplines

using BandedMatrices
using LinearAlgebra
using SparseArrays
using Test

function test_collocation(B::BSplineBasis, xcol, ::Type{T} = Float64) where {T}
    @inferred collocation_matrix(B, xcol, Matrix{T})
    @inferred collocation_matrix(B, xcol, BandedMatrix{T})
    @inferred collocation_matrix(B, xcol, SparseMatrixCSC{T})

    C_dense = collocation_matrix(B, xcol, Matrix{T})
    C_banded = collocation_matrix(B, xcol, BandedMatrix{T})
    C_sparse = collocation_matrix(B, xcol, SparseMatrixCSC{T})

    @test C_dense == C_banded
    @test C_banded == C_sparse
end

function test_galerkin()
    # Compare Galerkin mass matrix against analytical integrals for k = 2
    # (easy!).
    # Test with non-uniform grid (Chebyshev points).
    N = 40
    knots_base = [-cos(2pi * n / N) for n = 0:N]
    B = BSplineBasis(2, knots_base)
    t = knots(B)
    G = galerkin_matrix(B)
    let i = 8
        @test G[i, i] ≈ (t[i + 2] - t[i]) / 3
        @test G[i - 1, i] ≈ (t[i + 1] - t[i]) / 6
    end
end

function test_splines(B::BSplineBasis, knots_in)
    k = order(B)
    t = knots(B)

    @testset "Knots (k = $k)" begin
        let (ka, kb) = BasisSplines.multiplicity.(Ref(t), (1, length(t)))
            @test ka == kb == k
        end

        @test @views all(t[1:k] .== knots_in[1])
        @test @views all(t[(end - k + 1):end] .== knots_in[end])
        @test @views t[(k + 1):(end - k)] == knots_in[2:(end - 1)]
    end

    @testset "B-splines (k = $k)" begin
        N = length(B)
        @test_throws DomainError evaluate_bspline(B, 0, 0.2)
        @test_throws DomainError evaluate_bspline(B, N + 1, 0.2)

        # Verify values at the boundaries.
        @test evaluate_bspline(B, 1, t[1]) == 1.0
        @test evaluate_bspline(B, N, t[end]) == 1.0
    end

    xcol = collocation_points(B, method=Collocation.AvgKnots())

    @testset "Collocation (k = $k)" begin
        test_collocation(B, xcol)
    end

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
    knots_in = let N = 10 + k
        [-cos(n * π / N) for n = 0:N]
    end

    @inferred BSplineBasis(BSplineOrder(k), knots_in)
    @inferred (() -> BSplineBasis(k, knots_in))()

    g = BSplineBasis(k, knots_in)
    @test order(g) == k
    test_splines(g, knots_in)

    nothing
end

function main()
    test_splines(BSplineOrder(3))
    test_splines(BSplineOrder(4))
    @testset "Galerkin" begin
        test_galerkin()
    end
end

main()
