using BasisSplines

using BandedMatrices
using LinearAlgebra
using SparseArrays
using Test

function test_recombined(R::RecombinedBSplineBasis{D}) where {D}
    a, b = boundaries(R)
    N = length(R)
    # Verify that all basis functions satisfy the boundary conditions.
    @test all(1:N) do i
        f = BSpline(R, i)
        f(a, Derivative(D)) == f(b, Derivative(D)) == 0
    end
    nothing
end

function test_collocation(B::BSplineBasis, xcol, ::Type{T} = Float64) where {T}
    @inferred collocation_matrix(B, xcol, Matrix{T})
    @inferred collocation_matrix(B, xcol, BandedMatrix{T})
    @inferred collocation_matrix(B, xcol, SparseMatrixCSC{T})

    C_dense = collocation_matrix(B, xcol, Matrix{T})
    C_banded = collocation_matrix(B, xcol, BandedMatrix{T})
    C_sparse = collocation_matrix(B, xcol, SparseMatrixCSC{T})

    @test C_dense == C_banded
    @test C_banded == C_sparse

    @testset "Basis recombination" begin
        C = collocation_matrix(B, xcol)
        N = length(B)

        # Recombined basis for Dirichlet BCs (u = 0)
        @inferred RecombinedBSplineBasis(Derivative(0), B)
        R0 = RecombinedBSplineBasis(Derivative(0), B)
        test_recombined(R0)
        Nr = length(R0)
        @test Nr == N - 2

        # Collocation points for recombined basis, for implicit boundary
        # conditions (points at the boundaries are removed!).
        x = collocation_points(R0)
        @test length(x) == Nr
        @test x == @view xcol[2:end-1]

        @inferred collocation_matrix(R0, x)
        C0 = collocation_matrix(R0, x)
        @test size(C0) == (Nr, Nr)

        # C0 is simply the interior elements of C.
        let r = 2:(N - 1)
            @test C0 == @view C[r, r]
        end

        # Neumann BCs (du/dx = 0)
        R1 = RecombinedBSplineBasis(Derivative(1), B)
        test_recombined(R1)
        @test collocation_points(R1) == x  # same as for Dirichlet
        C1 = collocation_matrix(R1, x)
        @test size(C1) == (Nr, Nr)

        let r = 2:(N - 1)
            @test @views C1[:, 1] == C[r, 1] .+ C[r, 2]
            @test @views C1[:, Nr] == C[r, N - 1] .+ C[r, N]
            @test @views C1[:, 2:(Nr - 1)] == C[r, 3:(N - 2)]
        end
    end

    nothing
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

    nothing
end

function test_galerkin_recombined()
    @testset "Basis recombination" begin
        N = 40
        knots_base = [-cos(2pi * n / N) for n = 0:N]

        B = BSplineBasis(6, knots_base)
        R0 = RecombinedBSplineBasis(Derivative(0), B)
        R1 = RecombinedBSplineBasis(Derivative(1), B)

        N = length(B)
        Ñ = length(R0)
        @test Ñ == N - 2

        G = galerkin_matrix(B)
        G0 = galerkin_matrix(R0)
        G1 = galerkin_matrix(R1)

        @test size(G) == (N, N)
        @test size(G0) == size(G1) == (Ñ, Ñ)

        let r = 2:(N - 1)
            # G0 is simply the interior elements of G.
            @test G0 == @view G[r, r]

            # First row
            @test G1[1, 1] ≈ G[1, 1] + 2G[1, 2] + G[2, 2]  # (b1b1 + 2b1b2 + b2b2)
            @test @views G1[1, 2:(Ñ - 1)] ≈ G[1, 3:Ñ] .+ G[2, 3:Ñ]

            # Last row
            @test G1[Ñ, Ñ] ≈ G[N - 1, N - 1] + 2G[N, N - 1] + G[N, N]
            @test @views G1[Ñ, 2:(Ñ - 1)] ≈ G[N - 1, 3:Ñ] + G[N, 3:Ñ]

            # Interior
            @test @views G1[2:(Ñ - 1), 2:(Ñ - 1)] ≈ G[3:Ñ, 3:Ñ]
        end
    end

    nothing
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
    @test xcol[1] == knots_in[1]
    @test xcol[end] == knots_in[end]

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
        test_galerkin_recombined()
    end
    nothing
end

main()
