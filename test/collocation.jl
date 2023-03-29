using LinearAlgebra
using Random

function test_collocation_matrix()
    # Test special case of 1×1 matrix
    @testset "1×1 matrix" begin
        C = @inferred CollocationMatrix([0.0;;])
        @test_throws ZeroPivotException(1) lu!(C)
        C[1, 1] = 3
        F = lu(C)
        @test F.L == [1;;]
        @test F.U == [3;;]
        y = [6]
        x = F \ y
        @test x == [2]
    end

    # Special cases of triangular input matrices
    @testset "Upper triangular" begin
        U = CollocationMatrix(brand(Float32, 8, 8, 2, 2))
        U[Band(-1)] .= 0
        U[Band(-2)] .= 0
        F = lu(U)
        @test F.L == I
        @test F.U == U

        U[3, 3] = 0
        @test_throws ZeroPivotException(3) lu!(U)
    end

    @testset "Lower triangular" begin
        L = CollocationMatrix(brand(Float32, 8, 8, 2, 2))
        L[Band(1)] .= 0
        L[Band(2)] .= 0
        D = Diagonal(L)
        F = lu(L)
        @test F.U == D
        @test F.L * D ≈ L

        L[4, 4] = 0
        @test_throws ZeroPivotException(4) lu!(L)
    end

    @testset "Linear system" begin
        C = @inferred CollocationMatrix(rand(5, 20))
        C[Band(0)] .+= 4  # diagonally dominant matrix
        y = rand(20)
        x = @inferred C \ y
        @test C * x ≈ y
    end

    nothing
end

function test_collocation(B::BSplineBasis, ::Type{T} = Float64) where {T}
    xcol = collocation_points(B)
    @inferred collocation_matrix(B, xcol, Matrix{T})
    @inferred collocation_matrix(B, xcol, CollocationMatrix{T})
    @inferred collocation_matrix(B, xcol, SparseMatrixCSC{T})

    C_dense = collocation_matrix(B, xcol, Matrix{T})
    C_banded = collocation_matrix(B, xcol, CollocationMatrix{T})
    C_sparse = collocation_matrix(B, xcol, SparseMatrixCSC{T})

    @test C_dense == C_banded
    @test C_banded == C_sparse

    @testset "Greville sites" begin
        xs = collocation_points(B; method = Collocation.AvgKnots())
        it = Collocation.GrevilleSiteIterator(B)
        @test xs == collect(it)
    end

    @testset "Basis recombination" begin
        C = collocation_matrix(B, xcol)
        N = length(B)

        # Recombined basis for Dirichlet BCs (u = 0)
        @inferred RecombinedBSplineBasis(Derivative(0), B)
        R0 = RecombinedBSplineBasis(Derivative(0), B)
        Nr = length(R0)
        @test Nr == N - 2

        # Collocation points for recombined basis, for implicit boundary
        # conditions (points at the boundaries are removed!).
        x = collocation_points(R0)
        @test length(x) == Nr
        @test x ≈ @view xcol[2:end-1]

        @inferred collocation_matrix(R0, x)
        C0 = collocation_matrix(R0, x)
        @test size(C0) == (Nr, Nr)

        # C0 is simply the interior elements of C.
        let r = 2:(N - 1)
            @test C0 ≈ @view C[r, r]
        end

        # Neumann BCs (du/dx = 0)
        R1 = RecombinedBSplineBasis(Derivative(1), B)
        @test collocation_points(R1) == x  # same as for Dirichlet
        C1 = collocation_matrix(R1, x)
        @test size(C1) == (Nr, Nr)

        let r = 2:(N - 1)
            @test @views C1[:, 1] ≈ C[r, 1] .+ C[r, 2]
            @test @views C1[:, Nr] ≈ C[r, N - 1] .+ C[r, N]
            @test @views C1[:, 2:(Nr - 1)] ≈ C[r, 3:(N - 2)]
        end
    end

    @testset "Out-of-bounds points" begin
        a, b = boundaries(B)
        xs = [a - eps(a), a, (a + b) / 2, b, b + eps(b)]
        C = collocation_matrix(B, xs, Matrix{Float64})
        @test all(iszero, @view(C[1, :]))  # x < a
        for i ∈ 2:4
            @test sum(@view(C[i, :])) ≈ 1  # a ≤ x ≤ b (+ partition of unity)
        end
        @test all(iszero, @view(C[5, :]))  # x > b
    end

    nothing
end

@testset "Collocation matrix" begin
    test_collocation_matrix()
end

@testset "Collocation (k = $k)" for k ∈ (4, 5)
    t = gauss_lobatto_points(10 + k)
    B = BSplineBasis(k, t)
    test_collocation(B)
end
