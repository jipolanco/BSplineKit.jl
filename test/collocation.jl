function test_collocation(B::BSplineBasis, ::Type{T} = Float64) where {T}
    xcol = collocation_points(B)
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

@testset "Collocation (k = $k)" for k ∈ (4, 5)
    t = gauss_lobatto_points(10 + k)
    B = BSplineBasis(k, t)
    test_collocation(B)
end
