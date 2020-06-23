function test_galerkin()
    # Compare Galerkin mass matrix against analytical integrals for k = 2
    # (easy!).
    # Test with non-uniform grid (Chebyshev points).
    knots_base = gauss_lobatto_points(42)
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
        knots_base = gauss_lobatto_points(41)

        B = BSplineBasis(6, knots_base)
        R0 = RecombinedBSplineBasis(Derivative(0), B)
        R1 = RecombinedBSplineBasis(Derivative(1), B)
        R2 = RecombinedBSplineBasis(Derivative(2), B)

        test_galerkin_tensor(R0)
        test_galerkin_tensor(R1)

        N = length(B)
        Ñ = length(R0)
        @test Ñ == N - 2

        @inferred galerkin_matrix(B)
        @inferred galerkin_matrix(R2)
        G = galerkin_matrix(B)
        G0 = galerkin_matrix(R0)
        G1 = galerkin_matrix(R1)
        G2 = galerkin_matrix(R2)

        # Due to the partition of unity property, the sum of all elements
        # must be the size of the domain (= ∫ 1 dx).
        L = -(boundaries(B)...) * -1
        @test sum(G) ≈ L
        @test sum(G0) < L  # Dirichlet: ϕ_1 = b_2 => PoU is not satisfied
        @test sum(G1) ≈ L  # Neumann: ϕ_1 = b_1 + b_2 => PoU is satisfied
        @test !(sum(G2) ≈ L)  # here it's not true!

        Sym{M} = Hermitian{T,A} where {T, A<:M}

        # Some symmetric matrices
        @inferred galerkin_matrix(R1, Derivative.((1, 1)))
        @test galerkin_matrix(R1, Derivative.((1, 1))) isa Sym{BandedMatrix}
        @test galerkin_matrix(R1, SparseMatrixCSC{Float32}) isa Sym{SparseMatrixCSC{Float32}}

        # Some non-symmetric matrices
        @inferred galerkin_matrix(R1, Derivative.((0, 1)))
        @test galerkin_matrix(R1, Derivative.((0, 1))) isa BandedMatrix
        @test galerkin_matrix(R1, Derivative.((0, 1)), Matrix{Float64}) isa Matrix{Float64}
        @inferred galerkin_matrix((B, R1))
        @test galerkin_matrix((B, R1)) isa BandedMatrix

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

        @testset "Integration by parts" begin
            # We test the relation between the H = ⟨ ϕᵢ, ϕⱼ″ ⟩ and the
            # Q = ⟨ ϕᵢ′, ϕⱼ′ ⟩ matrices, associated to the Laplacian term
            # for instance in the heat equation. These are clearly related by
            # integration by parts.
            # If the (recombined) basis satisfies homogeneous Dirichlet or
            # Neumann boundary conditions, the term ϕᵢϕⱼ′ appearing in the IBP
            # should vanish at the boundaries, in which case the matrices are
            # simply related as H = -Q.
            # We test both boundary conditions.
            @testset "BC: $D" for D in Derivative.((0, 1))
                R = RecombinedBSplineBasis(D, B)
                H = galerkin_matrix(R, Derivative.((0, 2)))
                Q = galerkin_matrix(R, Derivative.((1, 1)))
                N = length(R)
                ε = N^2 * eps(eltype(Q))  # the noise seems to scale as N^2
                @test norm(H + Q, Inf) < norm(Q, Inf) * ε

                # Similar thing for the tensor F_{ijk} = ⟨ ϕᵢ, ϕⱼ, ϕₖ″ ⟩.
                T″ = galerkin_tensor(R, Derivative.((0, 0, 2)))
                T1 = galerkin_tensor(R, Derivative.((1, 0, 1)))
                T2 = galerkin_tensor(R, Derivative.((0, 1, 1)))
                @test T2 == permutedims(T1, (2, 1, 3))
                @test norm(T″ + T1 + T2, Inf) < norm(T1, Inf) * ε
            end
        end

        bcs = (Derivative.((0, )), Derivative.((0, 1)))
        @testset "Combining bases ($bc)" for bc in bcs
            R = RecombinedBSplineBasis(bc, B)
            M = galerkin_matrix((R, B))
            test_galerkin_tensor(R)
            @test M == galerkin_matrix((B, R))'
            let δ = num_constraints(R)
                n, m = size(M)
                @test n + 2δ == m
                r = num_recombined(R)
                # Verify that except from the borders, the matrix is the same as
                # the "original" matrix constructed from the original B-spline
                # basis.
                M_base = galerkin_matrix(B)
                p = δ + r
                I = (p + 1):(m - p)
                @test view(M, (r + 1):(n - r), I) == view(M_base, I, I)
            end
        end
    end

    nothing
end

function test_galerkin_tensor(R::RecombinedBSplineBasis,
                              derivs=Derivative.((0, 1, 0)))
    B = parent(R) :: BSplineBasis

    # The first two bases must be the same
    @test_throws ArgumentError galerkin_tensor((B, R, R), derivs)

    A_base = galerkin_tensor((B, B, B), derivs)
    A = galerkin_tensor((B, B, R), derivs)
    @inferred galerkin_tensor((B, B, R), derivs)

    N1, N2, N3 = size(A)
    δ = num_constraints(R)
    @test size(A_base) == (N1, N2, N3 + 2δ)

    # Verify that except from the borders, the tensor is the same as the
    # "original" tensor constructed from the original B-spline basis.
    r = num_recombined(R)
    p = δ + r
    m, n = N2, N3
    I = (p + 1):(m - p)
    @test view(A, I, I, (r + 1):(n - r)) == view(A_base, I, I, I)

    nothing
end

@testset "Galerkin" begin
    test_galerkin()
    test_galerkin_recombined()
end
