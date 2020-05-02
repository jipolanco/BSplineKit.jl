using BasisSplines

using BandedMatrices
using LinearAlgebra
using Random
using SparseArrays
using Test

# Chebyshev (Gauss-Lobatto) points.
gauss_lobatto_points(N) = [-cos(π * n / N) for n = 0:N]

function test_recombine_matrix(A::RecombineMatrix)
    @testset "Recombination matrix" begin
        M, N = size(A)
        @test M == N - 2

        n = order_bc(A)

        @test @views A[1:n, 1:(n + 1)] == A.ul            # upper-left corner
        @test @views A[(M - n + 1):M, (N - n):N] == A.lr  # lower-right corner
        @test @views A[(n + 1):(M - n), (n + 2):(N - n - 1)] == I

        u = rand(N)
        v1 = similar(u, M)
        v2 = similar(v1)

        # Test custom matrix-vector multiplication.
        # The second version uses the getindex() function to build a regular
        # sparse matrix, which is less efficient.
        mul!(v1, A, u)
        mul!(v2, sparse(A), u)
        @test v1 == v2

        # Test 5-argument mul!.
        mul!(v2, A, u, 1, 0)  # equivalent to 3-argument mul!
        @test v1 == v2

        randn!(u)  # use different u, otherwise A * u = v1
        mul!(v2, A, u, 2, -3)
        @test v2 == 2 * (A * u) - 3 * v1

        # Same for transposed / adjoint matrix.
        u1 = u
        u2 = similar(u)
        for At in (transpose(A), adjoint(A))
            v = v1
            mul!(u1, At, v)
            mul!(u2, sparse(At), v)
            @test u1 == u2

            mul!(u2, At, v, 1, 0)
            @test u1 == u2

            randn!(v)
            mul!(u2, At, v, 2, -3)
            @test u2 == 2 * (At * v) - 3 * u1
        end
    end

    nothing
end

function test_recombined(R::RecombinedBSplineBasis{D}) where {D}
    a, b = boundaries(R)
    N = length(R)
    k = order(R)
    test_recombine_matrix(recombination_matrix(R))

    # Verify that all basis functions satisfy the boundary conditions
    # for the derivative `D`, while they leave at least one degree of freedom
    # to set the other derivatives.

    # For instance, if D = 1 (Neumann BC), we want:
    #   (1) ϕⱼ'(a) = 0 ∀j,
    #   (2) ∑_j |ϕⱼ(a)| > 0,
    #   (3) ∑_j |ϕⱼ''(a)| > 0 (if k >= 3; otherwise the B-splines are not C²),
    # etc...

    for n = 0:(k - 1)
        bsum = sum(1:N) do i
            f = BSpline(R, i)
            fa = f(a, Derivative(n))
            fb = f(b, Derivative(n))
            abs(fa) + abs(fb)  # each of these must be 0 if n = D
        end
        # The sum must be zero if and only if n = D.
        # We consider that the result is zero, if it is negligible wrt the
        # derivative at the border of the first B-spline of the original
        # basis.
        B = parent(R)
        ε = 2 * eps(BSpline(B, 1)(a, Derivative(n)))
        if n == D
            @test bsum <= ε
        else
            @test bsum > 1
        end
    end

    nothing
end

# Verify that basis recombination gives the right boundary conditions.
function test_basis_recombination()
    knots_base = gauss_lobatto_points(40)
    k = 6
    B = BSplineBasis(k, knots_base)
    @test order_bc(B) === nothing
    @testset "Order $D" for D = 0:(k - 1)
        R = RecombinedBSplineBasis(Derivative(D), B)
        @test order_bc(R) === D
        test_recombined(R)
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

        @testset "Integration by parts (Laplacian)" begin
            # We test the relation between the H = ⟨ ϕᵢ, ϕⱼ'' ⟩ and the
            # Q = ⟨ ϕᵢ', ϕⱼ' ⟩ matrices, associated to the Laplacian term
            # for instance in the heat equation. These are clearly related by
            # integration by parts.
            # If the (recombined) basis satisfies homogeneous Dirichlet or
            # Neumann boundary conditions, the term ϕᵢϕⱼ' appearing in the IBP
            # should vanish at the boundaries, in which case the matrices are
            # simply related as H = -Q.
            # We test both boundary conditions.
            @testset "BC: $D" for D in Derivative.((0, 1))
                R = RecombinedBSplineBasis(D, B)
                H = galerkin_matrix(R, Derivative.((0, 2)))
                Q = galerkin_matrix(R, Derivative.((1, 1)))
                N = length(R)  # the noise seems to scale as N^2
                @test norm(H + Q, Inf) < norm(Q, Inf) * N^2 * eps(eltype(Q))
            end
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
    knots_in = gauss_lobatto_points(10 + k)

    @inferred BSplineBasis(BSplineOrder(k), knots_in)
    @inferred (() -> BSplineBasis(k, knots_in))()

    g = BSplineBasis(k, knots_in)
    @test order(g) == k
    test_splines(g, knots_in)

    nothing
end

function main()
    test_splines(BSplineOrder(4))
    test_splines(BSplineOrder(5))
    @testset "Galerkin" begin
        test_galerkin()
        test_galerkin_recombined()
    end
    @testset "Basis recombination" begin
        test_basis_recombination()
    end
    nothing
end

main()
