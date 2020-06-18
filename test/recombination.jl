function test_nzrows(A::RecombineMatrix)
    # Compare output of nzrows with similar functions for sparse arrays.
    # See ?nzrange.
    @testset "nzrows" begin
        S = sparse(A)
        rows = rowvals(S)
        vals = nonzeros(S)
        for j in axes(A, 2)
            ra = nzrows(A, j)
            rs = nzrange(S, j)
            @test length(ra) == length(rs)  # same number of non-zero rows
            for (ia, is) in zip(ra, rs)
                @test ia == rows[is]        # same non-zero rows
                @test A[ia, j] == vals[is]  # same values
            end
        end
    end
    nothing
end

function test_recombine_matrix(A::RecombineMatrix)
    @testset "Recombination matrix: $(constraints(A))" begin
        test_nzrows(A)
        N, M = size(A)
        @test M == N - 2 * num_constraints(A)

        n = num_recombined(A)
        c = num_constraints(A)

        # The sum of all rows is 2 in the recombined regions (because 2
        # recombined B-splines), and 1 in the centre.
        @test vec(sum(A, dims=1)) ≈ vcat(2 * ones(n), ones(M - 2n), 2 * ones(n))

        # Upper left corner, lower right corner, and centre.
        @test @views A[1:(n + c), 1:n] == A.ul
        @test @views A[(N - n - c + 1):N, (M - n + 1):M] == A.lr
        @test @views A[(n + c + 1):(N - n - c), (n + 1):(M - n)] == I

        u = rand(M)
        v1 = similar(u, N)
        v2 = similar(v1)

        # Test custom matrix-vector multiplication.
        # The second version uses the getindex() function to build a regular
        # sparse matrix, which is less efficient.
        mul!(v1, A, u)
        mul!(v2, sparse(A), u)
        @test v1 == v2

        # Test left division.
        let v = copy(v1)
            u2 = A \ v
            @test u2 ≈ u

            # Modify coefficient of original basis near the border, so that the
            # resulting function has no representation in the recombined basis.
            v[1] = randn()
            @test_throws NoUniqueSolutionError A \ v
        end

        # Test 5-argument mul!.
        mul!(v2, A, u, 1, 0)  # equivalent to 3-argument mul!
        @test v1 == v2

        randn!(u)  # use different u, otherwise A * u = v1
        mul!(v2, A, u, 2, -3)
        @test v2 == 2 * (A * u) - 3 * v1
    end

    nothing
end

function test_boundary_conditions(R::RecombinedBSplineBasis{D}) where {D}
    orders = BasisSplines.get_orders(constraints(R)...)
    @test length(orders) == BasisSplines.num_constraints(R)
    @test length(R) == length(parent(R)) - 2 * length(orders)

    a, b = boundaries(R)
    N = length(R)
    k = order(R)

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
        # TODO ideally, the result should be exactly zero...
        B = parent(R)
        ε = 2 * num_recombined(R) * eps(BSpline(B, 1)(a, Derivative(n)))
        if n ∈ orders
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
    k = 4
    B = BSplineBasis(k, knots_base)
    @test constraints(B) === ()
    @testset "Mixed derivatives" begin
        @inferred RecombineMatrix(Derivative.((0, 1)), B)
        @test_throws ArgumentError RecombineMatrix(Derivative.((0, 2)), B)
        @test_throws ArgumentError RecombineMatrix(Derivative.((1, 2)), B)
        M = RecombineMatrix(Derivative.((0, 1)), B)
        test_recombine_matrix(M)
    end
    @testset "Order $D" for D = 0:(k - 1)
        R = RecombinedBSplineBasis(Derivative(D), B)
        @test constraints(R) === (Derivative(D), )
        test_recombine_matrix(recombination_matrix(R))
        test_boundary_conditions(R)

        # Simultaneously satisfies BCs of orders 0 to D.
        @testset "Mixed BCs" begin
            derivs = ntuple(d -> Derivative(d - 1), D + 1)
            Rs = RecombinedBSplineBasis(derivs, B)
            @test constraints(Rs) === ntuple(n -> Derivative(n - 1), D + 1)
            test_recombine_matrix(recombination_matrix(Rs))
            test_boundary_conditions(Rs)
        end
    end
    nothing
end

@testset "Basis recombination" begin
    test_basis_recombination()
end
