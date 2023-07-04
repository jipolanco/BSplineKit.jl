using Test
using BSplineKit
using SparseArrays
using LinearAlgebra
using Random

using BSplineKit:
    AbstractDifferentialOp,
    DifferentialOpSum,
    LeftNormal,
    RightNormal

using BSplineKit.Recombinations:
    NoUniqueSolutionError,
    num_constraints,
    num_recombined

# Chebyshev (Gauss-Lobatto) points.
gauss_lobatto_points(N) = [-cos(π * n / N) for n = 0:N]

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
    bcleft, bcright = constraints(A)
    @assert bcleft === bcright  # only supported case for now
    @testset "Recombination matrix: $(bcleft)" begin
        test_nzrows(A)
        N, M = size(A)
        nl, nr = num_recombined(A)
        cl, cr = num_constraints(A)
        @assert nl == nr && cl == cr  # only supported case for now
        n = nl
        c = cl
        @test M == N - sum(num_constraints(A))

        # The sum of all rows is 2 in the recombined regions (because 2
        # recombined B-splines), and 1 in the centre.
        # Note that this is an arbitrary choice of the implementation.
        @test vec(sum(A, dims=1)) ≈ vcat(2 * ones(n), ones(M - 2n), 2 * ones(n))

        # Upper left corner, lower right corner, and centre.
        (ul, lr) = Recombinations.submatrices(A)
        @test @views A[1:(n + c), 1:n] == ul
        @test @views A[(N - n - c + 1):N, (M - n + 1):M] == lr
        @test @views A[(n + c + 1):(N - n - c), (n + 1):(M - n)] == I

        u = rand(M)
        v1 = similar(u, N)
        v2 = similar(v1)

        # Test custom matrix-vector multiplication.
        # The second version uses the getindex() function to build a regular
        # sparse matrix, which is less efficient.
        mul!(v1, A, u)
        mul!(v2, sparse(A), u)
        @test v1 ≈ v2

        # Test left division.
        let v = copy(v1)
            u2 = A \ v
            @test u2 ≈ u

            # Modify coefficients of original basis near the border, so that the
            # resulting function has no representation in the recombined basis.
            rand!(v)
            @test_throws NoUniqueSolutionError A \ v
        end

        # Test 5-argument mul!.
        mul!(v2, A, u, 1, 0)  # equivalent to 3-argument mul!
        @test v1 ≈ v2

        randn!(u)  # use different u, otherwise A * u = v1
        mul!(v2, A, u, 2, -3)
        @test v2 ≈ 2 * (A * u) - 3 * v1
    end

    nothing
end

function test_boundary_conditions(R::RecombinedBSplineBasis)
    bl, br = constraints(R)
    @assert bl === br "different BCs on each boundary not yet supported"
    test_boundary_conditions(bl, R)
end

function test_boundary_conditions(ops::Tuple{Vararg{Derivative}},
                                  R::RecombinedBSplineBasis)
    # TODO adapt this for different BCs on the left and right boundaries
    @assert (ops, ops) === constraints(R)
    @test length(ops) == num_constraints(R)[1] == num_constraints(R)[2]
    @test length(R) == length(parent(R)) - 2 * length(ops)

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
        # We could use `dot` here, but it doesn't make a difference for
        # single derivatives. It does make a difference for linear combinations
        # of derivatives (e.g. Robin BCs).
        bsum = sum(1:N) do i
            f = R[i]
            fa = f(a, Derivative(n))
            fb = f(b, Derivative(n))
            abs(fa) + abs(fb)  # each of these must be 0 if n = D
        end
        # The sum must be zero if and only if n = D.
        # We consider that the result is zero, if it is negligible wrt the
        # derivative at the border of the first B-spline of the original
        # basis.
        B = parent(R)
        ε = 2 * num_recombined(R)[1] * eps(B[1](a, Derivative(n)))
        if Derivative(n) ∈ ops
            @test bsum ≤ ε
        else
            @test bsum > 1
        end
    end

    nothing
end

# More general BCs (Robin-like BCs and more combinations).
function test_boundary_conditions(ops, R::RecombinedBSplineBasis)
    @assert (ops, ops) === constraints(R)
    @test length(ops) == num_constraints(R)[1] == num_constraints(R)[2]
    @test length(R) == length(parent(R)) - 2 * length(ops)

    a, b = boundaries(R)
    N = length(R)

    for op in ops
        op_a, op_b = dot.(op, (LeftNormal(), RightNormal()))
        bsum = sum(1:N) do i
            f = R[i]
            fa = f(a, op_a)
            fb = f(b, op_b)
            abs(fa) + abs(fb)
        end
        B = parent(R)
        ε = 2 * num_recombined(R)[1] * eps(B[1](a, op_a))
        @test bsum ≤ ε
    end

    nothing
end

# Test relation between `support` and `nonzero_in_segment` (they're inverse of
# each other, in some sense...).
function test_support(R)
    for i in eachindex(R)
        sup = support(R, i)
        for n in sup[1:end-1]
            i ∈ nonzero_in_segment(R, n) || return false
        end
    end
    true
end

# Verify that basis recombination gives the right boundary conditions.
function test_basis_recombination()
    knots_base = gauss_lobatto_points(40)
    k = 4
    B = BSplineBasis(k, knots_base)
    @test constraints(B) === ((), ())
    @testset "Mixed derivatives" begin
        M01 = @inferred RecombineMatrix(B, Derivative.((0, 1)))
        M02 = @inferred RecombineMatrix(B, Derivative.((0, 2)))
        @test M01 == @inferred RecombineMatrix(B, Derivative(0:1))

        # Derivative orders must be in increasing order.
        @test_throws ArgumentError RecombineMatrix(B, Derivative.((1, 0)))

        # Unsupported combinations of derivatives.
        @test_throws ArgumentError RecombineMatrix(B, Derivative.((1, 2)))

        M = RecombineMatrix(B, Derivative.((0, 1)))
        test_recombine_matrix(M)
    end

    @testset "Element type" begin
        M = @inferred RecombineMatrix(B, Derivative(1), Float32)
        @test eltype(M) === Float32
    end

    @test_throws ArgumentError RecombinedBSplineBasis(B, Derivative(k))

    @testset "Spline" begin
        R = RecombinedBSplineBasis(B, Derivative(1))
        coefs = rand(length(R))

        @testset "Equality" begin
            @test R != B
            @test R != RecombinedBSplineBasis(B, Derivative(0))
            @test R == RecombinedBSplineBasis(B, Derivative(1))
        end

        # This constructs a Spline using the recombined basis.
        S = @inferred Spline(R, coefs)
        @test S isa Spline

        @test coefficients(S) === coefs
        @test length(S) == length(R)
        @test basis(S) === R

        @test @allocated(Recombinations.parent_coefficients(R, coefs)) == 0

        # Construct a Spline in the original B-spline basis.
        @test @allocated(Splines.parent_spline(S)) == 0
        Sp = Splines.parent_spline(S)
        @test coefficients(Sp) == recombination_matrix(R) * coefs
        @test length(Sp) == length(B)
        @test basis(Sp) === B

        # Check that both splines are exactly the same
        @test Sp(0.4242) == S(0.4242)
        @test diff(Sp) == diff(S)  # this always returns a Spline in the original basis
        @test integral(Sp) == integral(S)  # same as above
    end

    @testset "Order $D" for D = 0:(k - 1)
        R = @inferred RecombinedBSplineBasis(B, Derivative(D))
        ops = (Derivative(D), )
        @test constraints(R) === (ops, ops)

        δl, δr = @inferred num_constraints(R)
        @test δl == δr == 1  # only one BC per boundary

        # In segment [t[k + 1], t[k + 2]], non-zero recombined functions are 1:k.
        @test @inferred(nonzero_in_segment(R, k + 1)) == 1:k

        # Near the boundaries, there are fewer recombined functions.
        @test nonzero_in_segment(R, k) == 1:(k - 1)
        @test nonzero_in_segment(R, k - 1) == 1:(k - 2)
        @test nonzero_in_segment(R, 2) == 1:1
        @test isempty(nonzero_in_segment(R, 1))

        N = length(parent(R))
        Nr = length(R)
        @test isempty(nonzero_in_segment(R, N + k - δr))
        @test nonzero_in_segment(R, N + k - 1 - δr) == Nr:Nr
        @test test_support(R)

        test_recombine_matrix(recombination_matrix(R))
        test_boundary_conditions(R)

        let
            N = length(R)
            a, b = boundaries(R)
            cl, cr = constraints(R)
            for f ∈ (repr, summary)
                @test startswith(
                    f(R), "$N-element RecombinedBSplineBasis of order $k",
                )
            end
        end

        @testset "Robin-like BCs" begin
            op = 1.1 * Derivative(D) + 4.2 * Derivative(D + 1)
            @test BSplineKit.max_order(op) == D + 1
            if D + 1 == k
                # "cannot resolve operators (...) with B-splines of order 4"
                @test_throws ArgumentError RecombinedBSplineBasis(B, op)
            else
                let Rs = RecombinedBSplineBasis(B, op)
                    @test constraints(Rs) === ((op, ), (op, ))
                    test_recombine_matrix(recombination_matrix(Rs))
                    test_boundary_conditions(Rs)
                end
                if D != 0
                    # Combine with Dirichlet BCs
                    ops = (Derivative(0), op)
                    Rs = RecombinedBSplineBasis(B, ops)
                    @test constraints(Rs) === (ops, ops)
                    test_recombine_matrix(recombination_matrix(Rs))
                    test_boundary_conditions(Rs)
                end
            end
        end

        # Simultaneously satisfies BCs of orders 0 to D.
        @testset "Mixed BCs" begin
            ops = ntuple(d -> Derivative(d - 1), D + 1)
            Rs = RecombinedBSplineBasis(B, ops)
            @test constraints(Rs) === (ops, ops)
            test_recombine_matrix(recombination_matrix(Rs))
            test_boundary_conditions(Rs)
        end
    end
    nothing
end

function test_natural_recombination()
    knots_base = gauss_lobatto_points(40)
    B = BSplineBasis(BSplineOrder(6), knots_base)
    R = RecombinedBSplineBasis(B, Natural())
    M = recombination_matrix(R)
    test_nzrows(M)
    nothing
end

# Matrix conditioning issue fixed in #73
function test_natural_recombination_conditioning()
    xdata = [
        0.1825841576261027
        0.21745398548214934
        0.2574540006430478
        0.37073734805196445
        0.3949709205453995
        0.41683805923570205
        0.4669292800550542
        0.5484125029182643
        0.8956857209308366
        0.9088143561930421
    ]
    ydata = [
        -0.8662273184639824
        1.4644999406051766
        -2.4972734631605986
        -0.579356630251197
        0.9412609863223902
        0.843508283506621
        0.798971854937239
        3.0575214498226995
        -0.05574466652240026
        -0.10031429900815753
    ]
    itp = interpolate(xdata, ydata, BSplineOrder(6), Natural())  # used to fail with assertion error
    @test itp.(xdata) ≈ ydata
    @test (Derivative(2) * itp)(xdata[begin]) < 1e-10
    @test (Derivative(2) * itp)(xdata[end]) < 1e-10
    nothing
end

@testset "Basis recombination" begin
    test_basis_recombination()
    @testset "Natural BCs" begin
        test_natural_recombination()
        test_natural_recombination_conditioning()
    end
    @testset "Hybrid BCs" begin
        B = BSplineBasis(BSplineOrder(4), 0:0.1:1)
        R = @inferred RecombinedBSplineBasis(B, Derivative(0), Derivative(1))
        cs = @inferred constraints(R)
        @test cs[1] === (Derivative(0),)  # left BCs
        @test cs[2] === (Derivative(1),)  # right BCs
        coefs = sqrt.(1:length(R))
        S = Spline(R, coefs)
        S′ = Derivative() * S
        @test S(0.0) == 0   # Dirichlet condition
        @test S(1.0) > 0.1  # no condition (but all coefs are positive)
        @test S′(0.0) > 0.1 # no condition (but function quickly increases from 0 to 1, so positive derivative)
        @test S′(1.0) == 0  # Neumann condition
    end
end
