using BSplineKit: BSplineOrder
using BSplineKit.BSplines: multiplicity

# Define a polynomial of degree p and its first derivative.
poly(x, ::Val{p}, ::Derivative{0}) where {p} =
    (p - x) * poly(x, Val(p - 1), Derivative(0))
poly(x, ::Val{0}, ::Derivative{0}) = one(x)

function poly(x, ::Val{p}, ::Derivative{1}) where {p}
    -poly(x, Val(p - 1), Derivative(0)) +
    (p - x) * poly(x, Val(p - 1), Derivative(1))
end
poly(x, ::Val{0}, ::Derivative{1}) = zero(x)

# Test a polynomial of degree k - 1.
# The splines should approximate the polynomial (and its derivatives) perfectly.
function test_polynomial(B::BSplineBasis)
    k = order(B)
    P = x -> poly(x, Val(k - 1), Derivative(0))
    P′ = x -> poly(x, Val(k - 1), Derivative(1))

    # Get coefficients using values at collocation points.
    # (TODO: use higher level interpolation function)
    coefs = let x = collocation_points(B)
        C = collocation_matrix(B, x)
        values = P.(x)
        C \ values
    end

    S = Spline(B, coefs)
    S′ = diff(S, Derivative(1))

    # Compare values on a finer grid.
    let Nx = 9 * length(B) + 42
        a, b = boundaries(B)
        x = LinRange(a, b, Nx)
        @test P.(x) ≈ S.(x)
        @test P′.(x) ≈ S′.(x)
    end

    nothing
end

function test_splines(B::BSplineBasis, knots_in)
    k = order(B)
    t = knots(B)

    @testset "Knots (k = $k)" begin
        let (ka, kb) = multiplicity.(Ref(t), (1, length(t)))
            @test ka == kb == k
        end

        @test @views all(t[1:k] .== knots_in[1])
        @test @views all(t[(end - k + 1):end] .== knots_in[end])
        @test @views t[(k + 1):(end - k)] == knots_in[2:(end - 1)]
    end

    @testset "B-splines (k = $k)" begin
        N = length(B)
        @test_throws DomainError evaluate(B, 0, 0.2)
        @test_throws DomainError evaluate(B, N + 1, 0.2)

        # Verify values at the boundaries.
        @test evaluate(B, 1, t[1]) == 1.0
        @test evaluate(B, N, t[end]) == 1.0
    end

    xcol = collocation_points(B, method=Collocation.AvgKnots())
    @test xcol[1] == knots_in[1]
    @test xcol[end] == knots_in[end]

    C = collocation_matrix(B, xcol)

    @testset "Spline (k = $k)" begin
        @testset "Polynomial" begin
            test_polynomial(B)
        end

        # Generate data at collocation points and get B-spline coefficients.
        ucol = cos.(xcol)
        coefs = C \ ucol

        @inferred Spline(B, coefs)
        S = Spline(B, coefs)
        @test all(S.(xcol) .≈ ucol)
        @test coefficients(S) === coefs

        # Create new spline, then compare it to S.
        let P = Spline(B)
            cp = coefficients(P)
            fill!(cp, 0)
            @test P != S
            @test !(P ≈ S)

            copy!(cp, coefs)  # copy coefficients of S
            @test P == S
            @test P ≈ S
        end
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

@testset "Splines" begin
    test_splines(BSplineOrder(4))
    test_splines(BSplineOrder(5))
end
