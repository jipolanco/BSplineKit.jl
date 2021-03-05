using BSplineKit: BSplineOrder
using BSplineKit.BSplines: multiplicity
using LinearAlgebra

eval_poly(x::AbstractVector, P) = [@evalpoly(t, P...) for t in x]

# Test a polynomial of degree k - 1.
# The splines should approximate the polynomial (and its derivatives) perfectly.
# This is also used to test the SplineInterpolations module.
function test_polynomial(x, ::BSplineOrder{k}) where {k}
    # Coefficients of polynomial of degree k - 1 (see ?evalpoly).
    # P(x) = -1 + 2x - 3x^2 + 4x^3 - ...
    P = ntuple(d -> (-d)^d, Val(k))
    P′ = ntuple(d -> d * P[d + 1], Val(k - 1))    # derivative
    Pint = (0, ntuple(d -> P[d] / d, Val(k))...)  # antiderivative

    # Interpolate polynomial at `x` locations.
    y = eval_poly(x, P)
    itp = @inferred interpolate(x, y, BSplineOrder(k))
    @test itp.(x) ≈ y

    S = spline(itp)
    @test length(S) == length(x)

    let x = (x[2] + x[3]) / 3
        @test itp(x) == S(x)  # these are equivalent
    end

    # "incompatible lengths of B-spline basis and collocation points"
    @test_throws(DimensionMismatch,
                 SplineInterpolations.Interpolation(basis(S), x[1:4], eltype(S)))

    # "input data has incorrect length"
    @test_throws DimensionMismatch interpolate!(itp, rand(length(x) - 1))

    S′ = diff(S, Derivative(1))
    Sint = integral(S)

    a, b = boundaries(basis(S))

    @test Sint(a) == 0  # this is an arbitrary choice
    Pint_a = @evalpoly(a, Pint...)

    # Compare values on a finer grid.
    let Nx = 9 * length(S) + 42
        x = LinRange(a, b, Nx)
        @test all(@evalpoly(x, P...) ≈ S(x) for x in x)
        @test all(@evalpoly(x, P′...) ≈ S′(x) for x in x)
        @test all(@evalpoly(x, Pint...) - Pint_a ≈ Sint(x) for x in x)
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

        @test_throws BoundsError B[0]
        @test_throws BoundsError B[N + 1]

        # In segment [t[k + 1], t[k + 2]], non-zero B-splines are 2:(k + 1).
        @test @inferred(nonzero_in_segment(B, k + 1)) == 2:(k + 1)

        # Fewer B-splines near the boundaries. This is usually not a problem
        # because knots are repeated there...
        @test nonzero_in_segment(B, 1) == 1:1
        @test nonzero_in_segment(B, N) == (N - k + 1):N
        @test nonzero_in_segment(B, N + 1) == (N - k + 2):N
        @test nonzero_in_segment(B, N + k - 1) == N:N
        @test isempty(nonzero_in_segment(B, N + k))

        # These intervals are not defined
        @test isempty(nonzero_in_segment(B, 0))
        @test isempty(nonzero_in_segment(B, length(knots(B))))

        # Verify values at the boundaries.
        a, b = boundaries(B)
        @test evaluate(B, 1, a) == 1.0
        @test evaluate(B, N, b) == 1.0

        @test string(B) == "$N-element BSplineBasis: order $k, domain [$a, $b]"

        xs = range(a, b; length = 4N + 1)
        us = similar(xs)
        vs = similar(us)
        i = N >> 2
        evaluate!(us, B, i, xs)
        @test us == evaluate(B, i, xs)
        @test us == BasisFunction(B, i).(xs)
        @test us == B[i].(xs)
        @test us == map!(B[i], vs, xs)
        @test us ≈ B[i, Float32].(xs)

        @testset "iterate" begin
            for (i, b) in enumerate(B)
                @test b === B[i]
            end
        end
    end

    xcol = collocation_points(B, method=Collocation.AvgKnots())
    @test xcol[1] == knots_in[1]
    @test xcol[end] == knots_in[end]

    C = @inferred collocation_matrix(B, xcol)

    let N = size(C, 1)
        T = eltype(C)
        l, u = bandwidths(C)
        @test startswith(
            sprint(show, MIME("text/plain"), C),
            "$N×$N CollocationMatrix{$T} with bandwidths ($l, $u):\n",
        )
    end

    @testset "LU factorisation" begin
        T = eltype(C)
        @test T === Float64
        @test C isa CollocationMatrix{T}
        F = @inferred factorize(C)
        @test F isa LU{T, <:CollocationMatrix{T}}
        @test F.L * F.U ≈ C
        @test F.P == I  # permutation matrix = identity matrix (no pivoting)
        @test F.p == 1:size(C, 1)  # no row permutations

        y = sin.(xcol)
        u = F \ y
        @test C * u ≈ y

        let v = similar(u, length(u) - 2)
            @test_throws DimensionMismatch ldiv!(v, F, y)
        end
    end

    @testset "Spline (k = $k)" begin
        @testset "Polynomial" begin
            test_polynomial(xcol, BSplineOrder(k))
        end

        # Generate data at collocation points and get B-spline coefficients.
        ucol = cos.(xcol)
        coefs = C \ ucol
        @test C * coefs ≈ ucol

        @inferred Spline(B, coefs)
        S = Spline(B, coefs)
        @test length(S) == length(B)
        @test all(S.(xcol) .≈ ucol)
        @test coefficients(S) === coefs
        @test diff(S, Derivative(0)) === S
        @test_nowarn show(devnull, S)

        # Broadcasting
        f(S, x) = S(x)
        @test f.(S, xcol) == S.(xcol)

        # Create new spline, then compare it to S.
        let P = Spline(undef, B)
            cp = coefficients(P)
            fill!(cp, 0)
            @test P != S
            @test !(P ≈ S)

            copy!(cp, coefs)  # copy coefficients of S
            @test P == S
            @test P ≈ S

            @test copy(S) == P
        end
    end

    nothing
end

function test_splines(::BSplineOrder{k}) where {k}
    breaks = (
        -1:0.05:1,
        gauss_lobatto_points(10 + k),
    )
    for x in breaks
        @inferred BSplineBasis(BSplineOrder(k), copy(x))
        @inferred BSplineBasis(BSplineOrder(k), copy(x), augment=Val(false))
        @inferred BSplineBasis(BSplineOrder(k), copy(x), augment=Val(true))
        @inferred (() -> BSplineBasis(k, copy(x)))()
        g = BSplineBasis(k, copy(x))

        @testset "BSplineBasis equality" begin
            h = BSplineBasis(k, copy(x))
            @test g == h

            h = BSplineBasis(BSplineOrder(k + 1), copy(x))
            @test g != h

            h = BSplineBasis(k, x .+ 1)
            @test g != h
        end

        @test order(g) == k
        test_splines(g, x)
    end
    nothing
end

@testset "Splines" begin
    test_splines(BSplineOrder(4))
    test_splines(BSplineOrder(5))
end
