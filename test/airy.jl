using BSplineKit
using LinearAlgebra
using QuadGK
using SpecialFunctions: airyai

# Solution of Airy differential equation using Galerkin method.
# Problem taken from Olver & Townsend SIAM Rev. 2013.
# The original equation is for a function `u` that satisfies inhomogeneous
# Dirichlet BCs at x = ±1.
# Here we perform the change of variables v(x) = u(x) - u₀(x), such that `v`
# satisfies *homogeneous* Dirichlet BCs.
function test_airy_equation()
    ε = 1e-4
    N = 100
    breaks = collect(range(-1, 1; length = N + 1))
    B = @inferred BSplineBasis(BSplineOrder(6), breaks)
    R = @inferred RecombinedBSplineBasis(B, Derivative(0))  # for Dirichlet BCs

    # Exact solution
    u_exact(x) = airyai(x / cbrt(ε))

    # Affine part of the solution, for change of variables.
    # This is the (unique) affine curve that satisfies the two inhomogeneous BCs.
    u_0(x) = ((1 - x) * u_exact(-1) + (1 + x) * u_exact(1)) / 2

    # This is to obtain coefficients from data at collocation points
    xs = @inferred collocation_points(B)
    Cfact = @inferred lu!(collocation_matrix(B, xs))

    # Variable coefficient operator: c(x) * v(x), with c(x) = x.
    # Note that we could expand c(x) = x using linear splines (k = 2), but we
    # use the same basis as for `v` for simplicity.
    # Here Mc is the linear Galerkin operator associated to multiplication by
    # c(x).
    Mc = let
        cs = @inferred Cfact \ xs  # coefficients for c(x) = x
        Tc = @inferred galerkin_tensor((R, R, B), Derivative.((0, 0, 0)))
        Mc = @inferred Tc * cs
        @test Mc isa BandedMatrix
        @test Mc ≈ Mc'  # the matrix is practically symmetric
        Hermitian(Mc)
    end

    # Second-order term: ε v''(x)
    # Taking advantage of integration by parts, we construct the matrix
    # M′[i, j] = ⟨ ϕ′[i] ϕ′[j] ⟩.
    # The boundary term is zero due to Dirichlet BCs.
    L = let
        L = @inferred galerkin_matrix(R, Derivative.((1, 1)))
        @test L isa Hermitian
        @test L.uplo == 'U'
        parent(L) .*= ε
        L
    end

    # We add the two LHS operators
    # We only need to add the upper triangular part, because of symmetry.
    parent(L) .+= UpperTriangular(Mc)

    # Factorisation
    # Apparently we can't do Cholesky because `Mc` is not SPD.
    # For LU, we can't pass a Hermitian matrix wrapper...
    Lfact = lu(BandedMatrix(L))

    # RHS resulting from change of variables
    f(x) = x * u_0(x)
    rhs = let
        fs = ldiv!(Cfact, f.(xs))     # coefficients of RHS term
        Mf = galerkin_matrix((R, B))  # RHS operator
        -Mf * fs
    end

    # Numerical solution
    vR = Lfact \ rhs  # coefficients of the solution in recombined basis R
    vB = @inferred recombination_matrix(R) * vR  # same in full basis B
    vsol = @inferred Spline(B, vB)  # solution as a spline
    usol(x) = vsol(x) + u_0(x)  # undo change of variables

    # Check error (this will depend on the choices of ε, N, k, …)
    err, _ = quadgk(x -> (usol(x) - u_exact(x))^2, -1, 1; rtol = 1e-8)
    @test err < 1e-6

    nothing
end

@testset "Airy equation" begin
    test_airy_equation()
end
