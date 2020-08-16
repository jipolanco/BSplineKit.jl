# # Heat equation
#
# In this first example, we numerically solve the [heat
# equation](https://en.wikipedia.org/wiki/Heat_equation)
#
# ```math
# \frac{∂u}{∂t} = ν \frac{∂^2 u}{∂x^2},
# ```
#
# in a bounded domain ``x ∈ [-1, 1]`` with homogeneous Neumann boundary
# conditions, ``∂_x u(±1, t) = 0``.

# ## Initial condition

# We impose the initial condition
#
# ```math
# u(x, 0) = u_0(x) = 1 + \cos(π x).
# ```

using Plots
using LaTeXStrings
plot(x -> 1 + cos(π * x), -1, 1, label=L"u_0(x)", xlabel=L"x", ylabel=L"u")

# ## Defining a B-spline basis

# The general idea is to approximate the unknown solution by a spline of order
# $k$.
# For this, we first need to define a B-spline basis.
# A B-spline basis is uniquely defined by its order $k$ and by a choice of
# *knot* locations within the spatial domain, which form the spatial grid.

# For this example, we take a uniform repartition of knots in ``[-1, 1]``.
knots_in = range(-1, 1, length=11)

# We then create a B-spline basis of order $k = 4$ using these knots.
using BSplineKit
B = BSplineBasis(4, knots_in)

# Note that the generated basis includes an *augmented* set of knots, in which
# each boundary is repeated $k$ times.
# In other words, the boundary knots have multiplicity $k$, while interior
# knots have multiplicity 1.
# This is common practice in bounded domains, and translates the fact that the
# solution does not need to be continuous at the boundaries.
# This provides additional degrees of freedom notably for the boundary
# conditions.
# This behaviour can be disabled via the `augment` argument of
# [`BSplineBasis`](@ref).

# We can now plot the knot locations and the generated B-spline basis:

function plot_basis(B; eval_args=(), kw...)
    plt = plot(; legend=false, xlabel=L"x", kw...)
    let t = knots(B)
        plot!(t, zeros(length(t)), marker=:x, color=:red)
    end
    for i = 1:length(B)
        bi = BSpline(B, i)
        plot!(plt, x -> bi(x, eval_args...), -1, 1, linewidth=1.5)
    end
    plt
end

plot_basis(B, ylabel=L"b_i(x)")

# ## Defining a recombined B-spline basis

# In BSplineKit, the recommended approach for solving boundary value problems is
# to use the basis recombination method.
# That is, to expand the solution onto a new basis consisting on linear
# combinations of B-splines ``b_i(x)``, such that each recombined basis function
# ``ϕ_j(x)`` individually satisfies the required homogeneous boundary conditions
# (BCs).
# Thanks to the local support of B-splines, basis recombination only involves a
# small number of B-splines near the boundaries.
#
# Using the [`RecombinedBSplineBasis`](@ref) type, we can easily define such
# recombined bases for many different BCs.
# In this example we generate a basis satisfying homogeneous Neumann BCs:

R = RecombinedBSplineBasis(Derivative(1), B)
plot_basis(R, ylabel=L"\phi_i(x)")

# We notice that, on each of the two boundaries, the two initial (or final)
# B-splines of the original basis have been combined to produce a single basis
# function that has zero derivative at each respective boundary.
# To verify this, we can plot the basis function derivatives:

plot_basis(R, ylabel=L"\phi_i^{\prime}(x)", eval_args=(Derivative(1), ))

# Note that the new basis has two less functions than the original one,
# reflecting a loss of two degrees of freedom corresponding to the new
# constraints on each boundary:

@assert length(R) == length(B) - 2

# ## Recombination matrix

# As stated above, the basis recombination approach consists in performing
# linear combinations of B-splines ``b_i`` to obtain a derived basis of
# functions ``ϕ_j`` satisfying certain boundary conditions.
# This can be conveniently expressed using a transformation matrix
# ``\mathbf{T}`` relating the two bases:
#
# ```math
# ϕ_j(x) = ∑_{i = 1}^N T_{ij} b_i(x)
# \quad \text{for } j = 1, 2, …, M,
# ```
#
# where ``N`` is the number of B-splines ``b_i``, and ``M < N`` is the number of
# ``ϕ_j`` functions (in this example, ``M = N - 2``).
#
# The recombination matrix associated to the generated basis can be obtained
# using [`recombination_matrix`](@ref):

T = recombination_matrix(R)

# Note that the matrix is almost an identity matrix, since most B-splines are
# kept intact in the new basis.
# This simple structure allows for very efficient computations using this
# matrix.
# The first and last columns indicate that Neumann BCs are imposed by adding the
# two first (and two last) B-splines, i.e.
#
# ```math
# ϕ_1(x) = b_1(x) + b_2(x),
# \qquad
# ϕ_M(x) = b_{N - 1}(x) + b_N(x).
# ```

# Finally, note that the recombination matrix is particularly useful for
# converting between coefficients from the two bases.
# Given a function that has a known representation in the recombined basis,
# ``f(x) = ∑_j u_j \, ϕ_j(x)``, its coefficients ``v_i`` in the original basis
# can be obtained by matrix-vector multiplication as `v = T * u`.

# ## Expanding the solution
#
# To solve the governing equation, the strategy is to project the unknown
# solution onto the chosen recombined basis.
# That is, we write the solution as
#
# ```math
# u(x, t) = \sum_{j = 1}^M u_j(t) \, ϕ_j(x).
# ```
#
# Plugging this representation into the heat equation, we find
#
# ```math
# \newcommand{\dd}{\mathrm{d}}
# ∑_j \frac{\dd u_j}{\dd t} \, ϕ_j(x) = ∑_j u_j \, ϕ_j''(x),
# ```
#
# where primes denote spatial derivatives.
