# # Heat equation
#
# In this example, we numerically solve the 1D [heat
# equation](https://en.wikipedia.org/wiki/Heat_equation)
#
# ```math
# \frac{∂θ}{∂t} = ν \frac{∂^2 θ}{∂x^2},
# ```
#
# in a bounded domain ``x ∈ [-1, 1]`` with homogeneous Neumann boundary
# conditions, ``∂_x θ(±1, t) = 0``.

# ## Defining a B-spline basis

# The general idea is to approximate the unknown solution by a spline of order
# ``k``.
# For this, we first define a B-spline basis ``\{ b_i(x); \, i = 1, …, N \}``, such
# that the solution at a given time $t$ is approximated by
#
# ```math
# θ(x, t) = ∑_{i = 1}^N v_i(t) b_i(x),
# ```
#
# where the $v_i$ are the B-spline coefficients.
#
# A B-spline basis is uniquely defined by its order ``k`` and by a choice of
# *knot* locations within the spatial domain, which form the spatial grid.

# For this example, we take a uniform repartition of knots in ``[-1, 1]``.
knots_in = range(-1, 1, length=11)

# We then create a B-spline basis of order ``k = 4`` using these knots.
using BSplineKit
B = BSplineBasis(BSplineOrder(4), knots_in)

# Note that the generated basis includes an *augmented* set of knots, in which
# each boundary is repeated ``k`` times:

knots(B)

# In other words, the boundary knots have multiplicity ``k``, while interior
# knots have multiplicity 1.
# This is common practice in bounded domains, and translates the fact that the
# solution does not need to be continuous at the boundaries.
# This provides additional degrees of freedom notably for the boundary
# conditions.
# This behaviour can be disabled via the `augment` argument of
# [`BSplineBasis`](@ref).

# We can now plot the knot locations (crosses) and the generated B-spline basis:

using CairoMakie
CairoMakie.activate!(type = "svg")

function plot_knots!(ax, ts; knot_offset = 0.03)
    ys = zero(ts)
    ## Add offset to distinguish knots with multiplicity > 1
    for i in eachindex(ts)[(begin + 1):end]
        if ts[i] == ts[i - 1]
            ys[i] = ys[i - 1] + knot_offset
        end
    end
    scatter!(ax, ts, ys; marker = :x, markersize = 20, color = :gray)
    ax
end

function plot_basis!(ax, B; eval_args = (), kws...)
    cmap = cgrad(:tab20)
    N = length(B)
    for (n, bi) in enumerate(B)
        color = cmap[(n - 1) / (N - 1)]
        lines!(ax, -1..1, x -> bi(x, eval_args...); color, linewidth = 2.5)
    end
    plot_knots!(ax, knots(B); kws...)
    ax
end

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "bᵢ(x)")
plot_basis!(ax, B)
fig

# ## Imposing boundary conditions

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

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "ϕᵢ(x)")
plot_basis!(ax, R)
fig

# We notice that, on each of the two boundaries, the two initial (or final)
# B-splines of the original basis have been combined to produce a single basis
# function that has zero derivative at each respective boundary.
# To verify this, we can plot the basis function derivatives:

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "ϕᵢ′(x)")
plot_basis!(ax, R; eval_args = (Derivative(1), ), knot_offset = 0.4)
fig

# Note that the new basis has two less functions than the original one,
# reflecting a loss of two degrees of freedom corresponding to the new
# constraints on each boundary:

length(B), length(R)

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
# can be obtained via the matrix-vector multiplication ``\bm{v} = \mathbf{T} \bm{u}``.

# ## Initial condition

# We come back now to the heat equation.
# We want to impose the following initial condition:
#
# ```math
# θ(x, 0) = θ_0(x) = 1 + \cos(π x).
# ```

# First, we want to approximate this initial condition in the recombined
# B-spline basis that we have just constructed.
# This may be easily done using [`approximate`](@ref), which interpolates the
# function by evaluating it over a discrete set of interpolation points.

θ₀(x) = 1 + cos(π * x)
θ₀_spline = approximate(θ₀, R)

# Note that the interpolation points don't include the boundaries, since there
# the boundary conditions are exactly satisfied by the spline approximation.

# To see that everything went well, we can plot the exact initial condition and
# its spline approximation, which show no important differences.

fig = Figure(resolution = (800, 400))
let ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "θ")
    lines!(ax, -1..1, θ₀; label = "θ₀(x)", color = :blue)
    lines!(ax, -1..1, x -> θ₀_spline(x); label = "Approximation", color = :orange, linestyle = :dash)
    axislegend(ax; position = :cb)
end
let ax = Axis(fig[1, 2]; xlabel = "x", ylabel = "Difference")
    lines!(ax, -1..1, x -> θ₀(x) - θ₀_spline(x))
end
fig

# Note that we have access to the B-spline coefficients ``v_i`` associated to
# the initial condition, which we will use further below:

v_init = coefficients(θ₀_spline)

# ## Expanding the solution
#
# To solve the governing equation, the strategy is to project the unknown
# solution onto the chosen recombined basis.
# That is, we approximate the solution as
#
# ```math
# θ(x, t) = \sum_{j = 1}^M u_j(t) \, ϕ_j(x).
# ```
#
# Plugging this representation into the heat equation, we find
#
# ```math
# \newcommand{\dd}{\mathrm{d}}
# ∑_j \frac{\dd θ_j}{\dd t} \, ϕ_j(x) = ∑_j θ_j \, ϕ_j''(x),
# ```
#
# where primes denote spatial derivatives.
#
# We thus have ``M`` unknowns $θ_j$.
# We can use the [method of mean weighted
# residuals](https://en.wikipedia.org/wiki/Method_of_mean_weighted_residuals) to
# find the coefficients $θ_j$, by projecting the above equation onto a chosen
# set of *test* functions $φ_i$:
#
# ```math
# ∑_j \frac{\mathrm{d} θ_j}{\mathrm{d} t} \, ⟨φ_i, ϕ_j⟩ = ∑_j θ_j \, ⟨φ_i, ϕ_j''⟩,
# ```
#
# where ``⟨ f, g ⟩ = ∫_0^1 f(x) \, g(x) \, \mathrm{d} x`` is the inner product
# between functions.
#
# Two of the most common choices of test functions are:
#
# - ``φ_i(x) = δ(x - x_i)``, where ``δ`` is Dirac's delta, and ``x_i`` are a set of
#   *collocation* points where the equation will be satisfied.
#   This approach is known as the **collocation** method.
#
# - ``φ_i(x) = ϕ_i(x)``, in which case this is the **Galerkin** method.
#
# We describe the solution using both methods in the following.
