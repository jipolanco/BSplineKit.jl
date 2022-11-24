# # [Function approximation](@id function-approximation-example)
#
# The objective of this example is to approximate a known function ``f`` by a
# spline.
#
# ## Exact function
#
# We consider the function ``f(x) = e^{-x} \cos(8πx)`` in the interval
# ``x ∈ [0, 1]``.

using CairoMakie
CairoMakie.activate!(type = "svg")

x_interval = 0..1
f(x) = exp(-x) * cospi(8x)

fig = Figure(resolution = (800, 600))
ax = Axis(fig[1, 1]; xlabel = "x")
lines!(ax, x_interval, f)
fig

# ## Approximation space

# To approximate this function using a spline, we first need to define a
# B-spline basis ``\{ b_i \}_{i = 1}^N`` describing a spline space.
# The approximating spline can then be written as
#
# ```math
# g(x) = ∑_{i = 1}^N c_i b_i(x),
# ```
#
# where the ``c_i`` are the B-spline coefficients describing the spline.
# The objective is thus to find the coefficients ``c_i`` that result in the best
# possible approximation of the function ``f``.
#
# Here we use splines of order ``k = 4`` (polynomial degree ``d = 3``, i.e.
# cubic splines).
# For simplicity, we choose the B-spline knots to be uniformly distributed.

using BSplineKit

ξs = range(x_interval; length = 15)
B = BSplineBasis(BSplineOrder(4), ξs)

# We plot below the knots and the basis functions describing the spline space.
# Note that knots are represented by grey crosses.

function plot_knots!(ax, ts; ybase = 0, knot_offset = 0.03, kws...)
    ys = zero(ts) .+ ybase
    ## Add offset to distinguish knots with multiplicity > 1
    if knot_offset !== nothing
        for i in eachindex(ts)[(begin + 1):end]
            if ts[i] == ts[i - 1]
                ys[i] = ys[i - 1] + knot_offset
            end
        end
    end
    scatter!(ax, ts, ys; marker = '×', color = :gray, markersize = 24, kws...)
    ax
end

function plot_basis!(ax, B; eval_args = (), kws...)
    cmap = cgrad(:tab20)
    N = length(B)
    ts = knots(B)
    hlines!(ax, 0; color = :gray)
    for (n, bi) in enumerate(B)
        color = cmap[(n - 1) / (N - 1)]
        i, j = extrema(support(bi))
        lines!(ax, ts[i]..ts[j], x -> bi(x, eval_args...); color, linewidth = 2.5)
    end
    plot_knots!(ax, ts; kws...)
    ax
end

fig = Figure(resolution = (800, 600))
ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "bᵢ(x)")
plot_basis!(ax, B; knot_offset = 0.05)
fig

# ## Approximating the function
#
# Three different methods are implemented in BSplineKit to approximate
# functions.
# In increasing order of accuracy and complexity, these are:
#
# ### 1. [`VariationDiminishing`](@ref)
#
# Implements Schoenberg's variation diminishing approximation.
# This simply consists on estimating the spline coefficients as ``c_i =
# f(x_i)``, where the ``x_i`` are the Greville sites.
# These are obtained by window-averaging the B-spline knots ``t_j``:
#
# ```math
# x_i = \frac{1}{k - 1} ∑_{j = 1}^{k - 1} t_{i + j}.
# ```
#
# This approximation is expected to preserve the shape of the function.
# However, as shown below, it is usually very inaccurate as an actual
# approximation, and should only be used when a qualitative estimation of ``f``
# is sufficient.

S_vd = approximate(f, B, VariationDiminishing())

# ### 2. [`ApproxByInterpolation`](@ref)
#
# Approximates the original function by interpolating on a discrete set of
# interpolation points.
# In other words, the resulting spline exactly matches ``f`` at those points.
#
# By default, the interpolation points are chosen as the Greville sites
# associated to the B-spline basis (using [`collocation_points`](@ref); see also
# [`Collocation.AvgKnots`](@ref)).
# For more control, the interpolation points may also be directly set via the
# [`ApproxByInterpolation`](@ref) constructor.
#
# In the below example, we pass the B-spline basis to the
# `ApproxByInterpolation` constructor, which automatically determines the
# collocation points as explained above.

S_interp = approximate(f, B, ApproxByInterpolation(B))  # or simply approximate(f, B)

# ### 3. [`MinimiseL2Error`](@ref)
#
# Approximates the function by minimising the ``L^2`` distance between ``f`` and
# its spline approximation ``g``.
#
# In other words, it minimises
# ```math
# \mathcal{L}[g] = {\left\lVert f - g \right\rVert}^2 = \left< f - g, f - g \right>,
# ```
# where
# ```math
# \left< u, v \right> = ∫_a^b u(x) \, v(x) \, \mathrm{d}x
# ```
# is the inner product between two functions, and ``a`` and ``b`` are the
# boundaries of the prescribed B-spline basis.
#
# One can show that the optimal coefficients ``c_i`` minimising the ``L^2`` error
# are the solution to the linear system ``\bm{M} \bm{c} = \bm{φ}``,
# where ``M_{ij} = \left< b_i, b_j \right>`` and ``φ_i = \left< b_i, f \right>``.
# These two terms are respectively computed by [`galerkin_matrix`](@ref) and
# [`galerkin_projection`](@ref).
#
# Indeed, this can be shown by taking the differential
# ```math
# δ\mathcal{L}[g] = \mathcal{L}[g + δg] - \mathcal{L}[g]
# = 2 \left< δg, g - f \right>,
# ```
# where ``δg`` is a small perturbation of the spline ``g``.
# The optimal spline ``g^*``, minimising the ``L^2`` distance, is such that
# ``δ\mathcal{L}[g^*] = 0``.
#
# Noting that ``g = c_i b_i`` (where summing is implicitly performed over
# repeated indices), the perturbation is given by ``δg = δc_i b_i``, as the
# B-spline basis is assumed fixed.
# The optimal spline then satisfies
# ```math
# \left< b_i, g^* - f \right> δc_i
# = \left[ \left< b_i, b_j \right> c_j^* - \left< b_i, f \right> \right] δc_i
# = \left[ M_{ij} c_j^* - φ_i \right] δc_i
# = 0
# ```
# for all perturbations ``δ\bm{c}``, leading to the linear system stated above.
#
# As detailed in [`galerkin_projection`](@ref), integrals are computed via
# Gauss--Legendre quadratures, in a way that ensures that the result is exact
# when ``f`` is a polynomial of degree up to ``k - 1`` (or more generally, a
# spline belonging to the space spanned by the chosen B-spline basis).

S_minL2 = approximate(f, B, MinimiseL2Error())

# ## Method comparison
#
# Below, the approximations using the three methods are compared to the actual
# function ``f``.

fig = Figure(resolution = (1000, 750))
colours = theme(fig.scene).palette.color[]
style_vd = (color = colours[3], label = "Variation diminishing")
style_interp = (color = colours[2], label = "Interpolation")
style_minL2 = (color = colours[1], label = "L² minimisation")
let ax = Axis(fig[1:2, 1]; xlabel = "x", ylabel = "Approximation")
    plot_knots!(ax, knots(B); knot_offset = nothing)
    lines!(ax, x_interval, f; color = :black, linewidth = 2, label = "Original")
    lines!(ax, x_interval, x -> S_vd(x); style_vd...)
    lines!(ax, x_interval, x -> S_interp(x); style_interp...)
    lines!(ax, x_interval, x -> S_minL2(x); style_minL2...)
    axislegend(ax)
end
let ax = Axis(fig[1, 2]; ylabel = "Difference with original")
    plot_knots!(ax, knots(B); knot_offset = nothing)
    lines!(ax, x_interval, x -> S_interp(x) - f(x); style_interp...)
    lines!(ax, x_interval, x -> S_minL2(x) - f(x); style_minL2...)
    hidexdecorations!(ax; grid = false)
    axislegend(ax; position = :rt, orientation = :horizontal)
end
let ax = Axis(fig[2, 2]; xlabel = "x", ylabel = "Squared difference", yscale = log10)
    ylims!(1e-8, 1e-2)
    plot_knots!(ax, knots(B); knot_offset = nothing, ybase = 1e-6)
    lines!(ax, x_interval, x -> abs2(S_interp(x) - f(x)); style_interp...)
    lines!(ax, x_interval, x -> abs2(S_minL2(x) - f(x)); style_minL2...)
end
fig

# As seen above, the **variation diminishing approximation**, while capturing the
# shape of the original function, doesn't really provide an accurate
# approximation of it.
#
# The other two methods are much more accurate.
# On the right half of the figure, a detailed comparison of the two is provided,
# by plotting the difference between each approximation and the actual ``f``
# function.
#
# The **interpolation method** works pretty well, matching exactly the actual
# function at the interpolation points.
# Note that, in this example, most interpolation points match the spline knots.
# This is because we're using splines of even degree (``k = 4``) and because
# knots are uniformly spaced.
#
# Nevertheless, when looking at the global error, the **``L^2`` minimisation
# method** works best, as expected.
# In particular, as seen above, it reduces the maximum approximation error (i.e.
# the ``L^∞`` distance, ``{\left\lVert f - g \right\rVert}_∞ = \max |f(x) -
# g(x)|``) compared to the interpolation approach.
