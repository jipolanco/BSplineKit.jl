# # [Spline interpolations](@id interpolation-example)
#
# ## Interpolating data
#
# BSplineKit can interpolate evenly and unevenly-distributed data.
#
# For example, we can try interpolating random data on randomly-distributed data
# points:

using Random
rng = MersenneTwister(42)

Ndata = 20
xs = range(0, 1; length = Ndata) .+ 0.01 .* randn(rng, Ndata)
sort!(xs)  # make sure coordinates are sorted
xs[begin] = 0; xs[end] = 1;   # not strictly necessary; just to set the data limits
ys = sinpi.(xs) .+ 0.02 .* randn(rng, Ndata);

# Let's start by plotting the generated data:

using CairoMakie
CairoMakie.activate!(type = "svg")  # hide
scatter(xs, ys; label = "Data", color = :black)

# To interpolate the data using splines, we may choose any arbitrary B-spline
# order ``k`` (in particular, ``k = 4`` corresponds to cubic splines).
# The main interpolation function is [`interpolate`](@ref).

using BSplineKit
S = interpolate(xs, ys, BSplineOrder(4))

# Let's plot the result:

lines!(0..1, S; label = "k = 4", color = Cycled(4 - 3))
current_figure()  # hide

# We can also choose other interpolation orders for comparison:

for k ∈ (5, 6, 8)
    S = interpolate(xs, ys, BSplineOrder(k))
    lines!(0..1, S; label = "k = $k", color = Cycled(k - 3))
end
axislegend()
current_figure()  # hide

# We see larger and larger oscillations, especially near the boundaries, as the
# spline order increases.
# We can try to fix this using **natural** splines, which impose some
# derivatives to be zero at the boundaries.

# ## Natural splines
#
# [Natural splines](https://en.wikipedia.org/wiki/Spline_(mathematics)#Examples)
# usually refer to cubic splines (order ``k = 4``) with the additional
# constraint ``S''(a) = S''(b) = 0`` at the boundaries (``x ∈ \{a, b\}``).
#
# In BSplineKit, this concept is generalised for all even spline orders ``k``,
# by setting all derivatives of order ``2, 3, …, k / 2`` to be zero at the
# boundaries.
# For instance, for ``k = 6`` (quintic splines), the constraint is ``S'' = S''' = 0``.
# We achieve this by using [basis recombination](@ref basis-recombination-api)
# to implicitly impose the wanted boundary conditions.
#
# The natural boundary condition not only allows to suppress oscillations near
# the boundaries, but is also quite convenient for interpolations, as it reduces
# the number of degrees of freedom such that the number of unique spline knots
# is equal to the number of data points.
# This simply means that one can set the knots to be equal to the data points.
# All of this is done internally when using this boundary condition.
#
# To impose natural boundary conditions, one just needs to pass
# [`Natural`](@ref) to `interpolate`, as illustrated below.

k = 8
S = interpolate(xs, ys, BSplineOrder(k))  # without BCs
Snat = interpolate(xs, ys, BSplineOrder(k), Natural())  # with natural BCs

# Let's look at the result:

scatter(xs, ys; label = "Data", color = :black)
lines!(0..1, S; label = "k = $k (original)", linewidth = 2)
lines!(0..1, Snat; label = "k = $k (natural)", linestyle = :dash, linewidth = 4)
axislegend()
current_figure()  # hide

# Clearly, the spurious oscillations are strongly suppressed near the
# boundaries.

# ## [Extrapolations](@id extrapolation-example)
#
# One can use extrapolation to evaluate splines outside of their domain of definition.
# A few different extrapolation strategies are implemented in BSplineKit.
# See [Extrapolation methods](@ref) for details.
#
# Below we compare a few possible extrapolation strategies included in BSplineKit.
#
# First, we generate and interpolate new data:

xs = 0:0.2:1
ys = 2 * cospi.(xs)
S = interpolate(xs, ys, BSplineOrder(4))

# One can directly evaluate these interpolations outside of the domain
# ``[0, 1]``, but the result will always be zero:

S(-0.32)

# To enable extrapolations, one must call [`extrapolate`](@ref) with the
# desired extrapolation strategy (see [Extrapolation methods](@ref) for a list).
# Here we compare both [`Flat`](@ref) and [`Smooth`](@ref) methods:

E_flat   = extrapolate(S, Flat())
E_smooth = extrapolate(S, Smooth())

#

fig = Figure(size = (600, 400))
ax = Axis(fig[1, 1])
scatter!(ax, xs, ys; label = "Data", color = :black)
lines!(ax, -0.5..1.5, S; label = "No extrapolation", linewidth = 2)
lines!(ax, -0.5..1.5, E_smooth; label = "Smooth", linestyle = :dash, linewidth = 2)
lines!(ax, -0.5..1.5, E_flat; label = "Flat", linestyle = :dot, linewidth = 2)
axislegend(ax)
fig

