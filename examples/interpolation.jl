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
ys = sinpi.(xs) .+ 0.02 .* randn(rng, Ndata)

# Let's start by plotting the generated data:

using CairoMakie
CairoMakie.activate!(type = "svg")  # hide

scatter(xs, ys; label = "Data")

# To interpolate the data using splines, we may choose any arbitrary B-spline
# order ``k`` (in particular, ``k = 4`` corresponds to cubic splines).
# The main interpolation function is [`interpolate`](@ref).

using BSplineKit

S = interpolate(xs, ys, BSplineOrder(4))

# Let's plot the result:

lines!(0..1, x -> S(x); label = "k = 4", color = Cycled(4 - 3))

# We can also choose other interpolation orders for comparison:

for k ∈ (5, 6, 8)
    S = interpolate(xs, ys, BSplineOrder(k))
    lines!(0..1, x -> S(x); label = "k = $k", color = Cycled(k - 3))
end

axislegend()
current_figure()  # hide

# We see larger and larger oscillations, especially near the boundaries, as the
# spline order increases.
# We can fix it by using **natural splines**, which impose some derivatives to
# be zero at the boundaries.

# ## Natural splines
#
# [Natural splines](https://en.wikipedia.org/wiki/Spline_(mathematics)#Examples)
# typically refer to cubic splines (order ``k = 4``) with the constraint
# ``S''(a) = S''(b) = 0`` at the boundaries ``x ∈ \{a, b\}``.
#
# In BSplineKit, this concept is generalised for all even spline orders ``k``,
# by setting all derivatives of order ``2, 3, …, k / 2`` to be zero at the
# boundaries.
# For instance, for ``k = 6`` (quintic splines), the constraint is ``S'' = S''' = 0``.
#
# BSplineKit achieves this by using [basis recombination](@ref basis-recombination-api)
# to implicitly impose the wanted boundary conditions.
#
# The natural boundary condition not only allows to suppress oscillations near
# the boundaries, but is also quite convenient for interpolations, as it reduces
# the number of degrees of freedom such that the number of unique B-spline knots
# can be equal to the number of data points.
# As a result, one can set the knots to be equal to the data points.
# All of this is done internally when using this boundary condition.
#
# To impose natural boundary conditions, one just needs to pass
# [`Natural`](@ref) to `interpolate`, as illustrated below.

k = 6
S = interpolate(xs, ys, BSplineOrder(k))  # without BCs
Snat = interpolate(xs, ys, BSplineOrder(k), Natural())  # with natural BCs

# Let's look at the result:

scatter(xs, ys; label = "Data")
lines!(0..1, x -> S(x); label = "k = $k (original)", linewidth = 2)
lines!(0..1, x -> Snat(x); label = "k = $k (natural)", linestyle = :dash, linewidth = 4)
axislegend()
current_figure()  # hide

# Clearly, the spurious oscillations are strongly suppressed near the
# boundaries.
