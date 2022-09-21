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

lines!(0..1, x -> S(x); label = "k = 4", color = Cycled(4 - 3))
current_figure()  # hide

# We can also choose other interpolation orders for comparison:

for k ∈ (5, 6, 8)
    S = interpolate(xs, ys, BSplineOrder(k))
    lines!(0..1, x -> S(x); label = "k = $k", color = Cycled(k - 3))
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
lines!(0..1, x -> S(x); label = "k = $k (original)", linewidth = 2)
lines!(0..1, x -> Snat(x); label = "k = $k (natural)", linestyle = :dash, linewidth = 4)
axislegend()
current_figure()  # hide

# Clearly, the spurious oscillations are strongly suppressed near the
# boundaries.

# ## Multidimensional interpolations
#
# Multidimensional interpolations are supported for data defined on
# rectilinear grids.
# This is done using
# [tensor-product](https://en.wikipedia.org/wiki/Tensor_product) splines.
# This means that the returned $N$-dimensional spline belongs to the space
# spanned by the tensor product of $N$ one-dimensional spline bases.
#
# In other words, if we consider the two-dimensional case for simplicity, the
# returned interpolator is of the form
#
# ```math
# f(x, y) = ∑_{i = 1}^{N_x} ∑_{j = 1}^{N_y}
#       c_{ij} \, b_{i}^X(x) \, b_{j}^Y(y),
# ```
#
# where the ``b_{i}^X`` and ``b_{j}^Y`` are the B-splines spanning two
# (generally different) spline bases ``B^X`` and ``B^Y``.
# Note that the two bases are completely independent.
# In principle, they can have different orders, knot definitions and boundary
# conditions.
#
# All of the above also applies to (arbitrary) higher dimensions ``N``.
# See [`interpolate`](@ref) for more details.
#
# Let's finish with a simple example in 2D:

## Define non-uniform 2D grid
Nx, Ny = 20, 30
rng = MersenneTwister(42)

xs = range(0, 1; length = Nx) .+ 0.01 .* randn(rng, Nx)
xs[begin] = 0
xs[end] = 1

ys = range(0, 2π; length = Ny) .+ 0.02 .* randn(rng, Ny)
ys[begin] = 0
ys[end] = 2π

## Generate some 2D data and interpolate
data = [exp(-x) * cos(y) + 0.01 * randn(rng) for x ∈ xs, y ∈ ys]

S = interpolate((xs, ys), data, BSplineOrder(4), Natural())

## Finally, let's plot the result on a finer grid
xplot = range(0, 1; length = 4Nx)
yplot = range(0, 2π; length = 4Ny)
Sdata = S.(xplot, yplot')

contourf(xplot, yplot, Sdata)
current_figure()  # hide
