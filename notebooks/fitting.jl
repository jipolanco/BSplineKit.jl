### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 4269cea4-79ac-11eb-20fb-6d35207dbbf8
begin
	using Revise
	using LinearAlgebra
	using BSplineKit
	using SparseArrays
	using Plots
end

# ╔═╡ 5a37f162-79ae-11eb-18fe-b329714a6617
f(x) = exp(-x / 2π) * cos(4x)

# ╔═╡ 6899fcb2-79c9-11eb-0d95-cb916a6751d3
xlims = 0, 2π

# ╔═╡ 4476d184-79ae-11eb-135f-af80be0c9bde
xdata = range(xlims...; length = 512 + 1)

# ╔═╡ 911d13ee-79b1-11eb-33b4-5d7885069097
xbreaks = range(xlims...; length = 32 + 1)

# ╔═╡ 81a5efec-79ae-11eb-01f1-31bc96f18a6d
ydata = map(x -> f(x) + 0.5 * randn(), xdata)

# ╔═╡ 22639830-79af-11eb-3834-fd118a685500
B0 = BSplineBasis(BSplineOrder(4), xbreaks)

# ╔═╡ 823cb81a-79c6-11eb-1e4d-75eb02469abf
# Impose u' = 0 at the boundaries
B = RecombinedBSplineBasis(Derivative(1), B0)
# B = B0

# ╔═╡ 2b263ef0-79af-11eb-3243-eb0ac08d1a46
C = collocation_matrix(B, xdata, SparseMatrixCSC{Float64});

# ╔═╡ 98f5a344-79af-11eb-1049-05b3511b5760
F = qr(C);

# ╔═╡ 6046792e-79af-11eb-1338-01159eefb4f3
coefs = F \ ydata

# ╔═╡ 6b405606-79af-11eb-3d80-8bdae47a03ea
fspline = Spline(B, coefs)

# ╔═╡ 6c65dff6-79c3-11eb-15ab-c9757e03c434
md"""## Least-squares as in de Boor

Similar to de Boor's chapter on least-squares approximation, with the difference that we use the continuous inner product (Galerkin matrix) on the left-hand side.

"""

# ╔═╡ 999dc716-79c5-11eb-3c07-93e12d6f0241
M = galerkin_matrix(B);

# ╔═╡ 3e88842a-79c4-11eb-12a7-9fb8f940971b
function lsq_weighted_data(xs, ys)
	gs = similar(ys)
	gs[begin] = ys[begin] * (xs[begin + 1] - xs[begin]) / 2
	is = eachindex(xs)[2:end-1]
	for i in is
		gs[i] = ys[i] * (xs[i + 1] - xs[i - 1]) / 2
	end
	gs[end] = ys[end] * (xs[end] - xs[end - 1]) / 2
	gs
end

# ╔═╡ 7be65118-79c3-11eb-09de-2f1fc37ef07d
gs = lsq_weighted_data(xdata, ydata)

# ╔═╡ 7f765e48-79c5-11eb-0730-43153731b317
rhs = C' * gs

# ╔═╡ a85cda4e-79c5-11eb-2a78-f1de71c9bc30
lsq_coefs = cholesky(M) \ rhs

# ╔═╡ b97a005e-79c5-11eb-1f41-3f5a4bd0d263
f_lsq = Spline(B, lsq_coefs)

# ╔═╡ 95461eda-79ae-11eb-3d0c-5bdb5411c9c9
begin
	scatter(xdata, ydata; label = "data", markersize=3, alpha = 0.2)
	plot!(f, xlims...; colour = :black, label = "f(x)", lw = 3)
	plot!(x -> fspline(x), xlims...; label = "Fit", lw = 4)
	plot!(x -> f_lsq(x), xlims...; ls = :dash, label = "Fit LSQ", lw = 4)
end

# ╔═╡ 012e964a-79ca-11eb-3eb6-5356e32db238
begin
	plot(x -> f_lsq(x) - f(x), xlims...)
	plot!(x -> fspline(x) - f(x), xlims...)
	scatter!(xbreaks, zeros(length(xbreaks)))
end

# ╔═╡ 1c0c5472-79ca-11eb-28ee-196377c2d294
begin
	plot(x -> (f_lsq(x) - fspline(x))^2, xlims...; yscale = :log10)
	scatter!(xbreaks, fill(1e-10, length(xbreaks)))
end

# ╔═╡ Cell order:
# ╠═5a37f162-79ae-11eb-18fe-b329714a6617
# ╠═6899fcb2-79c9-11eb-0d95-cb916a6751d3
# ╠═4476d184-79ae-11eb-135f-af80be0c9bde
# ╠═911d13ee-79b1-11eb-33b4-5d7885069097
# ╠═81a5efec-79ae-11eb-01f1-31bc96f18a6d
# ╠═22639830-79af-11eb-3834-fd118a685500
# ╠═823cb81a-79c6-11eb-1e4d-75eb02469abf
# ╠═2b263ef0-79af-11eb-3243-eb0ac08d1a46
# ╠═98f5a344-79af-11eb-1049-05b3511b5760
# ╠═6046792e-79af-11eb-1338-01159eefb4f3
# ╠═6b405606-79af-11eb-3d80-8bdae47a03ea
# ╠═95461eda-79ae-11eb-3d0c-5bdb5411c9c9
# ╟─6c65dff6-79c3-11eb-15ab-c9757e03c434
# ╠═7be65118-79c3-11eb-09de-2f1fc37ef07d
# ╠═7f765e48-79c5-11eb-0730-43153731b317
# ╠═999dc716-79c5-11eb-3c07-93e12d6f0241
# ╠═a85cda4e-79c5-11eb-2a78-f1de71c9bc30
# ╠═b97a005e-79c5-11eb-1f41-3f5a4bd0d263
# ╠═012e964a-79ca-11eb-3eb6-5356e32db238
# ╠═1c0c5472-79ca-11eb-28ee-196377c2d294
# ╠═3e88842a-79c4-11eb-12a7-9fb8f940971b
# ╠═4269cea4-79ac-11eb-20fb-6d35207dbbf8
