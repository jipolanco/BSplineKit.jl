### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 1a1cb966-75d8-11eb-0200-67aa7dc8d9f6
begin
	using Revise
	using BSplineKit
	using LinearAlgebra
end

# ╔═╡ 24d1e124-75d8-11eb-3ed2-07159fb779c0
breaks = -1:0.2:1

# ╔═╡ 41519470-75d8-11eb-37c9-37717713b53a
B = BSplineBasis(BSplineOrder(2), copy(breaks))

# ╔═╡ 307d442a-75db-11eb-0fd7-cda21072413f
fweight(x) = 1 - x^2

# ╔═╡ 4e6d7dc2-75d8-11eb-351a-ebb904648956
M = galerkin_matrix(B; fweight)

# ╔═╡ 4448cbfa-75db-11eb-2512-2117779405e8
M′ = galerkin_matrix(B, Derivative.((1, 1)); fweight)

# ╔═╡ e6d19636-75d8-11eb-1905-257106591c47
F = cholesky(M);

# ╔═╡ f349e5ec-75d8-11eb-1ae4-39bca9481f84
U = F.U;  # BandedMatrices (LAPACK?) stores U = Lᵀ instead of L

# ╔═╡ 26546340-75d9-11eb-0112-356ead018382
L = U'

# ╔═╡ 52d93c10-75d9-11eb-27b6-b5a31b361f36
T = inv(L)

# ╔═╡ 55791d8a-75db-11eb-0790-0b1a48151eec
T * M * T' ≈ I

# ╔═╡ 600ffc50-75db-11eb-2469-2302289fd7fd
T * M′ * T'

# ╔═╡ 289f443a-75d9-11eb-351e-d7a0846bf26e
L * L' ≈ M

# ╔═╡ Cell order:
# ╠═24d1e124-75d8-11eb-3ed2-07159fb779c0
# ╠═41519470-75d8-11eb-37c9-37717713b53a
# ╠═307d442a-75db-11eb-0fd7-cda21072413f
# ╠═4e6d7dc2-75d8-11eb-351a-ebb904648956
# ╠═4448cbfa-75db-11eb-2512-2117779405e8
# ╠═52d93c10-75d9-11eb-27b6-b5a31b361f36
# ╠═55791d8a-75db-11eb-0790-0b1a48151eec
# ╠═600ffc50-75db-11eb-2469-2302289fd7fd
# ╠═e6d19636-75d8-11eb-1905-257106591c47
# ╠═f349e5ec-75d8-11eb-1ae4-39bca9481f84
# ╠═26546340-75d9-11eb-0112-356ead018382
# ╠═289f443a-75d9-11eb-351e-d7a0846bf26e
# ╠═1a1cb966-75d8-11eb-0200-67aa7dc8d9f6
