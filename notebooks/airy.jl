### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 5abfca08-6935-11eb-37b5-374ea3b4ddf0
begin
	using Revise
	using BSplineKit
	using SpecialFunctions: airyai
	using CairoMakie
	using LinearAlgebra
	using BandedMatrices
	# using SparseArrays
	using QuadGK
end

# ╔═╡ 48367b6e-6955-11eb-3ea4-5b1f8099f8ab
md"""
# Airy equation

Example taken from Olver & Townsend, SIAM Rev. 2013.
"""

# ╔═╡ ab3e2c3c-6952-11eb-2c48-d5dea019ea78
N = 500

# ╔═╡ d1540532-6951-11eb-2df0-858510e4d960
breaks =
	vcat(
		range(-1, 0; length = N + 1),
		range(0, 1; length = (N ÷ 8) + 1)[2:end],
	)
	# collect(range(-1, 1; length = N + 1))
	# [-cos(π * i / N) for i = 0:N]

# ╔═╡ ce14d81c-6936-11eb-2876-fd539b26f77c
B = BSplineBasis(BSplineOrder(6), breaks)

# ╔═╡ e4b55060-6936-11eb-336d-7f6d75edabff
R = RecombinedBSplineBasis(Derivative(0), B)

# ╔═╡ 06d6d560-6937-11eb-03e3-5b2be0c46741
xs = collocation_points(B)

# ╔═╡ 947045b2-6937-11eb-0501-139f38e7783b
Cfact = lu!(collocation_matrix(B, xs));

# ╔═╡ 763b4c5a-6938-11eb-2969-958078a01dea
md"## Variable coefficient"

# ╔═╡ 86f7485a-6938-11eb-2775-3997a69d16dc
cs = Cfact \ xs

# ╔═╡ b9eb0044-6938-11eb-0ece-fb3c8466596d
Tc = galerkin_tensor((R, R, B), Derivative.((0, 0, 0)));

# ╔═╡ 2f232272-6945-11eb-3c84-4bf241954db3
Mc = Hermitian(Tc * cs)

# ╔═╡ 390c6438-694a-11eb-05ee-9da9e9f26d18
issymmetric(parent(Mc))

# ╔═╡ ca4a99fa-6946-11eb-3931-a7cb96c58684
summary(Mc)

# ╔═╡ ea8e4e44-6939-11eb-0a3c-4b669c5645c8
md"## LHS"

# ╔═╡ 69982892-6938-11eb-2972-333ba2799cf4
md"## RHS"

# ╔═╡ 32ba8d7e-6938-11eb-3056-b543422fa4f5
# RHS operator
Mf = galerkin_matrix((R, B))

# ╔═╡ b343c62e-694a-11eb-37bd-d14a3d58f8d8
md"## Solution"

# ╔═╡ 70f79dde-6938-11eb-13b3-f9775864234c
md"## Plots"

# ╔═╡ b71a4ed6-6935-11eb-2db1-c9bbdb97b9cc
u_exact(x; ε) = airyai(x / cbrt(ε))

# ╔═╡ 19b426e4-6937-11eb-2d24-436700fa442e
# Affine part of the solution
u_0(x; ε) = ((1 - x) * u_exact(-1; ε) + (1 + x) * u_exact(1; ε)) / 2

# ╔═╡ a9e506de-6935-11eb-39f0-fb44ea95ffd3
ε = 10^(-6)

# ╔═╡ f05d7e94-6939-11eb-3c0e-af2d31d8eee0
L = let
	L = galerkin_matrix(R, Derivative.((1, 1))) :: Hermitian
	A = parent(L)
	A .*= ε
	@assert L.uplo == 'U'
	A .+= UpperTriangular(Mc)
	L
end

# ╔═╡ e345757a-694a-11eb-0c84-71197d96685d
Lfact = lu(BandedMatrix(L))

# ╔═╡ 0a7d0790-693a-11eb-1ae0-d3ac008e6245
cond(BandedMatrix(L))

# ╔═╡ 8a214fa4-6937-11eb-382e-bf64ff050dbb
# RHS coefficients
fs = ldiv!(Cfact, xs .* u_0.(xs; ε))

# ╔═╡ 59760524-6938-11eb-0766-a193e2632d53
rhs = -Mf * fs

# ╔═╡ b9344f10-694a-11eb-21c5-c1bf6c0733e1
vR = Lfact \ rhs

# ╔═╡ ff706bd6-694a-11eb-10a5-01a9d40686f3
vB = recombination_matrix(R) * vR

# ╔═╡ 1b6b3ba6-694b-11eb-0f4e-a18cc5b6e991
Sv = Spline(B, vB)

# ╔═╡ 761215d4-694b-11eb-1089-a7ccde18df5b
usol(x; ε) = u_0(x; ε) + Sv(x)

# ╔═╡ 4a7dcc82-6951-11eb-1c74-ddfda1515953
uerr(x; ε) = (usol(x; ε) - u_exact(x; ε))^2

# ╔═╡ 079e6a38-6952-11eb-2d5f-a9d39f6ca4e0
quadgk(x -> uerr(x; ε), -1, 1)

# ╔═╡ 4b056762-6935-11eb-116e-47a5a46dbfff
let
	fig, ax, _ = plot(-1..1, x -> u_exact(x; ε))
	# plot!(ax, -1..1, x -> u_0(x; ε); color = :red)
	# plot!(ax, -1..1, x -> Spline(B, fs)(x); color = :blue)
	plot!(ax, -1..1, x -> usol(x; ε); color = :orange)
	fig
end

# ╔═╡ ff09558c-694b-11eb-379e-0f3b985db80a
let
	fig, ax, _ = plot(-1..1, x -> log10.(uerr.(x; ε)))
	ylims!(ax, -20, 0)
	fig
end

# ╔═╡ Cell order:
# ╟─48367b6e-6955-11eb-3ea4-5b1f8099f8ab
# ╠═ab3e2c3c-6952-11eb-2c48-d5dea019ea78
# ╠═d1540532-6951-11eb-2df0-858510e4d960
# ╠═ce14d81c-6936-11eb-2876-fd539b26f77c
# ╠═e4b55060-6936-11eb-336d-7f6d75edabff
# ╠═06d6d560-6937-11eb-03e3-5b2be0c46741
# ╠═947045b2-6937-11eb-0501-139f38e7783b
# ╟─763b4c5a-6938-11eb-2969-958078a01dea
# ╠═86f7485a-6938-11eb-2775-3997a69d16dc
# ╠═b9eb0044-6938-11eb-0ece-fb3c8466596d
# ╠═2f232272-6945-11eb-3c84-4bf241954db3
# ╠═390c6438-694a-11eb-05ee-9da9e9f26d18
# ╠═ca4a99fa-6946-11eb-3931-a7cb96c58684
# ╟─ea8e4e44-6939-11eb-0a3c-4b669c5645c8
# ╠═f05d7e94-6939-11eb-3c0e-af2d31d8eee0
# ╠═e345757a-694a-11eb-0c84-71197d96685d
# ╠═0a7d0790-693a-11eb-1ae0-d3ac008e6245
# ╟─69982892-6938-11eb-2972-333ba2799cf4
# ╠═8a214fa4-6937-11eb-382e-bf64ff050dbb
# ╠═32ba8d7e-6938-11eb-3056-b543422fa4f5
# ╠═59760524-6938-11eb-0766-a193e2632d53
# ╟─b343c62e-694a-11eb-37bd-d14a3d58f8d8
# ╠═b9344f10-694a-11eb-21c5-c1bf6c0733e1
# ╠═ff706bd6-694a-11eb-10a5-01a9d40686f3
# ╠═1b6b3ba6-694b-11eb-0f4e-a18cc5b6e991
# ╠═761215d4-694b-11eb-1089-a7ccde18df5b
# ╠═4a7dcc82-6951-11eb-1c74-ddfda1515953
# ╠═079e6a38-6952-11eb-2d5f-a9d39f6ca4e0
# ╟─70f79dde-6938-11eb-13b3-f9775864234c
# ╠═4b056762-6935-11eb-116e-47a5a46dbfff
# ╠═ff09558c-694b-11eb-379e-0f3b985db80a
# ╠═b71a4ed6-6935-11eb-2db1-c9bbdb97b9cc
# ╠═19b426e4-6937-11eb-2d24-436700fa442e
# ╠═a9e506de-6935-11eb-39f0-fb44ea95ffd3
# ╠═5abfca08-6935-11eb-37b5-374ea3b4ddf0
