"""
    galerkin_tensor(
        B::AbstractBSplineBasis,
        (D₁::Derivative, D₂::Derivative, D₃::Derivative),
        [T = Float64],
    )

Compute 3D banded tensor appearing from quadratic terms in Galerkin method.

As with [`galerkin_matrix`](@ref), it is also possible to combine different
functional bases by passing, instead of `B`, a tuple `(B₁, B₂, B₃)` of three
`AbstractBSplineBasis`.
For now, the first two bases, `B₁` and `B₂`, must have the same length.

The tensor is efficiently stored in a [`BandedTensor3D`](@ref) object.
"""
function galerkin_tensor end

function galerkin_tensor(
        Bs::NTuple{3,AbstractBSplineBasis},
        deriv::DerivativeCombination{3},
        ::Type{T} = Float64,
    ) where {T}
    _check_bases(Bs)
    dims = length.(Bs)
    b = order(first(Bs)) - 1   # band width
    if length(Bs[1]) != length(Bs[2])
        throw(DimensionMismatch("the first two bases must have the same lengths"))
    end
    δl = first(num_constraints(Bs[3])) - first(num_constraints(Bs[1]))
    A = BandedTensor3D{T}(undef, dims, Val(b), bandshift=(0, 0, δl))
    galerkin_tensor!(A, Bs, deriv)
end

galerkin_tensor(B::AbstractBSplineBasis, args...) =
    galerkin_tensor((B, B, B), args...)

"""
    galerkin_tensor!(
        A::BandedTensor3D,
        B::AbstractBSplineBasis,
        (D₁::Derivative, D₂::Derivative, D₃::Derivative),
    )

Compute 3D Galerkin tensor in-place.

See [`galerkin_tensor`](@ref) for details.
"""
function galerkin_tensor! end

function galerkin_tensor!(
        A::BandedTensor3D, Bs::NTuple{3,AbstractBSplineBasis},
        deriv::DerivativeCombination{3},
    )
    _check_bases(Bs)

    Ns = size(A)
    if Ns != length.(Bs)
        throw(DimensionMismatch("wrong dimensions of Galerkin tensor"))
    end

    Bi, Bj, Bl = Bs
    Ni, Nj, Nl = Ns
    @assert Ni == Nj  # verified at construction of BandedTensor3D

    ioff = first(num_constraints(Bi))
    joff = first(num_constraints(Bj))
    loff = first(num_constraints(Bl))

    # Orders and knots are assumed to be the same (see _check_bases).
    k = order(Bi)
    ts = knots(Bi)

    if bandwidth(A) != k - 1
        throw(ArgumentError("BandedTensor3D must have bandwidth = $(k - 1)"))
    end

    δl, _ = num_constraints(Bl) .- num_constraints(Bi)

    if bandshift(A) != (0, 0, δl)
        throw(ArgumentError("BandedTensor3D must have bandshift = (0, 0, $δl)"))
    end

    same_12 = Bi == Bj && deriv[1] == deriv[2]
    same_13 = Bi == Bl && deriv[1] == deriv[3]
    same_23 = Bj == Bl && deriv[2] == deriv[3]

    # Quadrature information (weights, nodes).
    quadx, quadw = _quadrature_prod(Val(3k - 3))

    fill!(A, 0)
    T = eltype(A)
    Al = zero(MMatrix{2k - 1, 2k - 1, T})
    nlast = last(eachindex(ts))

    @inbounds for n in eachindex(ts)
        n == nlast && break
        tn, tn1 = ts[n], ts[n + 1]
        tn1 == tn && continue  # interval of length = 0

        metric = QuadratureMetric(tn, tn1)
        xs = metric .* quadx

        ilast = n - ioff
        jlast = n - joff
        llast = n - loff
        l₀ = llast - order(Bs[3])

        for (x, w) ∈ zip(xs, quadw)
            _, bis = evaluate_all(Bi, x, deriv[1], T; ileft = ilast)
            _, bjs = same_12 ?
                (ilast, bis) : evaluate_all(Bj, x, deriv[2], T; ileft = jlast)
            _, bls = same_13 ?
                (ilast, bis) : same_23 ?
                (jlast, bjs) : evaluate_all(Bl, x, deriv[3], T; ileft = llast)

            # Iterate in increasing order (not sure if it really helps performance)
            bls = reverse(bls)

            # We compute the submatrix A[:, :, l] for each `l`.
            for (δl, bl) in pairs(bls)
                iszero(bl) && continue  # can prevent problems at the borders
                l = l₀ + δl
                y₀ = metric.α * w * bl
                @inbounds fill!(Al, 0)
                inds = BandedTensors.band_indices(A, l)
                # @assert inds == (l - k + 1:l + k - 1) .+ bandshift(A)[3]
                # @assert (ilast - k + 1:ilast) ⊆ inds
                # @assert (jlast - k + 1:jlast) ⊆ inds
                Δi = first(inds) - 1
                Δj = Δi
                for (δj, bj) ∈ pairs(bjs)
                    y₁ = y₀ * bj
                    j = jlast + 1 - δj
                    jj = j - Δj
                    for (δi, bi) ∈ pairs(bis)
                        i = ilast + 1 - δi  # actual index of basis function
                        ii = i - Δi         # index in submatrix Al
                        @inbounds Al[ii, jj] = y₁ * bi
                    end
                end
                @inbounds A[:, :, l] += Al
            end
        end
    end

    A
end
