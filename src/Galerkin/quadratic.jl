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
        throw(ArgumentError("the first two bases must have the same lengths"))
    end
    δl, δr = num_constraints(Bs[3]) .- num_constraints(Bs[1])
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
        throw(ArgumentError("wrong dimensions of Galerkin tensor"))
    end

    Bi, Bj, Bl = Bs
    Ni, Nj, Nl = Ns
    @assert Ni == Nj  # verified at construction of BandedTensor3D

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
    Al = @MMatrix zeros(T, 2k - 1, 2k - 1)
    nlast = last(eachindex(ts))

    for n in eachindex(ts)
        n == nlast && break
        tn, tn1 = ts[n], ts[n + 1]
        tn1 == tn && continue  # interval of length = 0

        metric = QuadratureMetric(tn, tn1)
        xs = metric .* quadx

        is = nonzero_in_segment(Bi, n)
        js = nonzero_in_segment(Bj, n)
        ls = nonzero_in_segment(Bl, n)

        bis = eval_basis_functions(Bi, is, xs, deriv[1])
        bjs = same_12 ? bis : eval_basis_functions(Bj, js, xs, deriv[2])
        bls = same_13 ? bis : same_23 ? bjs :
            eval_basis_functions(Bl, ls, xs, deriv[3])

        # We compute the submatrix A[:, :, l] for each `l`.
        for (nl, l) in enumerate(ls)
            fill!(Al, 0)
            inds = BandedTensors.band_indices(A, l)
            @assert is ⊆ inds && js ⊆ inds
            @assert size(Al, 1) == size(Al, 2) == length(inds)
            δi = searchsortedlast(inds, is[1]) - 1
            δj = searchsortedlast(inds, js[1]) - 1
            for nj in eachindex(js), ni in eachindex(is)
                Al[ni + δi, nj + δj] =
                    metric.α * ((bis[ni] .* bjs[nj] .* bls[nl]) ⋅ quadw)
            end
            # TODO
            # - nicer way to do this?
            # - don't forget +=
            A[:, :, l] = A[:, :, l].data + Al
        end
    end

    A
end
