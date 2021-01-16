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

function galerkin_tensor!(A::BandedTensor3D,
                          Bs::NTuple{3,AbstractBSplineBasis},
                          deriv::DerivativeCombination{3})
    _check_bases(Bs)

    Ns = size(A)
    if any(Ns .!= length.(Bs))
        throw(ArgumentError("wrong dimensions of Galerkin tensor"))
    end

    Ni, Nj, Nl = Ns
    @assert Ni == Nj  # verified earlier...

    Bi, Bj, Bl = Bs

    k = order(Bi)  # same for all bases (see _check_bases)
    t = knots(Bi)
    h = k - 1
    T = eltype(A)

    # Quadrature information (weights, nodes).
    quad = _quadrature_prod(3k - 3)

    Al = Matrix{T}(undef, 2k - 1, 2k - 1)
    δl, δr = num_constraints(Bl) .- num_constraints(Bi)
    @assert Ni == Nj == Nl + δl + δr

    if bandwidth(A) != k - 1
        throw(ArgumentError("BandedTensor3D must have bandwidth = $(k - 1)"))
    end
    if bandshift(A) != (0, 0, δl)
        throw(ArgumentError("BandedTensor3D must have bandshift = (0, 0, $δl)"))
    end

    for l = 1:Nl
        ll = l + δl
        istart = clamp(ll - h, 1, Ni)
        iend = clamp(ll + h, 1, Nj)
        is = istart:iend
        js = is

        band_ind = BandedTensors.band_indices(A, l)
        @assert issubset(is, band_ind) && issubset(js, band_ind)

        i0 = first(band_ind) - 1
        j0 = i0
        @assert i0 == ll - k

        fill!(Al, 0)

        tl = support(Bl, l)
        fl = x -> evaluate(Bl, l, x, deriv[3])

        for j in js
            tj = support(Bj, j)
            fj = x -> evaluate(Bj, j, x, deriv[2])
            for i in is
                ti = support(Bi, i)
                fi = x -> evaluate(Bi, i, x, deriv[1])
                t_inds = intersect(ti, tj, tl)
                isempty(t_inds) && continue
                f = x -> fi(x) * fj(x) * fl(x)
                Al[i - i0, j - j0] = _integrate(f, t, t_inds, quad)
            end
        end

        A[:, :, l] = Al
    end

    A
end
