function galerkin_matrix(
        B::BSplineBasis,
        ::Type{M} = BandedMatrix{Float64}
    ) where {M <: AbstractMatrix}
    N = length(B)
    A = allocate_galerkin_matrix(M, N, order(B))
    galerkin_matrix!(A, B)
end

allocate_galerkin_matrix(::Type{M}, N, k) where {M <: AbstractMatrix} =
    Symmetric(M(undef, N, N))

function allocate_galerkin_matrix(::Type{M}, N, k) where {M <: BandedMatrix}
    # The upper/lower bandwidths are:
    # - for even k: Nb = k / 2       (total = k + 1 bands)
    # - for odd  k: Nb = (k + 1) / 2 (total = k + 2 bands)
    # Note that the matrix is also symmetric, so we only need the upper band.
    Nb = (k + 1) >> 1
    A = M(undef, (N, N), (0, Nb))
    Symmetric(A)
end

function galerkin_matrix!(S::Symmetric, B::BSplineBasis)
    N = size(S, 1)

    if N != length(B)
        throw(ArgumentError("wrong dimensions of Galerkin matrix"))
    end

    fill!(S, 0)

    # The matrix is symmetric, so we fill only the upper part.
    # For now we assume that S uses the upper part of its parent.
    @assert S.uplo === 'U'
    A = parent(S)

    k = order(B)
    h = (k + 1) >> 2  # k/2 if k is even

    # Upper part: j >= i
    for j = 1:N
        # We're only visiting the elements that have non-zero values.
        # In other words, we know that S[i, j] = 0 outside the chosen interval.
        istart = clamp(j - h, 1, N)
        bj = BSpline(B, j)
        for i = istart:j
            bi = BSpline(B, i)
        end
    end

    S
end
