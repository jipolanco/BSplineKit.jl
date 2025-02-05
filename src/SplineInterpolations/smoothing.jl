@doc raw"""
    fit(xs, ys, λ::Real, [BSplineOrder(4)]; [weights = nothing])

Fit a cubic smoothing spline to the given data.

Returns a natural cubic spline which roughly passes through the data (points `(xs[i], ys[i])`)
given some smoothing parameter ``λ``.
Note that ``λ = 0`` means no smoothing and the results are equivalent to
all the data.

One can optionally pass a `weights` vector if one desires to give different weights to
different data points (e.g. if one wants the curve to pass as close as possible to a
specific point). By default all weights are ``w_i = 1``.

More precisely, the returned spline ``S(x)`` minimises:

```math
∑_{i = 1}^N w_i |y_i - S(x_i)|^2 + λ ∫_{x_1}^{x_N} \left[ S''(x) \right]^2 \, \mathrm{d}x
```

Only cubic splines (`BSplineOrder(4)`) are currently supported.

# Examples

```jldoctest smoothing_spline
julia> xs = (0:0.01:1).^2;

julia> ys = @. cospi(2 * xs) + 0.1 * sinpi(200 * xs);  # smooth + highly fluctuating components

julia> λ = 0.001;  # smoothing parameter

julia> S = fit(xs, ys, λ)
101-element Spline{Float64}:
 basis: 101-element RecombinedBSplineBasis of order 4, domain [0.0, 1.0], BCs {left => (D{2},), right => (D{2},)}
 order: 4
 knots: [0.0, 0.0, 0.0, 0.0, 0.0001, 0.0004, 0.0009, 0.0016, 0.0025, 0.0036  …  0.8836, 0.9025, 0.9216, 0.9409, 0.9604, 0.9801, 1.0, 1.0, 1.0, 1.0]
 coefficients: [0.946872, 0.631018, 1.05101, 1.04986, 1.04825, 1.04618, 1.04366, 1.04067, 1.03722, 1.03331  …  0.437844, 0.534546, 0.627651, 0.716043, 0.798813, 0.875733, 0.947428, 1.01524, 0.721199, 0.954231]
```
"""
function fit end

function fit(
        xs::AbstractVector, ys::AbstractVector, λ::Real, order::BSplineOrder{4};
        weights::Union{Nothing, AbstractVector} = nothing,
    )
    λ ≥ 0 || throw(DomainError(λ, "the smoothing parameter λ must be non-negative"))
    eachindex(xs) == eachindex(ys) || throw(DimensionMismatch("x and y vectors must have the same length"))
    N = length(xs)
    cs = similar(xs)

    T = eltype(cs)

    # Create natural cubic B-spline basis with knots = input points
    B = BSplineBasis(order, copy(xs))
    R = RecombinedBSplineBasis(B, Natural())

    # Compute collocation matrices for derivatives 0 and 2
    A = BandedMatrix{T}(undef, (N, N), (1, 1))  # 3 bands are enough
    D = similar(A)
    collocation_matrix!(A, R, xs, Derivative(0))
    collocation_matrix!(D, R, xs, Derivative(2))

    # Matrix containing grid steps (banded + symmetric, so we don't compute the lower part)
    Δ_upper = BandedMatrix{T}(undef, (N, N), (0, 1))
    fill!(Δ_upper, 0)
    @inbounds for i ∈ axes(Δ_upper, 1)[2:end-1]
        # Δ_upper[i, i - 1] = xs[i] - xs[i - 1]  # this one is obtained by symmetry
        Δ_upper[i, i] = 2 * (xs[i + 1] - xs[i - 1])
        Δ_upper[i, i + 1] = xs[i + 1] - xs[i]
    end
    Δ = Hermitian(Δ_upper)  # symmetric matrix with 3 bands

    # The integral of the squared second derivative is (H * cs) ⋅ (D * cs) / 6.
    H = Δ * D

    # Directly construct LHS matrix
    # M = Hermitian((H'D + D'H) * (λ / 6) + 2 * A' * W * A)  # usually positive definite

    # Construct LHS matrix trying to reduce computations
    B = H' * D  # this matrix has 5 bands
    buf_1 = (B .+ B') .* (λ / 6)  # = (H'D + D'H) * (λ / 6)
    buf_2 = if weights === nothing
        A' * A
    else
        A' * Diagonal(weights) * A  # = A' * W * A
    end

    @. buf_1 = buf_1 + 2 *  buf_2  # (H'D + D'H) * (λ / 6) + 2 * A' * W * A
    M = Hermitian(buf_1)  # the matrix is actually symmetric (usually positive definite)
    F = cholesky!(M)      # factorise matrix (assuming posdef)

    # Directly construct RHS
    # z = 2 * A' * (W * ys)  # RHS
    # ldiv!(cs, F, z)

    # Construct RHS trying to reduce allocations
    zs = copy(ys)
    if weights !== nothing
        eachindex(weights) == eachindex(xs) || throw(DimensionMismatch("the `weights` vector must have the same length as the data"))
        lmul!(Diagonal(weights), zs)  # zs = W * ys
    end
    mul!(cs, A', zs)  # cs = A' * (W * ys)
    lmul!(2, cs)      # cs = 2 * A' * (W * ys)

    ldiv!(F, cs)      # solve linear system

    Spline(R, cs)
end

fit(x, y, λ; kws...) = fit(x, y, λ, BSplineOrder(4); kws...)
