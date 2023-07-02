_normalise_ops(op::Natural, B) = _natural_ops(B)

# Generalised natural boundary conditions.
#
# This is equivalent to:
#
#       ops = (Derivative(2), Derivative(3), …, Derivative(k ÷ 2))
#
function _make_submatrix(::Natural, Bdata::Tuple, ::Type{T}) where {T}
    B, side = Bdata
    k = order(B)
    isodd(k) && throw(ArgumentError(
        lazy"`Natural` boundary condition only supported for even-order splines (got k = $k)"
    ))
    h = k ÷ 2

    # rhs = [h, 0, 0, …, 0] (the `h` at the beginning is kind of arbitrary)
    rhs = SVector(ntuple(i -> i == 1 ? T(h) : zero(T), Val(h + 1))...)

    A = if side === Val(:left)
        x = first(boundaries(B))
        js = 1:(h + 1)  # indices of left B-splines

        # Evaluate B-spline derivatives at boundary.
        bs = _natural_eval_derivatives(B, x, js, Val(h), T)

        # Construct linear systems for determining recombination matrix
        # coefficients.
        C1 = _natural_system_matrix(bs, 1)
        T1 = C1 \ rhs  # coefficients associated to recombined function ϕ₁

        C2 = _natural_system_matrix(bs, 2)
        T2 = C2 \ rhs  # coefficients associated to recombined function ϕ₂

        hcat(_remove_near_zeros(T1), _remove_near_zeros(T2))
    elseif side === Val(:right)
        x = last(boundaries(B))
        N = length(B)
        js = (N - h):N
        bs = _natural_eval_derivatives(B, x, js, Val(h), T)

        C1 = _natural_system_matrix(bs, 1)
        T1 = C1 \ rhs

        C2 = _natural_system_matrix(bs, 2)
        T2 = C2 \ rhs

        hcat(_remove_near_zeros(T1), _remove_near_zeros(T2))
    end

    ndrop = 0
    ndrop, A
end

function _remove_near_zeros(A::SArray; rtol = 100 * eps(eltype(A)))
    v = maximum(abs, A)
    ϵ = rtol * v
    typeof(A)((abs(x) < ϵ ? zero(x) : x) for x ∈ A)
end

@inline _natural_ops(B::BSplineBasis) = _natural_ops(BSplineOrder(order(B)))
@inline function _natural_ops(::BSplineOrder{k}) where {k}
    isodd(k) && throw(ArgumentError(
        lazy"`Natural` boundary condition only supported for even-order splines (got k = $k)"
    ))
    _natural_ops(Val(k ÷ 2))
end
@inline function _natural_ops(::Val{h}) where {h}
    @assert h ≥ 2
    (_natural_ops(Val(h - 1))..., Derivative(h))
end
@inline _natural_ops(::Val{1}) = ()

# Case h = 1: return empty (0 × 2) matrix.
_natural_eval_derivatives(B, x, js, ::Val{1}, ::Type{T}) where {T} =
    zero(SMatrix{0, 2, T})

# Evaluate derivatives 2:h of B-splines 1:(h + 1) at the boundaries.
@generated function _natural_eval_derivatives(
        B, x, js, ::Val{h}, ::Type{T},
    ) where {h, T}
    @assert h ≥ 2
    ex = quote
        @assert length(js) == $h + 1
        M = zero(MMatrix{$h - 1, $h + 1, $T})
    end
    for i ∈ 1:(h - 1)
        jlast = Symbol(:jlast_, i)
        bs = Symbol(:bs_, i)
        ileft = i == 1 ? :(nothing) : Symbol(:jlast_, i - 1)
        ex = quote
            $ex
            $jlast, $bs = evaluate_all(B, x, Derivative($i + 1), $T; ileft = $ileft)
            @assert length($bs) == order(B)
            for n ∈ axes(M, 2)
                j = js[n]
                δj = $jlast + 1 - j
                if δj ∈ eachindex($bs)
                    @inbounds M[$i, n] = $bs[δj]
                end
            end
        end
    end
    quote
        $ex
        SMatrix(M)
    end
end

# On the left boundary, `i` is the index of the resulting recombined basis
# function ϕᵢ.
function _natural_system_matrix(bs::SMatrix{h⁻, h⁺}, i) where {h⁻, h⁺}
    h = h⁻ + 1
    @assert h⁺ == h + 1
    M = similar(bs, Size(h⁺, h⁺))
    fill!(M, 0)
    @views M[1, :] .= 1  # arbitrary condition
    for i ∈ 2:h
        # This corresponds to the zero condition for the i-th derivative (with i ∈ 2:h)
        derivs = @view bs[i - 1, :]      # i-th derivative of B-splines {b[1], ..., b[h + 1]}
        dnorm = sqrt(sum(abs2, derivs))  # normalisation factor (improves condition number)
        for j ∈ axes(M, 2)
            M[i, j] = derivs[j] / dnorm
        end
    end
    # This is a locality condition: we want the matrix to be kind of banded.
    @assert i ∈ (1, 2)
    if i == 1
        M[end, end] = 1
    elseif i == 2
        M[end, 1] = 1
    end
    SMatrix(M)
end
