"""
    Spline

Spline function.

---

    Spline(b::BSplineBasis, coefs::AbstractVector)

Construct a spline from a B-spline basis and a vector of B-spline coefficients.
"""
struct Spline{k,  # B-spline order
              Basis <: BSplineBasis,
              T,  # type of coefficient (e.g. Float64, ComplexF64)
              CoefVector <: AbstractVector{T}}
    basis :: Basis
    coefs :: CoefVector
    buf   :: MVector{k,T}  # buffer for evaluation of splines

    function Spline(B::BSplineBasis, coefs::AbstractVector)
        length(coefs) == length(B) ||
            throw(ArgumentError("wrong number of coefficients"))
        Basis = typeof(B)
        T = eltype(coefs)
        CoefVector = typeof(coefs)
        k = order(B)
        @assert k >= 1
        buf = MVector(ntuple(d -> zero(T), Val(k)))
        new{k, Basis, T, CoefVector}(B, coefs, buf)
    end
end

"""
    Spline(B::BSplineBasis, [T=Float64])

Construct spline with uninitialised vector of coefficients.
"""
function Spline(B::BSplineBasis, ::Type{T}=Float64) where {T}
    coefs = Vector{T}(undef, length(B))
    Spline(B, coefs)
end

coefficients(S::Spline) = S.coefs
basis(S::Spline) = S.basis
knots(S::Spline) = knots(basis(S))
order(::Type{<:Spline{k}}) where {k} = k
order(S::Spline) = order(typeof(S))

"""
    (S::Spline)(x)

Evaluate spline at coordinate `x`.

The implementation uses [De Boor's algorithm](https://en.wikipedia.org/wiki/De_Boor's_algorithm).
"""
function (S::Spline)(x)
    T = eltype(S.coefs)
    t = knots(S)
    n = get_knot_interval(t, x)

    n === nothing && return zero(T)  # x is outside of knot domain

    k = order(S)
    @inbounds let c = S.coefs, d = S.buf
        for j = 1:k
            d[j] = c[j + n - k]
        end
        for r = 2:k, j = k:-1:r
            α = (x - t[j + n - k]) / (t[j + n - r + 1] - t[j + n - k])
            d[j] = (1 - α) * d[j - 1] + α * d[j]
        end
        d[k]
    end
end

"""
    diff(S::Spline, [N::Union{Val, Integer} = Val(1)]) -> Spline

Return `N`-th derivative of spline `S` as a new spline.
"""
function Base.diff(S::Spline, ::Val{Ndiff} = Val(1)) where {Ndiff}
    Ndiff :: Integer
    @assert Ndiff >= 1

    u = coefficients(S)
    t = knots(S)
    k = order(S)

    if Ndiff >= k
        throw(ArgumentError(
            "cannot differentiate order $k spline $Ndiff times!"))
    end

    @assert Base.require_one_based_indexing(u)
    du = copy(u)

    @inbounds for m = 1:Ndiff, i in Iterators.Reverse(eachindex(du))
        dt = t[i + k - m] - t[i]
        if iszero(dt) || i == 1
            # In this case, the B-spline that this coefficient is
            # multiplying is zero everywhere, so we can set this to zero.
            # From de Boor (2001, p. 117): "anything times zero is zero".
            du[i] = 0
        else
            du[i] = (k - m) * (du[i] - du[i - 1]) / dt
        end
    end

    # Finally, create lower-order spline with the given coefficients.
    # Note that the spline has `2 * Ndiff` fewer knots, and `Ndiff` fewer
    # B-splines.
    N = length(u)
    Nt = length(t)
    t_new = view(t, (1 + Ndiff):(Nt - Ndiff))
    B = BSplineBasis(Val(k - Ndiff), t_new, augment=false)

    Spline(B, view(du, (1 + Ndiff):N))
end

# Zeroth derivative: return S itself.
@inline Base.diff(S::Spline, ::Val{0}) = S
@inline Base.diff(S::Spline, Ndiff::Integer) = diff(S, Val(Ndiff))

"""
    integral(S::Spline)

Returns an antiderivative of the given spline as a new spline.

See de Boor 2001, p. 127.
"""
function integral(S::Spline)
    u = coefficients(S)
    t = knots(S)
    k = order(S)
    @assert Base.require_one_based_indexing(u)

    Nt = length(t)
    N = length(u)

    # Note that the new spline has 2 more knots and 1 more B-spline.
    t_int = similar(t, Nt + 2)
    t_int[2:(end - 1)] .= t
    t_int[1] = t_int[2]
    t_int[end] = t_int[end - 1]

    β = similar(u, N + 1)
    β[1] = 0

    @inbounds for i in eachindex(u)
        m = i + 1
        β[m] = 0
        for j = 1:i
            β[m] += u[j] * (t[j + k] - t[j]) / k
        end
    end

    B = BSplineBasis(Val(k + 1), t_int, augment=false)
    Spline(B, β)
end

function get_knot_interval(t::AbstractVector, x)
    # The result is such that t[n] <= x < t[n + 1]
    n = searchsortedlast(t, x)
    n == 0 && return nothing  # x < t[1]

    Nt = length(t)

    if n == Nt  # i.e. if x >= t[end]
        t_last = t[n]
        x > t_last && return nothing
        # If x is exactly on the last knot, decrease the index as necessary.
        while t[n] == t_last
            n -= 1
        end
    end

    n
end
