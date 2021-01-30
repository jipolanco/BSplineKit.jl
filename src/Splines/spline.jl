"""
    Spline

Spline function.

---

    Spline(b::BSplineBasis, coefs::AbstractVector)

Construct a spline from a B-spline basis and a vector of B-spline coefficients.

---

    Spline(undef, B::BSplineBasis, [T=Float64])

Construct a spline with uninitialised vector of coefficients.

---

    (S::Spline)(x)

Evaluate spline at coordinate `x`.
"""
struct Spline{
        T,  # type of coefficient (e.g. Float64, ComplexF64)
        Basis <: BSplineBasis,
        CoefVector <: AbstractVector{T},
    }
    basis :: Basis
    coefs :: CoefVector
    function Spline(B::BSplineBasis, coefs::AbstractVector)
        length(coefs) == length(B) ||
            throw(ArgumentError("wrong number of coefficients"))
        Basis = typeof(B)
        T = eltype(coefs)
        CoefVector = typeof(coefs)
        k = order(B)
        @assert k >= 1
        new{T, Basis, CoefVector}(B, coefs)
    end
end

Broadcast.broadcastable(S::Spline) = Ref(S)

Base.copy(S::Spline) = Spline(basis(S), copy(coefficients(S)))

function Base.show(io::IO, S::Spline)
    println(io, length(S), "-element ", typeof(S), ':')
    println(io, " order: ", order(S))
    println(io, " knots: ", knots(S))
    print(io, " coefficients: ", coefficients(S))
    nothing
end

Base.:(==)(P::Spline, Q::Spline) =
    basis(P) === basis(Q) && coefficients(P) == coefficients(Q)

Base.isapprox(P::Spline, Q::Spline; kwargs...) =
    basis(P) === basis(Q) &&
    isapprox(coefficients(P), coefficients(Q); kwargs...)

function Spline(::UndefInitializer, B::BSplineBasis,
                ::Type{T}=Float64) where {T}
    coefs = Vector{T}(undef, length(B))
    Spline(B, coefs)
end

"""
    coefficients(S::Spline)

Get B-spline coefficients of the spline.
"""
coefficients(S::Spline) = S.coefs

"""
    length(S::Spline)

Returns the number of coefficients in the spline.

Note that this is equal to the number of basis functions, `length(basis(S))`.
"""
Base.length(S::Spline) = length(coefficients(S))

"""
    eltype(S::Spline)

Returns type of element returned when evaluating the [`Spline`](@ref).
"""
Base.eltype(S::Spline{T}) where {T} = T

"""
    basis(S::Spline) -> BSplineBasis

Returns the associated B-spline basis.
"""
basis(S::Spline) = S.basis

knots(S::Spline) = knots(basis(S))
order(::Type{<:Spline{T,Basis}}) where {T,Basis} = order(Basis)
order(S::Spline) = order(typeof(S))

# TODO allow evaluating derivatives at point `x` (should be much cheaper than
# constructing a new Spline for the derivative)
function (S::Spline)(x)
    T = eltype(S)
    t = knots(S)
    n = get_knot_interval(t, x)
    n === nothing && return zero(T)  # x is outside of knot domain
    k = order(S)
    spline_kernel(coefficients(S), t, n, x, BSplineOrder(k))
end

function spline_kernel(c::AbstractVector{T},
                       t, n, x, ::BSplineOrder{k}) where {T,k}
    # Algorithm adapted from https://en.wikipedia.org/wiki/De_Boor's_algorithm
    if @generated
        quote
            w_0 = zero(T)  # this is to make the compiler happy with w_{j - 1}
            @nexprs $k j -> d_j = @inbounds c[j + n - $k]
            for r = 2:$k
                @nexprs $k j -> w_j = d_j  # copy coefficients
                @nexprs(
                    $k,
                    j -> d_j = if j ≥ r
                        α = @inbounds (x - t[j + n - k]) /
                                      (t[j + n - r + 1] - t[j + n - k])
                        (1 - α) * w_{j - 1} + α * w_{j}
                    else
                        w_j
                    end
                )
            end
            @nexprs 1 j -> d_{$k}  # return d_k
        end
    else
        # Similar, using tuples.
        # This version is a bit slower, but not much (~15% slower in some tests).
        d = @inbounds ntuple(j -> c[j + n - k], Val(k))
        for r = 2:k
            w = d
            d = ntuple(Val(k)) do j
                if j ≥ r
                    α = @inbounds (x - t[j + n - k]) /
                                  (t[j + n - r + 1] - t[j + n - k])
                    (1 - α) * w[j - 1] + α * w[j]
                else
                    w[j]
                end
            end
        end
        d[k]
    end
end

"""
    diff(S::Spline, [deriv::Derivative = Derivative(1)]) -> Spline

Return `N`-th derivative of spline `S` as a new spline.
"""
function Base.diff(S::Spline,
                   ::Derivative{Ndiff} = Derivative(1)) where {Ndiff}
    Ndiff :: Integer
    @assert Ndiff >= 1

    u = coefficients(S)
    t = knots(S)
    k = order(S)

    if Ndiff >= k
        throw(ArgumentError(
            "cannot differentiate order $k spline $Ndiff times!"))
    end

    Base.require_one_based_indexing(u)
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
    B = BSplineBasis(BSplineOrder(k - Ndiff), t_new, augment=Val(false))

    Spline(B, view(du, (1 + Ndiff):N))
end

# Zeroth derivative: return S itself.
Base.diff(S::Spline, ::Derivative{0}) = S

"""
    integral(S::Spline)

Returns an antiderivative of the given spline as a new spline.

The algorithm is described in de Boor 2001, p. 127.
"""
function integral(S::Spline)
    u = coefficients(S)
    t = knots(S)
    k = order(S)
    Base.require_one_based_indexing(u)

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

    B = BSplineBasis(BSplineOrder(k + 1), t_int, augment=Val(false))
    Spline(B, β)
end

function get_knot_interval(t::AbstractVector, x)
    n = searchsortedlast(t, x)  # t[n] <= x < t[n + 1]
    n == 0 && return nothing    # x < t[1]

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
