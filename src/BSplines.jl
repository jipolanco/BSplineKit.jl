"""
    module BSplines

Approximate and interpolate functions using B-splines.

## Notation

Different definitions of the spline order are used in the literature and in
numerical packages.
Here we use the definition used for instance on
[Wikipedia](https://en.wikipedia.org/wiki/B-spline#Introduction) or in
the [GSL docs](https://www.gnu.org/software/gsl/doc/html/bspline.html):
a spline of order `k` is a piecewise polynomial of degree `k - 1`.
Hence, for instance, cubic splines correspond to `k = 4`.
"""
module BSplines

export Collocation

export BSplineBasis, Spline
export knots, order
export augment_knots
export evaluate_bspline, evaluate_bspline!
export coefficients

using Reexport
using StaticArrays: MVector

"""
    BSplineBasis{k}

Grid for splines of order `k`.

---

    BSplineBasis(k::Union{Val,Integer}, knots::Vector; augment=true)

Create B-spline basis of order `k` with the given knots.

If `augment=true` (default), knots will be "augmented" so that both knot ends
have multiplicity `k`. See also [`augment_knots`](@ref).
"""
struct BSplineBasis{k, T}  # k: B-spline order
    N :: Int             # number of B-splines ("resolution")
    t :: Vector{T}       # knots (length = N + k)
    function BSplineBasis(::Val{k}, knots::AbstractVector{T};
                          augment=true) where {k,T}
        k :: Integer
        if k <= 0
            throw(ArgumentError("B-spline order must be k ≥ 1"))
        end
        t = augment ? augment_knots(knots, k) : knots
        N = length(t) - k
        new{k, T}(N, t)
    end
end

@inline BSplineBasis(k::Integer, args...; kwargs...) =
    BSplineBasis(Val(k), args...; kwargs...)

"""
    Spline

Spline function.

---

    Spline(b::BSplineBasis, coefs::AbstractVector)

Construct a spline from a B-spline basis and a vector of B-spline coefficients.
"""
struct Spline{k,  # B-spline order
              Basis <: BSplineBasis,
              T <: AbstractFloat,
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
    T = eltype(du)

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
    # Note that the spline has `2k * Ndiff` fewer knots, and `k * Ndiff` fewer
    # B-splines.
    N = length(u)
    Nt = length(t)
    t_new = view(t, (1 + Ndiff):(Nt - Ndiff))
    B = BSplineBasis(Val(k - Ndiff), t_new, augment=false)

    Spline(B, view(du, (1 + Ndiff):N))
end

@inline Base.diff(S::Spline, Ndiff::Integer) = diff(S, Val(Ndiff))

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

# Make BSplineBasis behave as scalars when broadcasting.
Broadcast.broadcastable(B::BSplineBasis) = Ref(B)

# Same boundary conditions at the two borders.
_make_boundary_diff(d::Integer) = _make_boundary_diff((d, d))
_make_boundary_diff(d::NTuple{2,<:Integer}) = Int.(d)

"""
    length(g::BSplineBasis)

Returns the number of B-splines composing a spline.
"""
Base.length(g::BSplineBasis) = g.N
Base.size(g::BSplineBasis) = (g.N, )

"""
    knots(g::BSplineBasis)
    knots(g::Spline)

Returns the knots of the spline grid.
"""
knots(g::BSplineBasis) = g.t

"""
    order(::Type{BSplineBasis})
    order(::Type{Spline})

Returns order of B-splines.
"""
order(::Type{<:BSplineBasis{k}}) where {k} = k
order(b::BSplineBasis) = order(typeof(b))

"""
    augment_knots(knots::AbstractVector, k::Integer)

Modifies the input knots to make sure that the first and last knot have
the multiplicity `k` for splines of order `k`.

Similar to [`augknt`](https://www.mathworks.com/help/curvefit/augknt.html) in
Matlab.
"""
function augment_knots(knots::AbstractVector{T},
                       k::Integer) :: Vector{T} where {T}
    N = length(knots)

    # Determine multiplicity of first and last knots in input.
    m_first = multiplicity(knots, 1)
    m_last = multiplicity(knots, N)

    if m_first == m_last == k
        return knots  # nothing to do
    end

    N_inner = N - m_first - m_last
    Nnew = N_inner + 2k
    t = Vector{float(T)}(undef, Nnew)  # augmented knots

    t_first = knots[1]
    t_last = knots[end]

    t[1:k] .= t_first
    t[(Nnew - k + 1):Nnew] .= t_last
    t[(k + 1):(k + N_inner)] .= @view knots[(m_first + 1):(m_first + N_inner)]

    t
end

"""
    evaluate_bspline(B::BSplineBasis, i::Integer, x, [T=Float64])

Evaluate i-th B-spline in the given basis at `x` (can be a coordinate or a
vector of coordinates).

See also [`evaluate_bspline!`](@ref).
"""
function evaluate_bspline(B::BSplineBasis, i::Integer, x::Real,
                          ::Type{T} = Float64) where {T}
    N = length(B)
    if !(1 <= i <= N)
        throw(DomainError(i, "B-spline index must be in 1:$N"))
    end
    evaluate_bspline(Val(order(B)), knots(B), i, x, T)
end

evaluate_bspline(B::BSplineBasis, i, x::AbstractVector, args...) =
    evaluate_bspline.(B, i, x, args...)

"""
    evaluate_bspline!(b::AbstractVector, B::BSplineBasis, i::Integer,
                      x::AbstractVector)

Evaluate i-th B-spline at positions `x` and write result to `b`.

See also [`evaluate_bspline`](@ref).
"""
function evaluate_bspline!(b::AbstractVector{T}, B::BSplineBasis, i,
                           x::AbstractVector) where {T}
    broadcast!(x -> evaluate_bspline(B, i, x, T), b, x)
end

# Specialisation for first order B-splines.
function evaluate_bspline(::Val{1}, t::AbstractVector, i::Integer, x::Real,
                          ::Type{T}) where {T}
    # Local support of the B-spline.
    @inbounds ta = t[i]
    @inbounds tb = t[i + 1]

    in_local_support(x, t, ta, tb) ? one(T) : zero(T)
end

# General case of order k >= 2.
function evaluate_bspline(::Val{k}, t::AbstractVector, i::Integer, x::Real,
                          ::Type{T}) where {T,k}
    k::Int
    @assert k >= 2

    # Local support of the B-spline.
    @inbounds ta = t[i]
    @inbounds tb = t[i + k]

    if !in_local_support(x, t, ta, tb)
        return zero(T)
    end

    @inbounds ta1 = t[i + 1]
    @inbounds tb1 = t[i + k - 1]

    # Recursively evaluate lower-order B-splines.
    y = zero(T)

    if tb1 != ta
        y += evaluate_bspline(Val(k - 1), t, i, x, T) * (x - ta) / (tb1 - ta)
    end

    if ta1 != tb
        y += evaluate_bspline(Val(k - 1), t, i + 1, x, T) * (tb - x) / (tb - ta1)
    end

    y
end


# In principle, the support of a B-spline is [ta, tb[.
# The exception is when tb is the last point in the domain, in which case we
# include it in the interval (otherwise the right boundary always has zero value).
@inline in_local_support(x, t, ta, tb) = (ta <= x < tb) || (x == tb == t[end])

"""
    multiplicity(knots, i)

Determine multiplicity of knot `knots[i]`.
"""
function multiplicity(knots::AbstractVector, i)
    @assert Base.require_one_based_indexing(knots)
    v = knots[i]
    m = 1

    # Check in both directions
    j = i - 1
    while j > 0 && knots[j] == v
        j -= 1
        m += 1
    end

    j = i + 1
    N = length(knots)
    while j <= N && knots[j] == v
        j += 1
        m += 1
    end

    m
end

include("Collocation.jl")

@reexport using .Collocation

end # module
