using StaticArrays: MVector

const BasisTuple{N} = Tuple{Vararg{AbstractBSplineBasis, N}} where {N}

"""
    Spline{T, N}

Represents an ``N``-dimensional spline function which returns values of type `T`.

For ``N ≥ 2``, this is a tensor-product spline.
That is, the spline space is given by the tensor product of ``N``
one-dimensional spline spaces.

---

    Spline(B::AbstractBSplineBasis, coefs::AbstractVector)

Construct a 1D spline from a B-spline basis and a vector of B-spline coefficients.

The spline can be then evaluated at any point `x` within the domain.

# Examples

## 1D splines

```jldoctest; filter = r"coefficients: \\[.*\\]"
julia> B = BSplineBasis(BSplineOrder(4), -1:0.2:1);

julia> coefs = rand(length(B));

julia> S = Spline(B, coefs)
13-element Spline{Float64, 1}:
 basis: 13-element BSplineBasis of order 4, domain [-1.0, 1.0]
 order: 4
 knots: [-1.0, -1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
 coefficients: [0.815921, 0.076499, 0.433472, 0.672844, 0.468371, 0.348423, 0.868621, 0.0831675, 0.369734, 0.401199, 0.990734, 0.565907, 0.984855]

julia> S(0.42)  # evaluate spline
0.7039651871881707

julia> S′ = Derivative(1) * S  # spline derivative
12-element Spline{1, Float64}:
 basis: 12-element BSplineBasis of order 3, domain [-1.0, 1.0]
 order: 3
 knots: [-1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0]
 coefficients: [1.48161, 0.775234, -1.48726, 0.280723, 0.792347, 1.97356, -3.15552, 4.37379, -1.88468, -1.91048, 2.7976, 4.1116]

julia> Sint = integral(S)  # spline integral
14-element Spline{Float64}:
 basis: 14-element BSplineBasis of order 5, domain [-1.0, 1.0]
 order: 5
 knots: [-1.0, -1.0, -1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0]
 coefficients: [0.0, 0.00680366, 0.0302884, 0.0810201, 0.0891722, 0.108553, 0.159628, 0.289645, 0.293442, 0.47219, 0.575551, 0.595757, 0.64653, 0.685621]
```

## Multidimensional (tensor-product) splines

```jldoctest; filter = r"coefficients: \\[.*\\]"
julia> Bx = BSplineBasis(BSplineOrder(4), -1:0.2:1)
13-element BSplineBasis of order 4, domain [-1.0, 1.0]
 knots: [-1.0, -1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]

julia> By = BSplineBasis(BSplineOrder(6), 0:0.05:1)
25-element BSplineBasis of order 6, domain [0.0, 1.0]
 knots: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2  …  0.8, 0.85, 0.9, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

julia> coefs = rand(length(Bx), length(By));

julia> S = Spline((Bx, By), coefs)
13×25 Spline{Float64, 2}:
 bases:
   (1) 13-element BSplineBasis of order 4, domain [-1.0, 1.0]
   (2) 25-element BSplineBasis of order 6, domain [0.0, 1.0]
 knots:
   (1) [-1.0, -1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
   (2) [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2  …  0.8, 0.85, 0.9, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
 coefficients: [0.523309 0.85789 … 0.503075 0.53592; 0.890179 0.909717 … 0.785562 0.746001; … ; 0.637745 0.583284 … 0.175474 0.266573; 0.776247 0.280784 … 0.390669 0.619563]
```

---

    Spline{T = Float64}(undef, B::AbstractBSplineBasis)
    Spline{T = Float64}(undef, Bx, By, …)

Construct a spline with uninitialised vector of coefficients.

In the second case, a tensor-product multidimensional spline is constructed.
"""
struct Spline{
        T,  # type of coefficient (e.g. Float64, ComplexF64)
        N,  # spline dimension
        Bases <: BasisTuple{N},
        CoefArray <: AbstractArray{T, N},
    }
    bases :: Bases
    coefs :: CoefArray

    function Spline(
            Bs::BasisTuple{N},
            coefs::AbstractArray{T, N},
        ) where {T, N}
        size(coefs) == length.(Bs) ||
            throw(ArgumentError("wrong number of coefficients"))
        Bases = typeof(Bs)
        CoefArray = typeof(coefs)
        @assert all(B -> order(B) ≥ 1, Bs)
        new{T, N, Bases, CoefArray}(Bs, coefs)
    end
end

Spline(B::AbstractBSplineBasis, args...) = Spline((B,), args...)

"""
    Spline1D

Alias for `Spline{T, 1}`, representing a one-dimensional spline.
"""
const Spline1D = Spline{<:Any, 1}

Broadcast.broadcastable(S::Spline) = Ref(S)

Base.copy(S::Spline) = Spline(bases(S), copy(coefficients(S)))

function Base.show(io::IO, S::Spline)
    T = eltype(S)
    N = ndims(S)
    println(io, Base.dims2string(size(S)), ' ', nameof(typeof(S)), "{$T, $N}", ':')
    if N == 1
        print(io, " basis: ")
        summary(io, first(bases(S)))
        println(io, "\n order: ", first(orders(S)))
        let io = IOContext(io, :compact => true, :limit => true)
            println(io, " knots: ", first(knots(S)))
            print(io, " coefficients: ", coefficients(S))
        end
    else
        print(io, " bases:\n")
        for (n, B) ∈ enumerate(bases(S))
            print(io, "   ($n) ")
            summary(io, B)
            print(io, '\n')
        end
        let io = IOContext(io, :compact => true, :limit => true)
            print(io, " knots:\n")
            for (n, ts) ∈ enumerate(knots(S))
                print(io, "   ($n) ", ts, '\n')
            end
            print(io, " coefficients: ", coefficients(S))
        end
    end
    nothing
end

Base.:(==)(P::Spline, Q::Spline) =
    bases(P) == bases(Q) && coefficients(P) == coefficients(Q)

Base.isapprox(P::Spline, Q::Spline; kwargs...) =
    bases(P) == bases(Q) &&
    isapprox(coefficients(P), coefficients(Q); kwargs...)

function Spline{T}(init, Bs::Vararg{AbstractBSplineBasis}) where {T}
    coefs = Array{T}(init, map(length, Bs))
    Spline(Bs, coefs)
end

Spline(init, B::AbstractBSplineBasis) = Spline{Float64}(init, B)

@deprecate(
    Spline(init, B::AbstractBSplineBasis, ::Type{T}) where {T},
    Spline{T}(init, B),
)

parent_spline(S::Spline) = parent_spline(basis(S), S)
parent_spline(::BSplineBasis, S::Spline) = S

"""
    coefficients(S::Spline{T,N}) -> AbstractArray{T,N}

Returns the array of B-spline coefficients of the spline.
"""
coefficients(S::Spline) = S.coefs

"""
    length(S::Spline)

Returns the number of coefficients in the spline.

For 1D splines, this is equal to the number of basis functions,
`length(first(bases(S)))`.
"""
Base.length(S::Spline) = length(coefficients(S))

"""
    size(S::Spline)

Same as `size(coefficients(S))`.
"""
Base.size(S::Spline) = size(coefficients(S))

"""
    eltype(::Type{<:Spline})
    eltype(S::Spline)

Returns type of element returned when evaluating the [`Spline`](@ref).
"""
Base.eltype(::Type{<:Spline{T}}) where {T} = T


Base.ndims(::Type{<:Spline{T, N}}) where {T, N} = N
Base.ndims(S::Spline) = ndims(typeof(S))

bases(S::Spline) = S.bases

knots(S::Spline) = map(knots, bases(S))
orders(S::Spline) = map(order, bases(S))

basis(S::Spline1D) = first(bases(S))
order(S::Spline1D) = first(orders(S))

# TODO allow evaluating derivatives at point `x` (should be much cheaper than
# constructing a new Spline for the derivative)
(S::Spline1D)(x) = _evaluate(basis(S), S, x)
(S::Spline)(xs...) = _evaluate_tensor_product(bases(S), S, xs)

function _evaluate(::BSplineBasis, S::Spline1D, x)
    T = eltype(S)
    t = knots(S)
    n = knot_interval(t, x)
    n === nothing && return zero(T)  # x is outside of knot domain
    k = order(S)
    spline_kernel(coefficients(S), t, n, x, BSplineOrder(k))
end

# Fallback, if the basis is not a regular BSplineBasis
_evaluate(::AbstractBSplineBasis, S::Spline1D, x) = parent_spline(S)(x)

function spline_kernel(
        c::AbstractVector{T}, t, n, x, ::BSplineOrder{k},
    ) where {T,k}
    # Algorithm adapted from https://en.wikipedia.org/wiki/De_Boor's_algorithm
    if @generated
        ex = quote
            @nexprs $k j -> d_j = @inbounds c[j + n - $k]
        end
        for r = 2:k, j = k:-1:r
            d_j = Symbol(:d_, j)
            d_p = Symbol(:d_, j - 1)
            jk = j - k
            jr = j - r
            ex = quote
                $ex
                α = @inbounds (x - t[$jk + n]) / (t[$jr + n + 1] - t[$jk + n])
                $d_j = $T((1 - α) * $d_p + α * $d_j)
            end
        end
        d_k = Symbol(:d_, k)
        quote
            $ex
            return $d_k
        end
    else
        # Similar using MVector (a bit slower than @generated version).
        spline_kernel_alt(c, t, n, x, BSplineOrder(k))
    end
end

function spline_kernel_alt(
        c::AbstractVector{T}, t, n, x, ::BSplineOrder{k},
    ) where {T, k}
    d = MVector(ntuple(j -> @inbounds(c[j + n - k]), Val(k)))
    @inbounds for r = 2:k
        dprev = d[r - 1]
        for j = r:k
            α = (x - t[j + n - k]) / (t[j + n - r + 1] - t[j + n - k])
            dtmp = dprev
            dprev = d[j]
            d[j] = (1 - α) * dtmp + α * dprev
        end
    end
    @inbounds d[k]
end

# TODO optimise!
# Note that the evaluation of B-splines is by far the most expensive, so I'm not
# sure we can do much better.
function _evaluate_tensor_product(
        Bs::BasisTuple{N}, S::Spline{T, N}, xs::Tuple{Vararg{Any,N}},
    ) where {T, N}
    @assert N ≥ 2  # there's a separate function for the case N = 1
    @assert Bs === bases(S)
    coefs = coefficients(S)
    ks = orders(S)
    # Evaluate all B-splines: ((i, bxs), (j, bys), …)
    bsp = map((B, x) -> B(x), Bs, xs)
    inds_base = CartesianIndex(map(first, bsp) .+ 1)  # (i + 1, j + 1, …)
    bsp_values = map(last, bsp)  # (bxs, bys, …)
    val = zero(T)
    inds = map(k -> Base.OneTo(k), ks)
    @inbounds for δs ∈ CartesianIndices(inds)
        I = inds_base - δs
        coef = coefs[I]
        bs = getindex.(bsp_values, Tuple(δs))
        val += coef * prod(bs)
    end
    val
end

"""
    *(op::Derivative, S::Spline1D) -> Spline1D

Returns `N`-th derivative of spline `S` as a new spline.

See also [`diff`](@ref).

# Examples

```jldoctest; filter = r"coefficients: \\[.*\\]"
julia> B = BSplineBasis(BSplineOrder(4), -1:0.2:1);

julia> S = Spline(B, rand(length(B)))
13-element Spline{Float64}:
 basis: 13-element BSplineBasis of order 4, domain [-1.0, 1.0]
 order: 4
 knots: [-1.0, -1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
 coefficients: [0.461501, 0.619799, 0.654451, 0.667213, 0.334672, 0.618022, 0.967496, 0.900014, 0.611195, 0.469467, 0.221618, 0.80084, 0.269533]

julia> Derivative(0) * S === S
true

julia> Derivative(1) * S
12-element Spline{Float64}:
 basis: 12-element BSplineBasis of order 3, domain [-1.0, 1.0]
 order: 3
 knots: [-1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0]
 coefficients: [2.37448, 0.259885, 0.0638088, -1.6627, 1.41675, 1.74737, -0.33741, -1.44409, -0.708643, -1.23925, 4.34416, -7.9696]

julia> Derivative(2) * S
11-element Spline{Float64}:
 basis: 11-element BSplineBasis of order 2, domain [-1.0, 1.0]
 order: 2
 knots: [-1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0]
 coefficients: [-21.146, -0.98038, -8.63255, 15.3972, 1.65313, -10.4239, -5.53341, 3.67724, -2.65301, 27.917, -123.138]
```
"""
Base.:*(op::Derivative, S::Spline1D) = _diff(basis(S), S, op)

"""
    diff(S::Spline1D, [op::Derivative = Derivative(1)]) -> Spline1D

Same as `op * S`.

Returns `N`-th derivative of spline `S` as a new spline.
"""
Base.diff(S::Spline, op = Derivative(1)) = op * S

_diff(::AbstractBSplineBasis, S, etc...) = diff(parent_spline(S), etc...)

_diff(::BSplineBasis, S::Spline1D, ::Derivative{0}) = S

function _diff(
        ::BSplineBasis, S::Spline1D, ::Derivative{Ndiff} = Derivative(1),
    ) where {Ndiff}
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
    du = similar(u)
    copy!(du, u)

    @inbounds for m = 1:Ndiff, i in Iterators.Reverse(eachindex(du))
        dt = t[i + k - m] - t[i]
        if iszero(dt) || i == 1
            # In this case, the B-spline that this coefficient is
            # multiplying is zero everywhere, so we can set this to zero.
            # From de Boor (2001, p. 117): "anything times zero is zero".
            du[i] = zero(eltype(du))
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
    B = BSplineBasis(BSplineOrder(k - Ndiff), t_new; augment = Val(false))

    Spline(B, view(du, (1 + Ndiff):N))
end

# Zeroth derivative: return S itself.
Base.diff(S::Spline1D, ::Derivative{0}) = S

"""
    integral(S::Spline1D)

Returns an antiderivative of the given spline as a new spline.

The algorithm is described in de Boor 2001, p. 127.
"""
integral(S::Spline1D) = _integral(basis(S), S)

_integral(::AbstractBSplineBasis, S, etc...) = integral(parent_spline(S), etc...)

function _integral(::BSplineBasis, S::Spline1D)
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
    β[1] = zero(eltype(β))

    @inbounds for i in eachindex(u)
        m = i + 1
        β[m] = zero(eltype(β))
        for j = 1:i
            β[m] += u[j] * (t[j + k] - t[j]) / k
        end
    end

    B = BSplineBasis(BSplineOrder(k + 1), t_int; augment = Val(false))
    Spline(B, β)
end

function knot_interval(t::AbstractVector, x)
    n = searchsortedlast(t, x)  # t[n] <= x < t[n + 1]
    n == 0 && return nothing    # x < t[1]

    Nt = length(t)

    if n == Nt  # i.e. if x >= t[end]
        t_last = t[n]
        x > t_last && return nothing
        # If x is exactly on the last knot, decrease the index as necessary.
        while t[n] == t_last
            n -= one(n)
        end
    end

    n
end
