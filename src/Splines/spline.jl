using StaticArrays: MVector

const BasisTuple{N} = Tuple{Vararg{AbstractBSplineBasis, N}} where {N}

"""
    Spline{N, T}

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

```jldoctest; setup = :( Random.seed!(42) )
julia> B = BSplineBasis(BSplineOrder(4), -1:0.2:1);

julia> coefs = rand(length(B));

julia> S = Spline(B, coefs)
13-element Spline{1, Float64}:
 basis: 13-element BSplineBasis of order 4, domain [-1.0, 1.0]
 order: 4
 knots: [-1.0, -1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
 coefficients: [0.173575, 0.321662, 0.258585, 0.166439, 0.527015, 0.483022, 0.390663, 0.802763, 0.721983, 0.372347, 0.0301856, 0.0793339, 0.663758]

julia> S(0.42)  # evaluate spline
0.6543543311366747

julia> S′ = Derivative(1) * S  # spline derivative
12-element Spline{1, Float64}:
 basis: 12-element BSplineBasis of order 3, domain [-1.0, 1.0]
 order: 3
 knots: [-1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0]
 coefficients: [2.22131, -0.473071, -0.460734, 1.80288, -0.219964, -0.461794, 2.0605, -0.403899, -1.74818, -1.71081, 0.368613, 8.76636]

julia> Sint = integral(S)  # spline integral
14-element Spline{1, Float64}:
 basis: 14-element BSplineBasis of order 5, domain [-1.0, 1.0]
 order: 5
 knots: [-1.0, -1.0, -1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0]
 coefficients: [0.0, 0.00867873, 0.0408449, 0.0796327, 0.11292, 0.218323, 0.314928, 0.393061, 0.553613, 0.69801, 0.772479, 0.777007, 0.78494, 0.818128]
```

## Multidimensional (tensor-product) splines

```jldoctest; setup = :( Random.seed!(42) )
julia> Bx = BSplineBasis(BSplineOrder(4), -1:0.2:1)
13-element BSplineBasis of order 4, domain [-1.0, 1.0]
 knots: [-1.0, -1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]

julia> By = BSplineBasis(BSplineOrder(6), 0:0.05:1)
25-element BSplineBasis of order 6, domain [0.0, 1.0]
 knots: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2  …  0.8, 0.85, 0.9, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

julia> coefs = rand(length(Bx), length(By));

julia> S = Spline((Bx, By), coefs)
13×25 Spline{2, Float64}:
 bases:
   (1) 13-element BSplineBasis of order 4, domain [-1.0, 1.0]
   (2) 25-element BSplineBasis of order 6, domain [0.0, 1.0]
 knots:
   (1) [-1.0, -1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
   (2) [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2  …  0.8, 0.85, 0.9, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
 coefficients: [0.173575 0.183429 … 0.00993008 0.320248; 0.321662 0.347508 … 0.251083 0.346025; … ; 0.433914 0.895838 … 0.483176 0.0793339; 0.211228 0.580212 … 0.425628 0.663758]

julia> S(-0.32, 0.54)
0.38520723534067336
```

---

    Spline(undef, B::AbstractBSplineBasis, [T = Float64])
    Spline(undef, (Bx, By, …), [T = Float64])

Construct a spline with uninitialised vector of coefficients.

In the second case, a tensor-product multidimensional spline is constructed.

The optional parameter `T` corresponds to the returned type when the spline is
evaluated.
"""
struct Spline{
        N,  # spline dimension
        T,  # type of coefficient (e.g. Float64, ComplexF64)
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
        new{N, T, Bases, CoefArray}(Bs, coefs)
    end
end

Spline(B::AbstractBSplineBasis, args...) = Spline((B,), args...)

Broadcast.broadcastable(S::Spline) = Ref(S)

Base.copy(S::Spline) = Spline(bases(S), copy(coefficients(S)))

function Base.show(io::IO, S::Spline)
    T = eltype(S)
    N = ndims(S)
    println(io, Base.dims2string(size(S)), ' ', nameof(typeof(S)), "{$N, $T}", ':')
    if N == 1
        print(io, " basis: ")
        summary(io, first(bases(S)))
        println(io, "\n order: ", first(orders(S)))
        let io = IOContext(io, :compact => true, :limit => true)
            println(io, " knots: ", knots(S))
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
            for (n, ts) ∈ enumerate(allknots(S))
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

function Spline(init, Bs::Tuple{Vararg{AbstractBSplineBasis}}, ::Type{T}) where {T}
    coefs = Array{T}(init, map(length, Bs))
    Spline(Bs, coefs)
end

Spline(init, Bs::Tuple{Vararg{AbstractBSplineBasis}}) = Spline(init, Bs, Float64)

# For 1D splines
Spline(init, B::AbstractBSplineBasis, args...) = Spline(init, (B,), args...)

"""
    coefficients(S::Spline{N,T}) -> AbstractArray{T,N}

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
Base.eltype(::Type{<:Spline{N, T}}) where {N, T} = T
Base.eltype(S::Spline) = eltype(typeof(S))

"""
    ndims(::Type{<:Spline})
    ndims(S::Spline)

Returns the dimensionality of the spline.
"""
Base.ndims(S::Spline) = ndims(typeof(S))
Base.ndims(::Type{<:Spline{N}}) where {N} = N

"""
    bases(S::Spline) -> (B₁, B₂, …)

Returns the B-spline bases associated to the spline.

The number of bases is equal to the dimensionality of the spline.
In particular, for 1D splines, this returns a one-element tuple.

See also [`basis(::Spline1D)`](@ref).
"""
bases(S::Spline) = S.bases

allknots(S::Spline) = map(knots, bases(S))
orders(S::Spline) = map(order, bases(S))
