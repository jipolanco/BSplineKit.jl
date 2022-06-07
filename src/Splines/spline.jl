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
