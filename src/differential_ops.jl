"""
    AbstractDifferentialOp

Represents a general diferential operator.
"""
abstract type AbstractDifferentialOp end

Broadcast.broadcastable(op::AbstractDifferentialOp) = Ref(op)

"""
    max_order(op::AbstractDifferentialOp)
    max_order(ops...)

Get maximum derivative order of one or more differential operators.
"""
function max_order end

max_order(ops::Vararg{AbstractDifferentialOp}) = max(max_order.(ops)...)

"""
    Derivative{n} <: AbstractDifferentialOp

Specifies the `n`-th derivative of a function.
"""
struct Derivative{n} <: AbstractDifferentialOp end

Derivative(n::Integer) = Derivative{n}()
max_order(::Derivative{n}) where {n} = n

Base.show(io::IO, D::Derivative{n}) where {n} = print(io, "D{", n, "}")

get_orders(::Derivative{n}, etc...) where {n} = (n, get_orders(etc...)...)
get_orders() = ()

"""
    ScaledDerivative{n} <: AbstractDifferentialOp

`n`-th derivative of a function scaled by a constant coefficient.
"""
struct ScaledDerivative{n,T<:Number} <: AbstractDifferentialOp
    D :: Derivative{n}
    α :: T
    function ScaledDerivative(D::Derivative{n}, α::T) where {n,T}
        # If we allow α = 0, then the output of functions such as max_order
        # is not a compile-time constant.
        iszero(α) && throw(ArgumentError("scale factor cannot be zero"))
        new{n,T}(D, α)
    end
end

Base.show(io::IO, s::ScaledDerivative) = print(io, s.α, " * ", s.D)

max_order(s::ScaledDerivative) = max_order(s.D)
get_orders(s::ScaledDerivative, etc...) = get_orders(s.D, etc...)

Base.:*(α::Number, D::Derivative) = ScaledDerivative(D, α)
Base.:*(D::Derivative, α) = α * D

"""
    DifferentialOpSum <: AbstractDifferentialOp

Sum of differential operators.
"""
struct DifferentialOpSum{
        Ops<:Tuple{Vararg{AbstractDifferentialOp}}} <: AbstractDifferentialOp
    ops :: Ops
    DifferentialOpSum(ops::Vararg{AbstractDifferentialOp}) = new{typeof(ops)}(ops)
end

Base.show(io::IO, D::DifferentialOpSum) = join(io, D.ops, " + ")

max_order(D::DifferentialOpSum) = max_order(D.ops...)

Base.:+(ops::Vararg{AbstractDifferentialOp}) = DifferentialOpSum(ops...)
