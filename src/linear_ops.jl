"""
    AbstractLinearOp

Represents a linear operator on a function.
"""
abstract type AbstractLinearOp end

Broadcast.broadcastable(op::AbstractLinearOp) = Ref(op)

"""
    Derivative{n} <: AbstractLinearOp

Specifies the `n`-th derivative of a function.
"""
struct Derivative{n} <: AbstractLinearOp end

@inline Derivative(n::Integer) = Derivative{n}()

"""
    ScaledDerivative{n} <: AbstractLinearOp

`n`-th derivative of a function scaled by a constant coefficient.
"""
struct ScaledDerivative{n,T<:Number} <: AbstractLinearOp
    α :: T
    ScaledDerivative(D::Derivative{n}, α::T) where {n,T} = new{n,T}(α)
end

Base.:*(α::Number, D::Derivative) = ScaledDerivative(D, α)
Base.:*(D::Derivative, α) = α * D

"""
    LinearOpSum <: AbstractLinearOp

Superposition of linear operations on functions.
"""
struct LinearOpSum{Ops<:Tuple{Vararg{AbstractLinearOp}}} <: AbstractLinearOp
    ops :: Ops
    LinearOpSum(ops::Vararg{AbstractLinearOp}) = new{typeof(ops)}(ops)
end

Base.:+(ops::Vararg{AbstractLinearOp}) = LinearOpSum(ops...)
