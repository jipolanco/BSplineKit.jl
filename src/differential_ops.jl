"""
    AbstractDifferentialOp

Represents a general diferential operator.
"""
abstract type AbstractDifferentialOp end

Broadcast.broadcastable(op::AbstractDifferentialOp) = Ref(op)

"""
    Derivative{n} <: AbstractDifferentialOp

Specifies the `n`-th derivative of a function.
"""
struct Derivative{n} <: AbstractDifferentialOp end

Derivative(n::Integer) = Derivative{n}()

"""
    ScaledDerivative{n} <: AbstractDifferentialOp

`n`-th derivative of a function scaled by a constant coefficient.
"""
struct ScaledDerivative{n,T<:Number} <: AbstractDifferentialOp
    α :: T
    ScaledDerivative(D::Derivative{n}, α::T) where {n,T} = new{n,T}(α)
end

Base.:*(α::Number, D::Derivative) = ScaledDerivative(D, α)
Base.:*(D::Derivative, α) = α * D

"""
    DifferentialOpSum <: AbstractDifferentialOp

Superposition of differential operators.
"""
struct DifferentialOpSum{
        Ops<:Tuple{Vararg{AbstractDifferentialOp}}} <: AbstractDifferentialOp
    ops :: Ops
    DifferentialOpSum(ops::Vararg{AbstractDifferentialOp}) = new{typeof(ops)}(ops)
end

Base.:+(ops::Vararg{AbstractDifferentialOp}) = DifferentialOpSum(ops...)
