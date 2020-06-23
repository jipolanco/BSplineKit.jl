"""
    DifferentialOps

Module defining types describing differential operators and compositions
thereof.
"""
module DifferentialOps

export
    AbstractDifferentialOp,
    Derivative,
    ScaledDerivative,
    DifferentialOpSum,
    max_order,
    mirror

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

"""
    mirror(op::AbstractDifferentialOp)

Change the sign of odd derivatives in differential operator.

This is useful for specifying boundary conditions on the left boundary, where
the normal direction is opposite to the coordinate direction ``x``.
"""
function mirror end

max_order(ops::Vararg{AbstractDifferentialOp}) = max(max_order.(ops)...)

"""
    Derivative{n} <: AbstractDifferentialOp

Specifies the `n`-th derivative of a function.
"""
struct Derivative{n} <: AbstractDifferentialOp end

Derivative(n::Integer) = Derivative{n}()
max_order(::Derivative{n}) where {n} = n

# We always return a ScaledDerivative for type stability...
mirror(D::Derivative) = mirror(1 * D)

Base.show(io::IO, D::Derivative{n}) where {n} = print(io, "D{", n, "}")

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

Base.show(io::IO, S::ScaledDerivative) = print(io, S.α, " * ", S.D)
max_order(S::ScaledDerivative) = max_order(S.D)

function mirror(S::ScaledDerivative{n}) where {n}
    r = isodd(n) ? -1 : 1
    ScaledDerivative(S.D, r * S.α)
end

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

mirror(D::DifferentialOpSum) = DifferentialOpSum(map(mirror, D.ops)...)

Base.:+(ops::Vararg{AbstractDifferentialOp}) = DifferentialOpSum(ops...)

end
