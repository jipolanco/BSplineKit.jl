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
    LeftNormal,
    RightNormal,
    max_order

import LinearAlgebra: dot

"""
    AbstractDifferentialOp

Represents a general differential operator.
"""
abstract type AbstractDifferentialOp end

"""
    AbstractNormalDirection

Represents the normal direction on a given domain boundary.
"""
abstract type AbstractNormalDirection end

"""
    LeftNormal <: AbstractNormalDirection

Specifies the normal direction on the left boundary of a 1D domain.

The left normal direction goes opposite to the coordinate axis.
"""
struct LeftNormal <: AbstractNormalDirection end

"""
    RightNormal <: AbstractNormalDirection

Specifies the normal direction on the right boundary of a 1D domain.

The right normal direction is equal to that of the coordinate axis.
"""
struct RightNormal <: AbstractNormalDirection end

Broadcast.broadcastable(op::AbstractDifferentialOp) = Ref(op)
Broadcast.broadcastable(dir::AbstractNormalDirection) = Ref(dir)

"""
    max_order(op::AbstractDifferentialOp)
    max_order(ops...)

Get maximum derivative order of one or more differential operators.
"""
function max_order end

"""
    dot(op::AbstractDifferentialOp, dir::AbstractNormalDirection) -> AbstractDifferentialOp

Project derivative along a normal direction.

This should be used to convert from a [normal
derivative](https://en.wikipedia.org/wiki/Directional_derivative#Normal_derivative)
at the boundaries, to a derivative along the coordinate axes of the domain.

In practice, this returns `op` for [`RightNormal`](@ref).
For [`LeftNormal`](@ref), it multiplies the odd-order derivatives by -1.
"""
dot(op::AbstractDifferentialOp, dir::AbstractNormalDirection) = dot(dir, op)

dot(::RightNormal, op) = op

max_order(ops::Vararg{AbstractDifferentialOp}) = max(max_order.(ops)...)

"""
    Derivative{n} <: AbstractDifferentialOp

Specifies the `n`-th derivative of a function.
"""
struct Derivative{n} <: AbstractDifferentialOp end

Derivative(n::Integer) = Derivative{n}()
max_order(::Derivative{n}) where {n} = n

# We return a ScaledDerivative for type stability.
dot(dir::LeftNormal, D::Derivative) = dot(dir, 1 * D) :: ScaledDerivative

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
Base.:-(S::ScaledDerivative) = ScaledDerivative(S.D, -S.α)

Base.:(==)(x::ScaledDerivative, y::ScaledDerivative) = x.D == y.D && x.α == y.α
Base.:(==)(x::ScaledDerivative, y::Derivative) = x.D == y && x.α == 1
Base.:(==)(x::Derivative, y::ScaledDerivative) = y == x

function dot(::LeftNormal, S::ScaledDerivative{n}) where {n}
    r = isodd(n) ? -1 : 1
    ScaledDerivative(S.D, r * S.α)
end

Base.:*(α::Number, D::Derivative) = ScaledDerivative(D, α)
Base.:*(D::Derivative, α) = α * D
Base.:-(D::Derivative) = -1 * D

"""
    DifferentialOpSum <: AbstractDifferentialOp

Sum of two differential operators.
"""
struct DifferentialOpSum{
        OpA <: AbstractDifferentialOp,
        OpB <: AbstractDifferentialOp,
    } <: AbstractDifferentialOp
    a :: OpA
    b :: OpB
end

Base.:+(a::AbstractDifferentialOp, b::AbstractDifferentialOp) = DifferentialOpSum(a, b)
Base.:-(a::AbstractDifferentialOp, b::AbstractDifferentialOp) = DifferentialOpSum(a, -b)

Base.:(==)(x::DifferentialOpSum, y::DifferentialOpSum) = x.a == y.a && x.b == y.b

Base.show(io::IO, D::DifferentialOpSum) = print(io, D.a, " + ", D.b)

max_order(D::DifferentialOpSum) = max_order(D.a, D.b)

dot(dir::LeftNormal, D::DifferentialOpSum) = DifferentialOpSum(dot.(dir, (D.a, D.b))...)

end
