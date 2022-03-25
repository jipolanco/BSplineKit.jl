"""
    DerivativeUnitRange{m, n} <: AbstractDifferentialOp

Specifies a range of derivatives.

# Examples

Two ways of constructing derivative ranges:

```jldoctest
julia> Derivative(2):Derivative(4)
Derivative(2:4)

julia> Derivative(2:4)
Derivative(2:4)

julia> Tuple(Derivative(2:4))
(D{2}, D{3}, D{4})
```
"""
struct DerivativeUnitRange{m, n} <: AbstractDifferentialOp end

Base.show(io::IO, ::DerivativeUnitRange{m, n}) where {m, n} =
    print(io, "Derivative($m:$n)")

@inline Base.:(:)(::Derivative{m}, ::Derivative{n}) where {m, n} =
    DerivativeUnitRange{m, n}()

@inline Derivative(r::UnitRange{<:Integer}) =
    Derivative(r.start):Derivative(r.stop)

@inline function Base.Tuple(::DerivativeUnitRange{m, n}) where {m, n}
    if m > n
        ()
    elseif m == n
        (Derivative(n),)
    else
        next = DerivativeUnitRange{m + 1, n}()
        (Derivative(m), Tuple(next)...)
    end
end
