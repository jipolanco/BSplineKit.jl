# Differential operators

```@meta
CurrentModule = BSplineKit.DifferentialOps
```

```@docs
DifferentialOps
```

## Operators

```@docs
AbstractDifferentialOp
Derivative
DerivativeUnitRange
ScaledDerivative
DifferentialOpSum
max_order
```

## Projections

```@docs
AbstractNormalDirection
LeftNormal
RightNormal
dot(::AbstractDifferentialOp, ::AbstractNormalDirection)
```
