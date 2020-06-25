# Differential operators

```@meta
CurrentModule = BSplineKit.DifferentialOps
```

```@docs
DifferentialOps
```

## Types

```@docs
AbstractDifferentialOp
Derivative
ScaledDerivative
DifferentialOpSum
AbstractNormalDirection
LeftNormal
RightNormal
```

## Functions

```@docs
dot(::AbstractDifferentialOp, ::AbstractNormalDirection)
max_order
```
