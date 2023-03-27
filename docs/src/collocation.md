# [Collocation tools](@id collocation-api)

```@meta
CurrentModule = BSplineKit
```

## Collocation points

```@docs
collocation_points
collocation_points!
```

### Point selection methods

```@docs
Collocation.SelectionMethod
Collocation.AvgKnots
Collocation.SameAsKnots
```

## Matrices

```@docs
CollocationMatrix
collocation_matrix
collocation_matrix!
Collocation.lu
Collocation.lu!
```

## Internals

```@docs
Collocation.CyclicTridiagonalMatrix
LinearAlgebra.ldiv!
```
