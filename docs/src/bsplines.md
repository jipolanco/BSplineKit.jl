# B-splines

```@meta
CurrentModule = BSplineKit.BSplines
```

```@docs
BSplines
```

## B-spline bases

```@docs
AbstractBSplineBasis
BSplineBasis
boundaries
order
knots
getindex
length(::BSplineBasis)
```

## Basis functions

```@docs
BasisFunction
support
common_support
evaluate
evaluate!
```

## Internals

```@docs
BSplineOrder
AugmentedKnots
augment_knots
multiplicity
```
