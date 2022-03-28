# [B-splines](@id BSplines-api)

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
evaluate_all
evaluate
evaluate!
nonzero_in_segment
```

## Internals

```@docs
BSplineOrder
AugmentedKnots
augment_knots!
multiplicity
```
