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

## Periodic B-spline bases

```@docs
PeriodicBSplineBasis
PeriodicKnots
period
```

## Basis functions

```@docs
BasisFunction
support
common_support
find_knot_interval
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
basis_to_array_index
has_parent_basis
multiplicity
```
