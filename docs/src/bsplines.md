# B-splines and splines

```@meta
CurrentModule = BSplineKit
```

## B-splines

### Types

```@docs
AbstractBSplineBasis
BSplineBasis
BSpline
BSplineOrder
```

### Functions

```@docs
boundaries
order
length
support
common_support
evaluate_bspline
evaluate_bspline!
```

#### Knots

```@docs
augment_knots
knots
multiplicity
```

## Splines

### Types

```@docs
Spline
```

### Functions

```@docs
coefficients
diff
integral
```
