# B-splines and splines

```@meta
CurrentModule = BSplineKit.BSplines
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
evaluate
evaluate!
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
