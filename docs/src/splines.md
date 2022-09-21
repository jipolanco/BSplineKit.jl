# [Splines](@id Splines-api)

```@meta
CurrentModule = BSplineKit.Splines
```

## Splines

```@docs
Spline
Spline1D
coefficients
eltype(::Spline)
length(::Spline)
size(::Spline)
ndims(::Spline)
bases(::Spline)
basis(::Spline1D)
```

## Derivatives and integrals

```@docs
*
diff
integral
```

## Spline wrappers

```@docs
SplineWrapper
spline
```
