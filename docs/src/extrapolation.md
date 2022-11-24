# [Extrapolation](@id extrapolation-api)

```@meta
CurrentModule = BSplineKit.SplineExtrapolations

DocTestSetup = quote
    using BSplineKit
end
```

Extrapolation of splines.

## Example usage

First interpolate some data, then extrapolate the result using [`Flat`](@ref)
boundary conditions:

```jldoctest extrapolate
julia> xdata = -1:0.2:1;

julia> ydata = 2 * cospi.(xdata)
11-element Vector{Float64}:
 -2.0
 -1.618033988749895
 -0.6180339887498947
  0.6180339887498947
  1.618033988749895
  2.0
  1.618033988749895
  0.6180339887498947
 -0.6180339887498947
 -1.618033988749895
 -2.0

julia> itp = interpolate(xdata, ydata, BSplineOrder(4))
SplineInterpolation containing the 11-element Spline{Float64}:
 basis: 11-element BSplineBasis of order 4, domain [-1.0, 1.0]
 order: 4
 knots: [-1.0, -1.0, -1.0, -1.0, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 1.0, 1.0, 1.0, 1.0]
 coefficients: [-2.0, -2.03018, -1.10276, 0.660559, 1.7279, 2.13605, 1.7279, 0.660559, -1.10276, -2.03018, -2.0]
 interpolation points: -1.0:0.2:1.0

julia> ext = extrapolate(itp, Flat())
SplineExtrapolation containing the 11-element Spline{Float64}:
 basis: 11-element BSplineBasis of order 4, domain [-1.0, 1.0]
 order: 4
 knots: [-1.0, -1.0, -1.0, -1.0, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 1.0, 1.0, 1.0, 1.0]
 coefficients: [-2.0, -2.03018, -1.10276, 0.660559, 1.7279, 2.13605, 1.7279, 0.660559, -1.10276, -2.03018, -2.0]
 extrapolation method: Flat()

julia> ext(1.0)
-2.0

julia> ext(1.1)
-2.0
```

## Functions

```@docs
extrapolate
```

## Extrapolation methods

```@docs
Flat
Smooth
```

## Internals

```@docs
AbstractExtrapolationMethod
SplineExtrapolation
```
