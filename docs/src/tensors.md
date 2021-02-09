# Banded tensors

```@meta
CurrentModule = BSplineKit.BandedTensors
```

## Banded tensors

```@docs
BandedTensor3D
bandshift
bandwidth
band_indices
```

## Slices

```@docs
SubMatrix
setindex!
```

## Linear algebra

```@docs
dot(::AbstractVector, ::SubMatrix, ::AbstractVector)
BandedTensors.muladd!
```
