# Basis recombination

```@meta
CurrentModule = BSplineKit.Recombinations
```

```@docs
Recombinations
```

## Recombined bases

```@docs
RecombinedBSplineBasis
Spline(::RecombinedBSplineBasis, coefs)
parent(::RecombinedBSplineBasis)
length(::RecombinedBSplineBasis)
constraints
num_constraints
num_recombined
recombination_matrix
```

## Recombination matrix

```@docs
RecombineMatrix
nzrows
```

## Internals

```@docs
NoUniqueSolutionError
```
