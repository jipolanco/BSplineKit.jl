# BSplineKit.jl

Tools for B-spline based Galerkin and collocation methods in Julia.

⚠ This package is currently in an **experimental** stage and under active
development.

Examples are coming in the future!

## Features

At this moment, this package provides:

- B-spline bases of arbitrary order on uniform and non-uniform grids;

- evaluation of splines and derivatives;

- spline interpolation of data on grids;

- basis recombination, for generating bases satisfying homogeneous boundary
  conditions.
  A wide variety of boundary conditions is supported, including Dirichlet,
  Neumann, Robin, and more complex variants;

- [banded](https://github.com/JuliaMatrices/BandedMatrices.jl) Galerkin and
  collocation matrices for solving differential equations, using B-spline and
  recombined bases;

- efficient "banded" 3D arrays as an extension of banded matrices.
  These can store 3D tensors associated to quadratic terms in Galerkin methods.

## References

- C. de Boor, *A Practical Guide to Splines*. New York: Springer-Verlag, 1978.

- J. P. Boyd, *Chebyshev and Fourier Spectral Methods*, Second Edition.
  Mineola, N.Y: Dover Publications, 2001.

- O. Botella and K. Shariff, *B-spline Methods in Fluid Dynamics*, Int. J. Comput.
  Fluid Dyn. 17, 133 (2003).
