# BSplineKit.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jipolanco.github.io/BSplineKit.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jipolanco.github.io/BSplineKit.jl/dev/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5150350.svg)](https://doi.org/10.5281/zenodo.5150350)

[![Build Status](https://github.com/jipolanco/BSplineKit.jl/workflows/CI/badge.svg)](https://github.com/jipolanco/BSplineKit.jl/actions)
[![Coverage](https://codecov.io/gh/jipolanco/BSplineKit.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jipolanco/BSplineKit.jl)

Tools for B-spline based Galerkin and collocation methods in Julia.

## Features

This package provides:

- B-spline bases of arbitrary order on uniform and non-uniform grids;

- evaluation of splines and their derivatives and integrals;

- spline interpolations and function approximation;

- basis recombination, for generating bases satisfying homogeneous boundary
  conditions using linear combinations of B-splines.
  Supported boundary conditions include Dirichlet, Neumann, Robin, and
  generalisations of these;

- banded Galerkin and collocation matrices for solving differential equations,
  using B-spline and recombined bases;

- efficient "banded" 3D arrays as an extension of banded matrices.
  These can store 3D tensors associated to quadratic terms in Galerkin methods.

## Example usage

The following is a very brief overview of some of the functionality provided
by this package.

- Interpolate discrete data using cubic splines (B-spline order `k = 4`):

  ```julia
  xdata = (0:10).^2  # points don't need to be uniformly distributed
  ydata = rand(length(xdata))
  itp = interpolate(xdata, ydata, BSplineOrder(4))
  itp(12.3)  # interpolation can be evaluated at any intermediate point
  ```

- Create B-spline basis of order `k = 6` (polynomial degree 5) from a given
  set of breakpoints:

  ```julia
  breaks = log2.(1:16)  # breakpoints don't need to be uniformly distributed either
  B = BSplineBasis(BSplineOrder(6), breaks)
  ```

- Approximate known function by a spline in a previously constructed basis:

  ```julia
  f(x) = exp(-x) * sin(x)
  fapprox = approximate(f, B)
  f(2.3), fapprox(2.3)  # (0.07476354233090601, 0.0747642348243861)
  ```

- Create derived basis satisfying homogeneous [Robin boundary
  conditions](https://en.wikipedia.org/wiki/Robin_boundary_condition) on the
  two boundaries:

  ```julia
  bc = Derivative(0) + 3Derivative(1)
  R = RecombinedBSplineBasis(B, bc)  # satisfies u ∓ 3u' = 0 on the left/right boundary
  ```

- Construct [mass matrix](https://en.wikipedia.org/wiki/Mass_matrix) and
  [stiffness matrix](https://en.wikipedia.org/wiki/Stiffness_matrix) for
  the Galerkin method in the recombined basis:

  ```julia
  # By default, M and L are Hermitian banded matrices
  M = galerkin_matrix(R)
  L = galerkin_matrix(R, (Derivative(1), Derivative(1)))
  ```

- Construct banded 3D tensor associated to non-linear term of the [Burgers
  equation](https://en.wikipedia.org/wiki/Burgers%27_equation):

  ```julia
  T = galerkin_tensor(R, (Derivative(0), Derivative(1), Derivative(0)))
  ```

See the [heat equation
example](https://jipolanco.github.io/BSplineKit.jl/stable/generated/heat/) in
the docs for the use of these tools to solve partial differential equations.

## References

- C. de Boor, *A Practical Guide to Splines*. New York: Springer-Verlag, 1978.

- J. P. Boyd, *Chebyshev and Fourier Spectral Methods*, Second Edition.
  Mineola, N.Y: Dover Publications, 2001.

- O. Botella and K. Shariff, *B-spline Methods in Fluid Dynamics*, Int. J. Comput.
  Fluid Dyn. 17, 133 (2003).
