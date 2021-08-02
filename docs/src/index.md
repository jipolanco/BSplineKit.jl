# BSplineKit.jl

Tools for B-spline based Galerkin and collocation methods in Julia.

## Features

This package provides:

- [B-spline bases](@ref BSplines-api) of arbitrary order on uniform and
  non-uniform grids;

- evaluation of [splines](@ref Splines-api) and their derivatives and
  integrals;

- [spline interpolations](@ref interpolation-api) and
  [function approximation](@ref function-approximation-api);

- [basis recombination](@ref basis-recombination-api), for generating bases
  satisfying homogeneous boundary conditions using linear combinations of
  B-splines.
  Supported boundary conditions include Dirichlet, Neumann, Robin, and
  generalisations of these;

- banded [Galerkin](@ref galerkin-api) and [collocation](@ref collocation-api)
  matrices for solving differential equations, using B-spline and recombined
  bases;

- efficient ["banded" 3D arrays](@ref banded-tensors-api) as an extension of
  banded matrices.
  These can store 3D tensors associated to quadratic terms in Galerkin methods.

## Example usage

The following is a very brief overview of some of the functionality provided
by this package.
See the examples in the sidebar for more details.

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
  R = RecombinedBSplineBasis(bc, B)  # satisfies u ∓ 3u' = 0 on the left/right boundary
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

## Comparison with similar packages

This project presents several similarities with the excellent
[BSplines](https://github.com/sostock/BSplines.jl) package.
This includes various types and functions which have the same names (e.g.
`BSplineBasis`, `Spline`, `knots`, `order`).
In most cases this is pure coincidence, even though some inspiration was later
taken from that package (for instance, the idea of a `Derivative` type).

Some design differences with the BSplines package include:

- in BSplineKit, the B-spline order `k` is considered a compile-time
  constant, as it is encoded in the `BSplineBasis` type.
  This leads to important performance gains when evaluating splines.
  It also enables the construction of efficient 3D banded structures based on
  stack-allocated
  [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl);

- we do not assume that knots are repeated `k` times at the boundaries, even
  though this is still the default when creating a B-spline basis.
  This is to allow the possibility of imposing periodic boundary conditions
  on the basis.

In addition to the common functionality,
BSplineKit provides easy to use tools for solving
[boundary-value problems](https://en.wikipedia.org/wiki/Boundary_value_problem)
using B-splines.
This includes the generation of bases satisfying a chosen set of boundary
conditions (*basis recombination*), as well as the construction of
arrays for solving such problems using collocation and Galerkin methods.

## References

- C. de Boor, *A Practical Guide to Splines*. New York: Springer-Verlag, 1978.

- J. P. Boyd, *Chebyshev and Fourier Spectral Methods*, Second Edition.
  Mineola, N.Y: Dover Publications, 2001.

- O. Botella and K. Shariff, *B-spline Methods in Fluid Dynamics*, Int. J. Comput.
  Fluid Dyn. 17, 133 (2003).
