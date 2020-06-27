# BSplineKit.jl

Tools for B-spline based Galerkin and collocation methods in Julia.

⚠ This package is currently in an **experimental** stage and under active
development.

## Features

At this moment, this package provides:

- B-spline bases of arbitrary order on uniform and non-uniform grids;

- evaluation of splines and their derivatives, as well as data interpolation;

- basis recombination, for generating bases satisfying homogeneous boundary
  conditions, using linear combinations of B-splines.
  A wide variety of boundary conditions is supported, including Dirichlet,
  Neumann, Robin, and more complex variants;

- [banded](https://github.com/JuliaMatrices/BandedMatrices.jl) Galerkin and
  collocation matrices for solving differential equations, using B-spline and
  recombined bases.
  Integrals for Galerkin matrices are computed exactly (up to machine
  precision) using [Gauss—Legendre
  quadratures](https://github.com/JuliaApproximation/FastGaussQuadrature.jl);

- efficient "banded" 3D arrays as an extension of banded matrices.
  These can store 3D tensors associated to quadratic terms in Galerkin methods.

## Example usage

The following is a very brief overview of some of the functionality provided
by this package.

Create B-spline basis of order ``k = 4`` (polynomial degree 3) from a given
set of breakpoints:

```julia
breaks = -1:0.1:1
B = BSplineBasis(4, breaks)
```

Create derived basis satisfying the homogeneous [Robin boundary
conditions](https://en.wikipedia.org/wiki/Robin_boundary_condition)
``u + 3 \frac{∂u}{∂n} = 0`` on the two boundaries:

```julia
bc = Derivative(0) + 3Derivative(1)
R = RecombinedBSplineBasis(bc, B)  # satisfies u ∓ 3u' = 0 on the left/right boundary
```

Construct [mass matrix](https://en.wikipedia.org/wiki/Mass_matrix) and
[stiffness matrix](https://en.wikipedia.org/wiki/Stiffness_matrix) for
the Galerkin method from the functions in the recombined basis:

```julia
# By default, M and L are Hermitian banded matrices
M = galerkin_matrix(R)
L = galerkin_matrix(R, (Derivative(1), Derivative(1)))
```

Construct banded 3D tensor associated to non-linear term of the [Burgers
equation](https://en.wikipedia.org/wiki/Burgers%27_equation):

```julia
T = galerkin_tensor(R, (Derivative(0), Derivative(1), Derivative(0)))
```

A tutorial showcasing this and much more functionality is coming in the
future.

## Similar projects

This project presents several similarities with the great
[BSplines](https://github.com/sostock/BSplines.jl) package.
This includes various types and functions which have the same names (e.g.
`BSplineBasis`, `Spline`, `knots`, `order`).
In most cases this is pure coincidence, as I wasn't aware of BSplines when
development started.
Some inspiration was later taken from that package (including, for instance,
the idea of a `Derivative` type).

Some design differences with the BSplines package include:

- in BSplineKit, the B-spline order `k` is considered a compile-time
  constant, as it is encoded in the BSplineBasis type. This may lead to some
  performance improvements. More importantly, it enables the construction of
  efficient 3D banded structures based on stack-allocated
  [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl);

- we do not assume that knots are repeated `k` times at the boundaries, even
  though this is still the default when creating a B-spline basis.
  This is to allow the possibility of imposing periodic boundary conditions
  on the basis.

Overall, the BSplines package is probably the better option if one wants to
approximate functions and interpolate data using B-splines, as BSplineKit has
not been yet optimised for these purposes.

BSplineKit provides easy to use functionality for solving
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
