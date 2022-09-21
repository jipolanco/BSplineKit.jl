using ..Collocation: collocation_points

"""
    ApproxByInterpolation <: AbstractApproxMethod

Approximate function by a spline that passes through the given set of points.

The number of points must be equal to the length of the B-spline basis defining
the approximation space.

See belows for different ways of specifying the interpolation points.

---

    ApproxByInterpolation(xs::AbstractVector)
    ApproxByInterpolation((xs, ys, …))

Specifies an approximation by interpolation at the given points `xs`.

The second variant is useful for approximation of multidimensional functions
using tensor-product splines.

---

    ApproxByInterpolation(B::AbstractBSplineBasis)
    ApproxByInterpolation((Bx, By, …))

Specifies an approximation by interpolation at an automatically-determined set
of points, which are expected to be appropriate for the given basis.

The second variant is useful for approximation of multidimensional functions
using tensor-product splines.

The interpolation points are determined by calling [`collocation_points`](@ref).
"""
struct ApproxByInterpolation{
        Points <: Tuple{Vararg{AbstractVector}},
    } <: AbstractApproxMethod
    xs :: Points
end

function Base.show(io::IO, m::ApproxByInterpolation)
    xs = m.xs
    N = length(xs)
    let io = IOContext(io, :compact => true, :limit => true)
        if N == 1
            print(io, "interpolation at ", xs[1])
        else
            print(io, "interpolation at:")
            for (n, x) ∈ enumerate(xs)
                print(io, "\n  ($n) ", x)
            end
        end
    end
end

ApproxByInterpolation(Bs::Tuple{Vararg{AbstractBSplineBasis}}) =
    ApproxByInterpolation(map(collocation_points, Bs))

ApproxByInterpolation(B::AbstractBSplineBasis) = ApproxByInterpolation((B,))

function approximate(
        f::F, Bs::Tuple{Vararg{AbstractBSplineBasis}},
        m::ApproxByInterpolation,
    ) where {F}
    S = SplineInterpolation(undef, Bs, m.xs)
    A = SplineApproximation(m, spline(S), S)
    approximate!(f, A)
end

function _approximate!(f::F, A, m::ApproxByInterpolation) where {F}
    @assert method(A) === m
    S = data(A)
    xs = SplineInterpolations.interpolation_points(S)
    @assert xs === method(A).xs
    ys = coefficients(S)  # just to avoid allocating extra vector
    @inbounds for I ∈ CartesianIndices(ys)  # = (i, j, …)
        x⃗ = getindex.(xs, Tuple(I))  # x⃗[1][i], x⃗[2][j], …
        ys[I] = f(x⃗...)
    end
    interpolate!(S, ys)
    A
end
