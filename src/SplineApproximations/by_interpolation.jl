using ..Collocation: collocation_points

"""
    ApproxByInterpolation <: AbstractApproxMethod

Approximate function by a spline that passes through the given set of points.

The number of points must be equal to the length of the B-spline basis defining
the approximation space.

See belows for different ways of specifying the interpolation points.

---

    ApproxByInterpolation(xs::AbstractVector)

Specifies an approximation by interpolation at the given points `xs`.

---

    ApproxByInterpolation(B::AbstractBSplineBasis)

Specifies an approximation by interpolation at an automatically-determined set
of points, which are expected to be appropriate for the given basis.

The interpolation points are determined by calling [`collocation_points`](@ref).
"""
struct ApproxByInterpolation{Points <: AbstractVector} <: AbstractApproxMethod
    xs :: Points
end

Base.show(io::IO, m::ApproxByInterpolation) =
    print(io, "interpolation at ", m.xs)

ApproxByInterpolation(B::AbstractBSplineBasis) =
    ApproxByInterpolation(collocation_points(B))

function approximate(f, B::AbstractBSplineBasis, m::ApproxByInterpolation)
    S = SplineInterpolation(undef, B, m.xs)
    A = SplineApproximation(m, spline(S), S)
    approximate!(f, A)
end

function _approximate!(f, A, m::ApproxByInterpolation)
    @assert method(A) === m
    S = data(A)
    xs = SplineInterpolations.interpolation_points(S)
    @assert xs === method(A).xs
    ys = coefficients(S)  # just to avoid allocating extra vector
    @inbounds for i in eachindex(xs, ys)
        ys[i] = f(xs[i])
    end
    interpolate!(S, ys)
    A
end
