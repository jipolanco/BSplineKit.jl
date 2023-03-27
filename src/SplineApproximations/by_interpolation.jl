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
struct ApproxByInterpolation{Points <: AbstractVector, Coefs <: AbstractVector} <: AbstractApproxMethod
    xs :: Points
    ys :: Coefs  # only used in specific cases (cubic periodic splines for now)

    function ApproxByInterpolation(xs::AbstractVector)
        ys = empty(xs)
        new{typeof(xs), typeof(ys)}(xs, ys)
    end
end

function Base.show(io::IO, m::ApproxByInterpolation)
    let io = IOContext(io, :compact => true, :limit => true)
        print(io, "interpolation at ", m.xs)
    end
end

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
    if S.C_copy === nothing
        ys = coefficients(S)  # just to avoid allocating extra vector
    else
        # In this case, assume that ys and coefficients(S) cannot be aliased.
        # This is the case for cubic periodic splines.
        ys = m.ys
        resize!(ys, length(xs))
    end
    @inbounds for i in eachindex(xs, ys)
        ys[i] = f(xs[i])
    end
    interpolate!(S, ys)
    A
end
