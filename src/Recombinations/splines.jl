# This is similar to LazyArrays.ApplyArray(vcat, ...) but without the extra allocations.
struct LazyVcatVector{T, Data <: Tuple{Vararg{AbstractVector}}} <: AbstractVector{T}
    data   :: Data
    length :: Int
    function LazyVcatVector(data::Tuple)
        T = Base.promote_eltype(data...)
        foreach(Base.require_one_based_indexing, data)
        n = sum(length, data)
        new{T, typeof(data)}(data, n)
    end
end

LazyVcatVector(args...) = LazyVcatVector(args)

Base.size(v::LazyVcatVector) = (v.length,)

Base.@propagate_inbounds Base.getindex(v::LazyVcatVector, i::Integer) = _getindex(v, i, i, v.data...)
Base.@propagate_inbounds _getindex(v, i, j, u, etc...) = j ≤ length(u) ? u[j] : _getindex(v, i, j - length(u), etc...)
@inline _getindex(v, i, j) = throw(BoundsError(v, i))

## ================================================================================ ##

"""
    parent_coefficients(R::RecombinedBSplineBasis, coefs::AbstractVector)

Returns the coefficients associated to the parent B-spline basis, from the
coefficients `coefs` in the recombined basis.

Note that this function doesn't allocate, since it returns a lazy concatenation
of two StaticArrays and a view of the `coefs` vector.
"""
function parent_coefficients(R::RecombinedBSplineBasis, coefs::AbstractVector)
    length(coefs) == length(R) ||
        throw(DimensionMismatch("wrong number of coefficients"))

    M = recombination_matrix(R)

    # Upper-left and lower-right submatrices
    A, C = submatrices(M)

    N = length(coefs)
    Na = size(A, 2)
    Nc = size(C, 2)
    H = N - Nc

    Base.require_one_based_indexing(coefs)

    xA = @inbounds view(coefs, 1:Na)
    xC = @inbounds view(coefs, (H + 1):N)

    yA = A * xA
    yB = @inbounds view(coefs, (Na + 1):H)
    yC = C * xC

    LazyVcatVector(yA, yB, yC)
end

# Returns spline written in the parent basis (usually a regular BSplineBasis).
# Those splines can be evaluated, differentiated, etc... using known algorithms.
function Splines.parent_spline(R::RecombinedBSplineBasis, S::Spline)
    @assert basis(S) === R
    B = parent(R)
    coefs = parent_coefficients(R, coefficients(S))
    Spline(B, coefs)
end
