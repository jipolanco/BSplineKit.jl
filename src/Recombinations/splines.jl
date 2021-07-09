using LazyArrays: ApplyArray

"""
    parent_coefficients(R::RecombinedBSplineBasis, coefs::AbstractVector)

Returns the coefficients associated to the parent B-spline basis, from the
coefficients `coefs` in the recombined basis.

Note that this function doesn't allocate, since it returns a lazy concatenation
(via LazyArrays.jl) of two StaticArrays and a view of the `coefs` vector.
"""
function parent_coefficients(R::RecombinedBSplineBasis, coefs::AbstractVector)
    length(coefs) == length(R) ||
        throw(DimensionMismatch("wrong number of coefficients"))

    M = recombination_matrix(R)

    # Upper-left and lower-right submatrices
    A = M.ul
    C = M.lr

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

    ApplyArray(vcat, yA, yB, yC)
end

# Returns spline written in the parent basis (usually a regular BSplineBasis).
# Those splines can be evaluated, differentiated, etc... using known algorithms.
function Splines.parent_spline(R::RecombinedBSplineBasis, S::Spline)
    @assert basis(S) === R
    B = parent(R)
    coefs = parent_coefficients(R, coefficients(S))
    Spline(B, coefs)
end
