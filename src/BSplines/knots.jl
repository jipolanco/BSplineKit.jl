"""
    augment_knots(knots::AbstractVector, k::Union{Integer,BSplineOrder})

Modifies the input knots to make sure that the first and last knot have
multiplicity `k` for splines of order `k`.

It is assumed that border knots have multiplicity 1 at the borders.
That is, border coordinates should *not* be repeated in the input.
"""
function augment_knots(knots::AbstractVector{T}, ::BSplineOrder{k}) where {T,k}
    # The idea is to "sandwich" the input knots with static vectors, returning
    # a lazy array.
    t_left = @SVector fill(first(knots), k - 1)
    t_right = @SVector fill(last(knots), k - 1)
    Vcat(t_left, knots, t_right)
end

@inline augment_knots(knots, k::Integer) = augment_knots(knots, BSplineOrder(k))

"""
    multiplicity(knots, i)

Determine multiplicity of knot `knots[i]`.
"""
function multiplicity(knots::AbstractVector, i)
    Base.require_one_based_indexing(knots)
    v = knots[i]
    m = 1

    # Check in both directions
    j = i - 1
    while j > 0 && knots[j] == v
        j -= 1
        m += 1
    end

    j = i + 1
    N = length(knots)
    while j <= N && knots[j] == v
        j += 1
        m += 1
    end

    m
end
