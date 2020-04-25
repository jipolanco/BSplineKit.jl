"""
    augment_knots(knots::AbstractVector, k::Union{Integer,BSplineOrder})

Modifies the input knots to make sure that the first and last knot have
the multiplicity `k` for splines of order `k`.

Similar to [`augknt`](https://www.mathworks.com/help/curvefit/augknt.html) in
Matlab.
"""
function augment_knots(knots::AbstractVector{T},
                       k::Integer) :: Vector{T} where {T}
    N = length(knots)

    # Determine multiplicity of first and last knots in input.
    m_first = multiplicity(knots, 1)
    m_last = multiplicity(knots, N)

    if m_first == m_last == k
        return knots  # nothing to do
    end

    N_inner = N - m_first - m_last
    Nnew = N_inner + 2k
    t = Vector{float(T)}(undef, Nnew)  # augmented knots

    t_first = knots[1]
    t_last = knots[end]

    t[1:k] .= t_first
    t[(Nnew - k + 1):Nnew] .= t_last
    t[(k + 1):(k + N_inner)] .= @view knots[(m_first + 1):(m_first + N_inner)]

    t
end

augment_knots(knots, ::BSplineOrder{k}) where {k} = augment_knots(knots, k)

"""
    multiplicity(knots, i)

Determine multiplicity of knot `knots[i]`.
"""
function multiplicity(knots::AbstractVector, i)
    @assert Base.require_one_based_indexing(knots)
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
