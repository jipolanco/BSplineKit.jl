# Functions for efficiently evaluating all non-zero B-splines (and/or their
# derivatives) at a given point.

using Base.Cartesian: @ntuple
using Base: @propagate_inbounds

# TODO
# - what to do on the right boundary?
#   * what happens when knots are not augmented??
#   * what happens when x > xright? (extrapolation)
# - add a non-generated variant (use `if @generated`)
# - derivatives
# - define (::BSplineBasis)(x) as an alias for `evaluate_all`
# - what about recombined bases?

"""
    evaluate_all(
        B::BSplineBasis, x::Real, [T = float(typeof(x))];
        [ileft = nothing],
    ) -> i, bs

Evaluate all B-splines which are non-zero at coordinate `x`.

Returns a tuple `(i, bs)`, where `i` is an index identifying the basis functions
that were computed, and `bs` is a tuple with the actual values.

More precisely:

- `i` is the index of the first B-spline knot ``t_{i}`` when going from ``x``
  towards the left.
  In other words, it is such that ``t_{i} ≤ x < t{i + 1}``.

  It is effectively computed as `i = searchsortedlast(knots(B), x)`.
  If one already knows the right value of `i`, and wants to avoid this
  computation, one can manually pass `i` via the optional `ileft` keyword
  argument.

- `bs` is a tuple of B-splines evaluated at ``x``:

  ```math
  (b_i(x), b_{i - 1}(x), …, b_{i - k + 1}(x)).
  ```

  It contains ``k`` values, where ``k`` is the order of the B-spline basis.
  Note that values are returned in backwards order starting from the ``i``-th
  B-spline.
"""
@propagate_inbounds function evaluate_all(
        B::BSplineBasis, x::Real, ::Type{T}; kws...,
    ) where {T}
    _evaluate_all_gen(knots(B), x, BSplineOrder(order(B)), T; kws...)
end

@propagate_inbounds function _knotdiff(x, ts, i, n)
    ti = ts[i]
    tj = ts[i + n]
    y = (x - ti) / (tj - ti)
    # TODO
    # - do I need this?
    # - is this actually right?
    # - what happens with knots with multiplicity > 1?
    ifelse(x == ti == tj, one(y), y)  # avoid returning 0/0
end

@generated function _evaluate_all_gen(
        ts::AbstractVector, x::Real, ::BSplineOrder{k}, ::Type{T};
        ileft = nothing,
    ) where {k, T}
    @assert k ≥ 1
    ex = quote
        # TODO is this correct?
        xleft = ts[begin + k - 1]
        xright = ts[end - k + 1]
        i = isnothing(ileft) ? searchsortedlast(ts, x) : ileft
        if x < xleft || x > xright
            return (i, @ntuple($k, j -> zero($T)))
        end
        bs_1 = if x == xright
            i = lastindex(ts) - k
            @inbounds (T(x == ts[i + 1]),)
        else
            @inbounds (T(ts[i] ≤ x < ts[i + 1]),)
        end
        @assert checkbounds(Bool, ts, (i - k + 1):(i + k))
    end
    for q ∈ 2:k
        bp = Symbol(:bs_, q - 1)
        bq = Symbol(:bs_, q)
        ex_q = quote
            Δs = @ntuple $(q - 1) j -> @inbounds(_knotdiff(x, ts, i - j + 1, $q - 1))
            $bq = _evaluate_step(Δs, $bp, BSplineOrder($q), $T)
        end
        push!(ex.args, ex_q)
    end
    bk = Symbol(:bs_, k)
    push!(ex.args, :( return i, $bk ))
    ex
end

@generated function _evaluate_step(Δs, bp, ::BSplineOrder{k}, ::Type{T}) where {k, T}
    bs = :(())
    for j = 1:k
        ex = quote
            bj = zero($T)
        end
        if j ≠ 1
            push!(ex.args, :( @inbounds bj += (1 - Δs[$j - 1]) * bp[$j - 1] ))
        end
        if j ≠ k
            push!(ex.args, :( @inbounds bj += Δs[$j] * bp[$j] ))
        end
        push!(bs.args, ex)
    end
    bs
end

evaluate_all(B, x; kws...) = evaluate_all(B, x, float(typeof(x)); kws...)
