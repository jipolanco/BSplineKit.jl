# Functions for efficiently evaluating all non-zero B-splines (and/or their
# derivatives) at a given point.

using Base.Cartesian: @ntuple
using Base: @propagate_inbounds
using StaticArrays: MVector

# TODO
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
    @boundscheck checkbounds(ts, i:(i + n))
    @inbounds ti = ts[i]
    @inbounds tj = ts[i + n]
    # @assert ti ≠ tj
    (x - ti) / (tj - ti)
end

# TODO
# - this is redundant with Splines.knot_interval...
"""
    find_knot_interval(ts::AbstractVector, x::Real) -> (i, zone)

Finds the index ``i`` corresponding to the knot interval ``[t_i, t_{i + 1}]``
that should be used to evaluate B-splines at location ``x``.

The knot vector is assumed to be sorted in non-decreasing order.

It also returns a `zone` integer, which is:

- `0`  if `x` is within the knot domain (`ts[begin] ≤ x ≤ ts[end]`),
- `-1` if `x < ts[begin]`,
- `1`  if `x > ts[end]`.

This function is functionally equivalent to de Boor's `INTERV` routine (de Boor
2001, p. 74).
"""
function find_knot_interval(ts::AbstractVector, x::Real)
    if x < first(ts)
        return firstindex(ts), -1
    end
    i = searchsortedlast(ts, x)
    Nt = lastindex(ts)
    if i == Nt
        tlast = ts[Nt]
        while true
            i -= 1
            ts[i] ≠ tlast && break
        end
        zone = (x > tlast) ? 1 : 0
        return i, zone
    else
        return i, 0  # usual case
    end
end

function _evaluate_all_gen(
        ts::AbstractVector, x::Real, ::BSplineOrder{k}, ::Type{T};
        ileft = nothing,
    ) where {k, T}
    if @generated
        @assert k ≥ 1
        ex = quote
            i, zone = _find_knot_interval(ts, x, ileft)
            if zone ≠ 0
                return (i, @ntuple($k, j -> zero($T)))
            end
            # We assume that the value of `i` corresponds to the good interval,
            # even if `ileft` was passed by the user.
            # This is the same as in de Boor's BSPLVB routine.
            # In particular, this allows plotting the extension of a polynomial
            # piece outside of its knot interval (as in de Boor's Fig. 6, p. 114).
            bs_1 = one(T)
        end
        for q ∈ 2:k
            bp = Symbol(:bs_, q - 1)
            bq = Symbol(:bs_, q)
            ex = quote
                $ex
                Δs = @ntuple $(q - 1) j -> @inbounds(_knotdiff(x, ts, i - j + 1, $q - 1))
                $bq = _evaluate_step(Δs, $bp, BSplineOrder($q), $T)
            end
        end
        bk = Symbol(:bs_, k)
        quote
            $ex
            return i, $bk
        end
    else
        _evaluate_all_alt(ts, x, BSplineOrder(k), T; ileft = ileft)
    end
end

# Non-@generated version
function _evaluate_all_alt(
        ts::AbstractVector, x::Real, ::BSplineOrder{k}, ::Type{T};
        ileft = nothing,
    ) where {k, T}
    @assert k ≥ 1
    i, zone = _find_knot_interval(ts, x, ileft)
    if zone ≠ 0
        return (i, ntuple(j -> zero(T), Val(k)))
    end
    bq = zero(MVector{k, T})
    Δs = zero(MVector{k - 1, T})
    bq[1] = one(T)
    @inbounds for q ∈ 2:k
        for j ∈ 1:(q - 1)
            Δs[j] = _knotdiff(x, ts, i - j + 1, q - 1)
        end
        bp = bq[1]
        Δp = Δs[1]
        bq[1] = Δp * bp
        for j = 2:(q - 1)
            bpp, bp = bp, bq[j]
            Δpp, Δp = Δp, Δs[j]
            bq[j] = Δp * bp + (1 - Δpp) * bpp
        end
        bq[q] = (1 - Δp) * bp
    end
    i, Tuple(bq)
end

function _find_knot_interval(ts, x, ileft)
    if isnothing(ileft)
        i, zone = find_knot_interval(ts, x)
    else
        i = ileft
        zone = (x < first(ts)) ? -1 : (x > last(ts)) ? 1 : 0
    end
    i, zone
end

@generated function _evaluate_step(Δs, bp, ::BSplineOrder{k}, ::Type{T}) where {k, T}
    ex = quote
        @inbounds b_1 = Δs[1] * bp[1]
    end
    for j = 2:(k - 1)
        bj = Symbol(:b_, j)
        ex = quote
            $ex
            @inbounds $bj = (1 - Δs[$j - 1]) * bp[$j - 1] + Δs[$j] * bp[$j]
        end
    end
    b_last = Symbol(:b_, k)
    quote
        $ex
        @inbounds $b_last = (1 - Δs[$k - 1]) * bp[$k - 1]
        @ntuple $k b
    end
end

evaluate_all(B, x; kws...) = evaluate_all(B, x, float(typeof(x)); kws...)
