# Functions for efficiently evaluating all non-zero B-splines (and/or their
# derivatives) at a given point.

using Base.Cartesian: @ntuple
using Base: @propagate_inbounds
using StaticArrays: MVector
using ForwardDiff: ForwardDiff

"""
    evaluate_all(
        B::AbstractBSplineBasis, x::Real,
        [op = Derivative(0)], [T = float(typeof(x))];
        [ileft = nothing],
    ) -> i, bs

Evaluate all B-splines which are non-zero at coordinate `x`.

Returns a tuple `(i, bs)`, where `i` is an index identifying the basis functions
that were computed, and `bs` is a tuple with the actual values.

More precisely:

- `i` is the index of the first B-spline knot ``t_{i}`` when going from ``x``
  towards the left.
  In other words, it is such that ``t_{i} ≤ x < t_{i + 1}``.

  It can be effectively computed as `i = searchsortedlast(knots(B), x)`.
  If the correct value of `i` is already known, one can avoid this computation by
  manually passing this index via the optional `ileft` keyword argument.

- `bs` is a tuple of B-splines evaluated at ``x``:

  ```math
  (b_i(x), b_{i - 1}(x), …, b_{i - k + 1}(x)).
  ```

  It contains ``k`` values, where ``k`` is the order of the B-spline basis.
  Note that values are returned in backwards order starting from the ``i``-th
  B-spline.

## Computing derivatives

One can pass the optional `op` argument to compute B-spline derivatives instead
of the actual B-spline values.

## Examples

See [`AbstractBSplineBasis`](@ref) for some examples using the alternative
evaluation syntax `B(x, [op], [T]; [ileft])`, which calls this function.
"""
function evaluate_all end

@propagate_inbounds function evaluate_all(
        B::BSplineBasis, x::Real, op::Derivative, ::Type{T}; kws...,
    ) where {T <: Number}
    _evaluate_all(knots(B), x, BSplineOrder(order(B)), op, T; kws...)
end

@propagate_inbounds evaluate_all(
    B, x, op::AbstractDifferentialOp = Derivative(0); kws...,
) = evaluate_all(B, x, op, float(typeof(x)); kws...)

@propagate_inbounds evaluate_all(B, x, ::Type{T}; kws...) where {T <: Number} =
    evaluate_all(B, x, Derivative(0), T; kws...)

@propagate_inbounds function _knotdiff(x::Real, ts::AbstractVector, i, n)
    j = i + n
    @boundscheck checkbounds(ts, i)
    @boundscheck checkbounds(ts, j)
    @inbounds ti = ts[i]
    @inbounds tj = ts[j]
    # @assert ti ≠ tj
    (x - ti) / (tj - ti)
end

"""
    find_knot_interval(ts::AbstractVector, x::Real, [ileft = nothing]) -> (i, zone)

Finds the index ``i`` corresponding to the knot interval ``[t_i, t_{i + 1}]``
that should be used to evaluate B-splines at location ``x``.

The knot vector is assumed to be sorted in non-decreasing order.

It also returns a `zone` integer, which is:

- `0`  if `x` is within the knot domain (`ts[begin] ≤ x ≤ ts[end]`),
- `-1` if `x < ts[begin]`,
- `1`  if `x > ts[end]`.

This function is functionally equivalent to de Boor's `INTERV` routine (de Boor
2001, p. 74).

If one already knows the location `i` associated to the knot interval, then one
can pass it as the optional `ileft` argument, in which case only the zone needs
to be computed.
"""
function find_knot_interval end

_isless(a, b) = a < b
_isless(a, b::ForwardDiff.Dual) = a < b.value  # workaround changes in ForwardDiff 1.0 (https://github.com/JuliaDiff/ForwardDiff.jl/pull/481)

function find_knot_interval(ts::AbstractVector, x::Real, ::Nothing = nothing)
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
        zone = _isless(tlast, x) ? 1 : 0
        return i, zone
    else
        return i, 0  # usual case
    end
end

function find_knot_interval(ts::AbstractVector, x::Real, ileft::Integer)
    zone = _knot_zone(ts, x)
    ileft, zone
end

_knot_zone(ts::AbstractVector, x) = (x < first(ts)) ? -1 : (x > last(ts)) ? 1 : 0

function _evaluate_all(
        ts::AbstractVector, x::Real, ::BSplineOrder{k},
        op::Derivative{0}, ::Type{T};
        ileft = nothing,
    ) where {k, T}
    if @generated
        @assert k ≥ 1
        ex = quote
            i, zone = find_knot_interval(ts, x, ileft)
            if zone ≠ 0
                return (i, @ntuple($k, j -> zero($T)))
            end
            # We assume that the value of `i` corresponds to the good interval,
            # even if `ileft` was passed by the user.
            # This is the same as in de Boor's BSPLVB routine.
            # In particular, this allows plotting the extension of a polynomial
            # piece outside of its knot interval (as in de Boor's Fig. 6, p. 114).
            bs_1 = (one(T),)
        end
        for q ∈ 2:k
            bp = Symbol(:bs_, q - 1)
            bq = Symbol(:bs_, q)
            ex = quote
                $ex
                Δs = @ntuple(
                    $(q - 1),
                    j -> @inbounds($T(_knotdiff(x, ts, i - j + 1, $q - 1))),
                )
                $bq = _evaluate_step(Δs, $bp, BSplineOrder($q))
            end
        end
        bk = Symbol(:bs_, k)
        quote
            $ex
            return i, $bk
        end
    else
        _evaluate_all_alt(ts, x, BSplineOrder(k), op, T; ileft = ileft)
    end
end

# Derivatives
function _evaluate_all(
        ts::AbstractVector, x::Real, ::BSplineOrder{k},
        op::Derivative{n}, ::Type{T};
        ileft = nothing,
    ) where {k, n, T}
    if @generated
        n::Int
        @assert n ≥ 1
        # We first need to evaluate the B-splines of order p.
        p = k - n
        if p < 1
            # Derivatives are zero. The returned index is arbitrary...
            return :( firstindex(ts), ntuple(_ -> zero($T), Val($k)) )
        end
        bp = Symbol(:bs_, p)
        ex = quote
            i, $bp = _evaluate_all(ts, x, BSplineOrder($p), Derivative(0), $T; ileft)
        end
        for q ∈ (p + 1):k
            bp = Symbol(:bs_, q - 1)
            bq = Symbol(:bs_, q)
            ex = quote
                $ex
                $bq = _evaluate_step_deriv(ts, i, $bp, BSplineOrder($q), $T)
            end
        end
        bk = Symbol(:bs_, k)
        quote
            $ex
            return i, $bk
        end
    else
        _evaluate_all_alt(ts, x, BSplineOrder(k), op, T; ileft = ileft)
    end
end

# Non-@generated version
@inline function _evaluate_all_alt(
        ts::AbstractVector, x::Real, ::BSplineOrder{k},
        ::Derivative{0}, ::Type{T};
        ileft = nothing,
    ) where {k, T}
    @assert k ≥ 1
    i, zone = find_knot_interval(ts, x, ileft)
    if zone ≠ 0
        return (i, ntuple(j -> zero(T), Val(k)))
    end
    bq = MVector{k, T}(undef)
    Δs = MVector{k - 1, T}(undef)
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

# Derivatives / non-@generated variant
function _evaluate_all_alt(
        ts::AbstractVector, x::Real, ::BSplineOrder{k},
        ::Derivative{n}, ::Type{T};
        ileft = nothing,
    ) where {k, n, T}
    n::Int
    @assert n ≥ 1
    # We first need to evaluate the B-splines of order m.
    m = k - n
    if m < 1
        # Derivatives are zero. The returned index is arbitrary...
        return firstindex(ts), ntuple(_ -> zero(T), Val(k))
    end
    i, bp = _evaluate_all(ts, x, BSplineOrder(m), Derivative(0), T; ileft)
    bq = MVector{k, T}(undef)
    us = MVector{k - 1, T}(undef)
    bq[1:m] .= bp
    for q = (m + 1):k
        p = q - 1
        for δj = 1:p
            @inbounds us[δj] = bq[δj] / (ts[i + q - δj] - ts[i + 1 - δj])
        end
        @inbounds bq[1] = p * us[1]
        for j = 2:(q - 1)
            # Note: adding @inbounds here can slow down stuff (from 25ns to
            # 80ns), which is very strange!! (Julia 1.8-beta2)
            bq[j] = p * (us[j] - us[j - 1])
        end
        @inbounds bq[q] = -p * us[p]
    end
    i, Tuple(bq)
end

@inline @generated function _evaluate_step(Δs, bp, ::BSplineOrder{k}) where {k}
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

@inline @generated function _evaluate_step_deriv(
        ts, i, bp, ::BSplineOrder{k}, ::Type{T},
    ) where {k, T}
    p = k - 1
    ex = quote
        us = @ntuple(
            $p,
            δj -> @inbounds($T(bp[δj] / (ts[i + $k - δj] - ts[i + 1 - δj]))),
        )
        @inbounds b_1 = $p * us[1]
    end
    for j = 2:(k - 1)
        bj = Symbol(:b_, j)
        ex = quote
            $ex
            @inbounds $bj = $p * (-us[$j - 1] + us[$j])
        end
    end
    b_last = Symbol(:b_, k)
    quote
        $ex
        @inbounds $b_last = -$p * us[$p]
        @ntuple $k b
    end
end
