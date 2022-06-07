module Splines

export
    Spline,
    Spline1D,
    coefficients,
    integral

export
    SplineWrapper,
    spline

using Base.Cartesian: @nexprs

using ..BSplines
using ..DifferentialOps

import ..BSplines: basis, knots, order

include("spline.jl")
include("wrapper.jl")

end
