module Splines

export
    Spline,
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
include("periodic.jl")
include("wrapper.jl")

end
