module Splines

export
    Spline,
    Spline1D,
    coefficients,
    integral,
    bases

export
    SplineWrapper,
    spline

using Base.Cartesian: @nexprs

using ..BSplines
using ..DifferentialOps

import ..BSplines: basis, knots, order

include("spline.jl")
include("spline_1d.jl")
include("spline_nd.jl")
include("wrapper.jl")

end
