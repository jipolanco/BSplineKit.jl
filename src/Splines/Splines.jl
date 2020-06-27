module Splines

export Spline, coefficients, integral

using StaticArrays: MVector

using ..BSplines
using ..DifferentialOps

import ..BSplines: basis, knots, order

include("spline.jl")

end
