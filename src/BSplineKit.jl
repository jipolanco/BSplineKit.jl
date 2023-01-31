module BSplineKit

using Reexport
using SnoopPrecompile

include("BandedTensors/BandedTensors.jl")
@reexport using .BandedTensors

include("DifferentialOps/DifferentialOps.jl")
using .DifferentialOps
export Derivative

include("BoundaryConditions/BoundaryConditions.jl")
@reexport using .BoundaryConditions

include("BSplines/BSplines.jl")
@reexport using .BSplines

include("Splines/Splines.jl")
@reexport using .Splines

include("Recombinations/Recombinations.jl")
@reexport using .Recombinations

include("Collocation/Collocation.jl")
@reexport using .Collocation

include("Galerkin/Galerkin.jl")
@reexport using .Galerkin

include("SplineInterpolations/SplineInterpolations.jl")
@reexport using .SplineInterpolations

include("SplineApproximations/SplineApproximations.jl")
@reexport using .SplineApproximations

include("SplineExtrapolations/SplineExtrapolations.jl")
@reexport using .SplineExtrapolations

@precompile_setup begin
    breakpoints = (
        0:0.1:1,
        [-cospi(n / 10) for n = 0:10],
    )
    orders = BSplineOrder.((3, 4, 5, 6))
    xdata = sort!(rand(10))
    ydata = randn(10)
    @precompile_all_calls begin
        for breaks ∈ breakpoints, ord ∈ orders
            B = BSplineBasis(ord, copy(breaks))
            B(0.32)
            S = Spline(B, rand(length(B)))
            S(0.32)
            Derivative() * S
            interpolate(xdata, ydata, ord)
            iseven(order(ord)) && interpolate(xdata, ydata, ord, Natural())
            approximate(sinpi, B, MinimiseL2Error())  # triggers compilation of Galerkin stuff
        end
    end
end

end
