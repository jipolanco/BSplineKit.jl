module BSplineKit

using Reexport
using PrecompileTools
using LinearAlgebra: LinearAlgebra # needed for docs

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

@setup_workload begin
    breakpoints = (
        0:0.1:1,
        [-cospi(n / 10) for n = 0:10],
    )
    orders = BSplineOrder.((3, 4, 5, 6))
    xdata = [0.018079550379361597, 0.20891143632845843, 0.3371123927912859,
             0.48426795422372015, 0.4990275860762785, 0.5020191965405584,
             0.6045276714774676, 0.9132439519895715, 0.9378929008606806,
             0.9990743367962296]
    ydata = [0.9659213033691207, 0.5006256871054274, -0.11752572291690448,
             -0.10892570774193748, 0.4288654158945879, -1.4091257090035867,
             -1.3036094032101744, 0.307577760341694, -2.2452255598067254,
             -0.817504902235905]
    @compile_workload begin
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
