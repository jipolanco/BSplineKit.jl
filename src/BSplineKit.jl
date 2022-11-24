module BSplineKit

export Derivative

using Reexport

include("BandedTensors/BandedTensors.jl")
@reexport using .BandedTensors

include("DifferentialOps/DifferentialOps.jl")
using .DifferentialOps

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

end
