module BSplineKit

export Derivative

using Reexport

include("BandedTensors/BandedTensors.jl")
@reexport using .BandedTensors

include("DifferentialOps/DifferentialOps.jl")
using .DifferentialOps

include("BSplines/BSplines.jl")
@reexport using .BSplines

include("Recombinations/Recombinations.jl")
@reexport using .Recombinations

include("Splines/Splines.jl")
@reexport using .Splines

include("Collocation/Collocation.jl")
@reexport using .Collocation

include("Galerkin/Galerkin.jl")
@reexport using .Galerkin

include("Interpolations/Interpolations.jl")
@reexport using .Interpolations

end
