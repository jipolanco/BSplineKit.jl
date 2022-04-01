using BSplineKit

using Documenter
using Literate

const MAKE_FAST = "--fast" in ARGS  # skip some checks in makedocs

docmeta = quote
    using BSplineKit
    import Random
    Random.seed!(42)
end

# This is to make sure that doctests in docstrings are executed correctly.
DocMeta.setdocmeta!(BSplineKit, :DocTestSetup, docmeta; recursive=true)
DocMeta.setdocmeta!(BSplineKit.BandedTensors, :DocTestSetup,
                    :(using BSplineKit.BandedTensors); recursive=true)

with_checks = !MAKE_FAST

# Generate examples using Literate
# See https://github.com/fredrikekre/Literate.jl/blob/master/docs/make.jl
example_dir = joinpath(@__DIR__, "..", "examples")
output_dir = joinpath(@__DIR__, "src/generated")

for example in ["interpolation.jl", "approximation.jl", "heat.jl", ]
    filename = joinpath(example_dir, example)
    Literate.markdown(filename, output_dir, documenter=true)
end
@time makedocs(
    sitename="BSplineKit.jl",
    format=Documenter.HTML(
        prettyurls=true,
        # load assets in <head>
        assets=["assets/tomate.js"],
    ),
    modules=[BSplineKit],
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "generated/interpolation.md",
            "generated/approximation.md",
            "generated/heat.md",
        ],
        "Library" => [
            "bsplines.md",
            "splines.md",
            "interpolation.md",
            "approximation.md",
            "recombination.md",
            "tensors.md",
            "galerkin.md",
            "collocation.md",
            "boundary_conditions.md",
            "Internals" => ["diffops.md"],
        ],
    ],
    doctest=with_checks,
    linkcheck=with_checks,
    checkdocs=:all,
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/jipolanco/BSplineKit.jl",
    forcepush = true,
    push_preview = true,  # PRs deploy at https://jipolanco.github.io/BSplineKit.jl/previews/PR##
)
