using Documenter
using BasisSplines

const MAKE_FAST = "--fast" in ARGS  # skip some checks in makedocs

# This is to make sure that doctests in docstrings are executed correctly.
DocMeta.setdocmeta!(BasisSplines, :DocTestSetup,
                    :(using BasisSplines); recursive=false)
DocMeta.setdocmeta!(BasisSplines.BandedTensors, :DocTestSetup,
                    :(using BasisSplines.BandedTensors); recursive=true)

with_checks = !MAKE_FAST

@time makedocs(
    sitename="BasisSplines.jl",
    format=Documenter.HTML(
        prettyurls=true,
        # load assets in <head>
        # assets=["assets/custom.css",
        #         "assets/matomo.js"],
    ),
    modules=[BasisSplines],
    pages=[
        "Home" => "index.md",
        "bsplines.md",
        "recombination.md",
        "tensors.md",
        "galerkin.md",
        "collocation.md",
        # "tutorial.md",
        # "More examples" => [
        #     "examples/in-place.md",
        #     "examples/gradient.md",
        # ],
        # "Library" => [
        #     "PencilFFTs.md",
        #     "Transforms.md",
        #     "PencilArrays.md",
        #     "PencilIO.md",
        # ],
        # "benchmarks.md",
    ],
    doctest=with_checks,
    linkcheck=with_checks,
    checkdocs=:all,
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
# deploydocs(
#     repo="github.com/jipolanco/BasisSplines.jl",
#     forcepush=true,
# )
