using Documenter
using BSplineKit

const MAKE_FAST = "--fast" in ARGS  # skip some checks in makedocs

# This is to make sure that doctests in docstrings are executed correctly.
DocMeta.setdocmeta!(BSplineKit, :DocTestSetup,
                    :(using BSplineKit); recursive=false)
DocMeta.setdocmeta!(BSplineKit.BandedTensors, :DocTestSetup,
                    :(using BSplineKit.BandedTensors); recursive=true)

with_checks = !MAKE_FAST

@time makedocs(
    sitename="BSplineKit.jl",
    format=Documenter.HTML(
        prettyurls=true,
        # load assets in <head>
        assets=["assets/matomo.js"],
    ),
    modules=[BSplineKit],
    pages=[
        "Home" => "index.md",
        "Library" => [
            "bsplines.md",
            "splines.md",
            "recombination.md",
            "tensors.md",
            "galerkin.md",
            "collocation.md",
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
    repo="github.com/jipolanco/BSplineKit.jl",
    forcepush=true,
)
