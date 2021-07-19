using BSplineKit
using Test

@testset "Knots" begin
    breaks = -1:0.1:1
    for ξs ∈ (breaks, collect(breaks))
        k = 4
        ts = BSplines.AugmentedKnots{k}(ξs)
        ys = collect(ts)
        for x ∈ -1.05:0.5:1.05
            @test searchsortedlast(ts, x) == searchsortedlast(ys, x)
        end
    end
end
