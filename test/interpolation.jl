using Random
using StaticArrays: SVector

function test_interpolation(ord::BSplineOrder, ::Type{Ty} = Float64) where {Ty}
    rng = MersenneTwister(42)
    ndata = 40
    xs = sort(randn(rng, ndata))

    # This is Int   if Ty = Int
    #         Int32 if Ty = SVector{N, Int32}
    Tdata = eltype(Ty)

    ys = if Tdata <: Integer
        [rand(rng, Ty) .% Ty(16) for _ = 1:ndata]  # values in -15:15
    else
        randn(rng, Ty, ndata)
    end

    k = order(ord)

    @testset "No BCs" begin
        S = @inferred interpolate(xs, ys, ord)
        @test S.(xs) ≈ ys
    end

    if iseven(k)
        @testset "Natural BCs" begin
            S = @inferred interpolate(xs, ys, ord, Natural())
            @test S.(xs) ≈ ys
            ts = @inferred knots(S)
            Nt = length(ts)
            ts_unique = view(ts, k:(Nt - k + 1))
            @test allunique(ts_unique)

            # Check that unique knots are the same as data locations (this is not
            # the case without natural BCs)
            @test length(ts_unique) == length(xs)
            @test ts_unique == xs

            # Check that some derivatives are zero.
            for n = 2:(k ÷ 2)
                Sder = Derivative(n) * S
                for x ∈ boundaries(basis(S))
                    @test norm(Sder(x)) < norm(S(x)) * 1e-7
                end
            end
        end
    end

    nothing
end

@testset "Interpolation" begin
    @testset "k = $k" for k ∈ (3, 4, 6, 8)
        test_interpolation(BSplineOrder(k), Float64)
    end
    types = (Int32, ComplexF32, SVector{2, Float32})
    @testset "T = $T" for T ∈ types
        test_interpolation(BSplineOrder(4), T)
    end
    @testset "Integer xdata" begin
        xdata = (0:10).^2
        ydata = rand(length(xdata))
        itp = interpolate(xdata, ydata, BSplineOrder(4))
        @test itp.(xdata) ≈ ydata
    end
end
