import BSplineKit
using BSplineKit.BandedTensors

import BSplineKit.BandedTensors: SubMatrix

using BandedMatrices
using LinearAlgebra: dot
using Random
using Test

function test_banded_tensors()
    A = @inferred BandedTensor3D{Float32}(undef, (20, 20, 16), Val(2))
    rand!(A, -99:99)
    BandedTensors.drop_out_of_bands!(A)

    @testset "Slices" begin
        @inferred A[:, :, 4]
        @inferred A[:, :, 3:5]
        S = A[:, :, 4]
        Acut = A[:, :, 3:5]
        @test S isa SubMatrix
        @test Acut isa BandedTensor3D
        @test Acut[:, :, 2] == S  # SubMatrix comparison
        @test S[S.inds, S.inds] == S.data  # slice over non-zero indices
    end

    @testset "Contraction" begin
        b = rand(size(A, 3))
        Y = @inferred A * b
        @test Y isa BandedMatrix
        @test Y[:, 3] ≈ A[:, 3, :] * b
    end

    @testset "Generalised dot product" begin
        for k in axes(A, 3)
            S = A[:, :, k] :: SubMatrix
            u = rand(size(S, 1))
            v = rand(size(S, 2))

            # Efficient implementation
            x = dot(u, S, v)

            # Inefficient computation using full matrix
            y = dot(u, Array(S), v)

            @test x ≈ y
        end
    end

    nothing
end

@testset "BandedTensors" begin
    test_banded_tensors()
end
