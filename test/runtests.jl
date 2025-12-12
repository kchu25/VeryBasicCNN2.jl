using VeryBasicCNN2
using Test

@testset "VeryBasicCNN2.jl" begin
    @testset "Final Nonlinearity" begin
        # Test default (identity)
        ranges_default = VeryBasicCNN2.nucleotide_ranges()
        @test ranges_default.final_nonlinearity === identity
        
        # Test tanh variants
        ranges_tanh = VeryBasicCNN2.nucleotide_ranges_tanh()
        @test ranges_tanh.final_nonlinearity === tanh
        
        ranges_simple_tanh = VeryBasicCNN2.nucleotide_ranges_simple_tanh()
        @test ranges_simple_tanh.final_nonlinearity === tanh
        
        ranges_fixed_tanh = VeryBasicCNN2.nucleotide_ranges_fixed_pool_stride_tanh()
        @test ranges_fixed_tanh.final_nonlinearity === tanh
        
        # Test amino acid variants
        ranges_aa_tanh = VeryBasicCNN2.amino_acid_ranges_tanh()
        @test ranges_aa_tanh.final_nonlinearity === tanh
        
        ranges_aa_fixed_tanh = VeryBasicCNN2.amino_acid_ranges_fixed_pool_stride_tanh()
        @test ranges_aa_fixed_tanh.final_nonlinearity === tanh
        
        # Test model creation with identity
        hp_identity = VeryBasicCNN2.generate_random_hyperparameters(
            batch_size=32, 
            ranges=ranges_default
        )
        model_identity = VeryBasicCNN2.SeqCNN(
            hp_identity, (4, 41), 10; 
            final_nonlinearity=ranges_default.final_nonlinearity,
            use_cuda=false
        )
        @test model_identity.final_nonlinearity === identity
        
        # Test model creation with tanh
        hp_tanh = VeryBasicCNN2.generate_random_hyperparameters(
            batch_size=32, 
            ranges=ranges_tanh
        )
        model_tanh = VeryBasicCNN2.SeqCNN(
            hp_tanh, (4, 41), 10; 
            final_nonlinearity=ranges_tanh.final_nonlinearity,
            use_cuda=false
        )
        @test model_tanh.final_nonlinearity === tanh
        
        # Test create_model uses ranges.final_nonlinearity
        model_auto_tanh = VeryBasicCNN2.create_model(
            (4, 41), 10, 32;
            ranges=ranges_tanh,
            use_cuda=false
        )
        @test model_auto_tanh.final_nonlinearity === tanh
    end
end
