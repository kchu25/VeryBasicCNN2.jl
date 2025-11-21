module VeryBasicCNN2

# Write your package code here.

using Flux
using CUDA
using Random
using ChainRulesCore: @ignore_derivatives

# ============================================================================
# Constants
# ============================================================================

const DEFAULT_FLOAT_TYPE = Float32
const SOFTMAX_ALPHA = DEFAULT_FLOAT_TYPE(500)

# ============================================================================
# Core Components (Order Matters)
# ============================================================================

include("hyperparameters.jl")
include("utils.jl")
include("layers.jl")
include("model.jl")
include("forward.jl")
include("loss.jl")
include("convert.jl")

# ============================================================================
# Exports
# ============================================================================

# Types
export SeqCNN
export HyperParameters, HyperParamRanges
export LearnedPWMs, LearnedCodeImgFilters, MBConvBlock

# Hyperparameter functions
export generate_random_hyperparameters
export nucleotide_ranges, amino_acid_ranges
export nucleotide_ranges_simple, nucleotide_ranges_fixed_pool_stride
export amino_acid_ranges_fixed_pool_stride
export receptive_field, with_batch_size, with_mbconv, num_layers
export efficientnet_mbconv_config, with_efficientnet_mbconv

# Model construction
export create_model
export create_model_nucleotides, create_model_nucleotides_simple
export create_model_nucleotides_fixed_pool_stride
export create_model_aminoacids, create_model_aminoacids_fixed_pool_stride

# Forward pass functions
export predict_from_sequences, predict_from_code
export compute_code_at_layer, extract_features

# Loss functions
export compute_training_loss, huber_loss, masked_mse

# Utilities
export model2cpu, model2gpu
export final_conv_embedding_length

end
