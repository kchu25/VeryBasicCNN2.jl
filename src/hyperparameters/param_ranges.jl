# ============================================================================
# Hyperparameter Range Specifications
# ============================================================================

"""
    HyperParamRanges

Specification of valid ranges for random hyperparameter generation.
"""
Base.@kwdef struct HyperParamRanges
    num_img_layers_range = 3:5
    pfm_length_range = 3:9
    num_base_filters_range = 72:12:512
    conv_filter_range = 128:32:512
    conv_filter_height_range = 1:5
    pool_size_range = 1:2
    stride_range = 1:2
    num_no_pool_layers::Int = 0
    batch_size_options = [64, 128, 256]
    final_layer_filters::Int = 48
    base_pool_size::Int = 1
    base_stride::Int = 1
    softmax_alpha = SOFTMAX_ALPHA
    infer_base_layer_code::Bool = true
    
    # MBConv options (default: phi=0, i.e., 2 blocks with expansion=4)
    num_mbconv_range = 2:2
    mbconv_expansion_options = [4]
end

const DEFAULT_RANGES = HyperParamRanges()

# ============================================================================
# Domain-Specific Range Presets
# ============================================================================

"""
    nucleotide_ranges(; kwargs...)

Hyperparameter ranges optimized for nucleotide sequences (DNA/RNA).
4-letter alphabet, typical motif lengths 6-12nt.
"""
nucleotide_ranges(; kwargs...) = HyperParamRanges(;
    num_img_layers_range = 3:5,
    pfm_length_range = 6:12,
    num_base_filters_range = 48:24:256,
    conv_filter_range = 64:32:256,
    conv_filter_height_range = 2:4,
    infer_base_layer_code = true,
    kwargs...
)

"""
    amino_acid_ranges(; kwargs...)

Hyperparameter ranges optimized for amino acid sequences (proteins).
20-letter alphabet, larger filters needed.
"""
amino_acid_ranges(; kwargs...) = HyperParamRanges(;
    num_img_layers_range = 3:5,
    pfm_length_range = 5:10,
    num_base_filters_range = 64:32:320,
    conv_filter_range = 96:48:384,
    conv_filter_height_range = 6:12,
    batch_size_options = [32, 64, 128],
    infer_base_layer_code = false,
    kwargs...
)

# Simplified ranges for testing
nucleotide_ranges_simple(; kwargs...) = HyperParamRanges(;
    num_img_layers_range = 2:3,
    pfm_length_range = 7:9,
    num_base_filters_range = 32:64,
    conv_filter_range = 32:64,
    conv_filter_height_range = 2:4,
    kwargs...
)

# Fixed pooling/stride for controlled experiments
nucleotide_ranges_fixed_pool_stride(; kwargs...) = HyperParamRanges(;
    num_img_layers_range = 4:5,
    pfm_length_range = 3:2:7,
    num_base_filters_range = 16:8:48,
    conv_filter_range = 64:32:256,
    conv_filter_height_range = 4:8,
    pool_size_range = 2:3, # overrides by num_no_pool_layers in generation
    stride_range = 2:3, # overrides by num_no_pool_layers in generation
    num_no_pool_layers = 2,
    infer_base_layer_code = false,
    kwargs...
)

amino_acid_ranges_fixed_pool_stride(; kwargs...) = HyperParamRanges(;
    num_img_layers_range = 4:5,
    pfm_length_range = 3:2:7,
    num_base_filters_range = 16:8:64,
    conv_filter_range = 16:8:48,
    conv_filter_height_range = 3:7,
    pool_size_range = 2:3, # overrides by num_no_pool_layers in generation
    stride_range = 2:3, # overrides by num_no_pool_layers in generation
    batch_size_options = [32, 64, 128],
    num_no_pool_layers = 2,
    infer_base_layer_code = false,
    kwargs...
)
