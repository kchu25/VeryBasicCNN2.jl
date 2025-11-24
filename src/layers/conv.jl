# ============================================================================
# Learnable Convolutional Filters (Intermediate Layers)
# ============================================================================

"""
    LearnedCodeImgFilters

Learnable convolutional filters for intermediate CNN layers.

# Fields
- `filters`: 4D array (height, width, 1, num_filters)

# Forward Pass
Applies normalized convolution followed by ReLU activation.
"""
struct LearnedCodeImgFilters
    filters::AbstractArray{DEFAULT_FLOAT_TYPE, 4}
    
    function LearnedCodeImgFilters(;
        input_channels::Int,
        filter_height::Int,
        num_filters::Int,
        init_scale::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(1e-3),
        use_cuda::Bool = false,
        rng = Random.GLOBAL_RNG
    )
        filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                      (filter_height, input_channels, 1, num_filters))

        if use_cuda
            filters = cu(filters)
        end
        
        return new(filters)
    end
    
    # Direct constructor for loading models
    LearnedCodeImgFilters(filters) = new(filters)
end

Flux.@layer LearnedCodeImgFilters

"""
    prepare_conv_params(conv_filters::LearnedCodeImgFilters, hp::HyperParameters; use_sparsity=false)

Prepare convolutional filter parameters.

# Returns
- L2-normalized filters, optionally with sparsity weighting
"""
function prepare_conv_params(conv_filters::LearnedCodeImgFilters, hp::HyperParameters; 
                             use_sparsity=false)
    return normalize_filters_l2(conv_filters.filters; 
                               softmax_alpha=hp.softmax_strength_img_fil,
                               use_sparsity=use_sparsity)
end

"""
    (conv_filters::LearnedCodeImgFilters)(code_input, hp::HyperParameters; use_sparsity=false)

Forward pass through convolutional layer.

# Process
1. L2-normalize filters (optionally with sparsity)
2. Convolve with input code
3. Apply ReLU activation

# Returns
- Activated code (4D: height, width, filters, batch)
"""
function (conv_filters::LearnedCodeImgFilters)(code_input, hp::HyperParameters; 
                                                use_sparsity=false)
    normalized_filters = prepare_conv_params(conv_filters, hp; use_sparsity=use_sparsity)
    gradient = conv(code_input, normalized_filters; pad=0, flipped=true)
    code = Flux.NNlib.relu(gradient)
    return code
end
