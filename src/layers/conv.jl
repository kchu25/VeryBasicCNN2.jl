# ============================================================================
# Learnable Convolutional Filters (Intermediate Layers)
# ============================================================================

"""   
    layernorm(x, gamma, beta; eps=DEFAULT_FLOAT_TYPE(1e-5))

Apply Layer Normalization.

# Arguments
- `x`: Input tensor (height, width, channels, batch)
- `gamma`: Scale parameters (per channel)
- `beta`: Shift parameters (per channel)
- `eps`: Numerical stability constant

# Returns
- Normalized and affine-transformed tensor
"""
function layernorm(x, gamma, beta; eps=DEFAULT_FLOAT_TYPE(1e-5))
    # Normalize over spatial and channel dimensions (everything except batch)
    dims = (1, 4) # spatial and batch
    x_mean = mean(x; dims=dims)
    x_var = var(x; dims=dims, corrected=false)
    x_norm = (x .- x_mean) ./ sqrt.(x_var .+ eps)
    
    # Scale and shift with learnable parameters (broadcast over spatial dims)
    gamma_r = reshape(gamma, (1, :, 1, 1))
    beta_r = reshape(beta, (1, :, 1, 1))
    return gamma_r .* x_norm .+ beta_r
end

"""   
    LearnedCodeImgFilters

Learnable convolutional filters for intermediate CNN layers.

# Fields
- `filters`: 4D array (height, width, 1, num_filters)
- `ln_gamma`: LayerNorm scale parameter (optional, per channel)
- `ln_beta`: LayerNorm shift parameter (optional, per channel)

# Forward Pass
Applies normalized convolution followed by ReLU activation, optionally with LayerNorm.
"""
struct LearnedCodeImgFilters
    filters::AbstractArray{DEFAULT_FLOAT_TYPE, 4}
    ln_gamma::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 1}}
    ln_beta::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 1}}
    
    function LearnedCodeImgFilters(;
        input_channels::Int,
        filter_height::Int,
        num_filters::Int,
        init_scale::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(0.01),
        use_layernorm::Bool = false,
        use_cuda::Bool = false,
        rng = Random.GLOBAL_RNG
    )
        # keep thhis for now to ensure training stability; full debug later
        init_scale = DEFAULT_FLOAT_TYPE(0.01); 
        
        filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                      (filter_height, input_channels, 1, num_filters))

        # Initialize LayerNorm params if needed
        if use_layernorm
            ln_gamma = init_scale .* ones(DEFAULT_FLOAT_TYPE, num_filters)
            ln_beta = zeros(DEFAULT_FLOAT_TYPE, num_filters)
        else
            ln_gamma = nothing
            ln_beta = nothing
        end

        if use_cuda
            filters = cu(filters)
            ln_gamma = isnothing(ln_gamma) ? nothing : cu(ln_gamma)
            ln_beta = isnothing(ln_beta) ? nothing : cu(ln_beta)
        end
        
        return new(filters, ln_gamma, ln_beta)
    end
    
    # Direct constructor for loading models (backward compatibility)
    LearnedCodeImgFilters(filters) = new(filters, nothing, nothing)
    LearnedCodeImgFilters(filters, ln_gamma, ln_beta) = new(filters, ln_gamma, ln_beta)
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
