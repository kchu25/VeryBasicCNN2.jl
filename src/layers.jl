# ============================================================================
# Learnable PWM Layer (Base Layer)
# ============================================================================

"""
    LearnedPWMs

Learnable Position Weight Matrices for the first CNN layer.

# Fields
- `filters`: 4D array (alphabet_size, motif_len, 1, num_filters)
- `activation_scaler`: Scalar activation parameter

# Forward Pass
Applies PWM convolution followed by ReLU activation scaled by learned parameter.
"""
struct LearnedPWMs
    filters::AbstractArray{DEFAULT_FLOAT_TYPE, 4}
    activation_scaler::AbstractArray{DEFAULT_FLOAT_TYPE, 1}
    
    function LearnedPWMs(;
        filter_width::Int,
        filter_height::Int = 4,
        num_filters::Int,
        init_scale::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(1e-1),
        use_cuda::Bool = false,
        rng = Random.GLOBAL_RNG
    )
        # Initialize with small random values
        filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, 
                                      (filter_height, filter_width, 1, num_filters))
        scaler = init_scale .* rand(rng, DEFAULT_FLOAT_TYPE, 1)

        if use_cuda
            filters = cu(filters)
        end

        return new(filters, scaler)
    end
    
    # Direct constructor for loading models
    LearnedPWMs(filters, scaler) = new(filters, scaler)
end

Flux.@layer LearnedPWMs

"""
    prepare_pwm_params(pwms::LearnedPWMs; reverse_comp=false)

Prepare PWM parameters for forward pass.

# Returns
- `pwm_matrix`: Position weight matrix (log odds ratios)
- `eta`: Squared and clamped activation scaler
"""
function prepare_pwm_params(pwms::LearnedPWMs; reverse_comp=false)
    pwm_matrix = create_pwm(pwms.filters; reverse_comp=reverse_comp)
    eta = square_clamp(pwms.activation_scaler)
    return pwm_matrix, eta
end

"""
    (pwms::LearnedPWMs)(sequences; reverse_comp=false)

Forward pass through PWM layer.

# Process
1. Create PWM from learned frequencies
2. Convolve with input sequences  
3. Apply ReLU with learned scaling

# Returns
- Code activations (3D: length, filters, batch)
"""
function (pwms::LearnedPWMs)(sequences; reverse_comp=false)
    pwm, eta = prepare_pwm_params(pwms; reverse_comp=reverse_comp)
    gradient = conv(sequences, pwm; pad=0, flipped=true)
    code = Flux.NNlib.relu.(eta[1] .* gradient)
    return code
end

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

# ============================================================================
# MBConv Block (Optional EfficientNet-style refinement)
# ============================================================================

"""
    MBConvBlock

Mobile Inverted Bottleneck Convolution (MBConv) block.
Simple refinement layer that preserves dimensions.

# Fields
- `expand_filters`: Expansion conv filters (1, in_channels, 1, expanded)
- `dw_filters`: Depthwise conv filters (kernel_size, 1, expanded, 1) 
- `project_filters`: Projection conv filters (1, expanded, 1, out_channels)
- `use_skip`: Whether to use skip connection
"""
struct MBConvBlock
    expand_filters::Union{AbstractArray{DEFAULT_FLOAT_TYPE, 4}, Nothing}
    dw_filters::AbstractArray{DEFAULT_FLOAT_TYPE, 4}
    project_filters::AbstractArray{DEFAULT_FLOAT_TYPE, 4}
    use_skip::Bool
    
    function MBConvBlock(;
        in_channels::Int,
        out_channels::Int,
        kernel_size::Int = 3,
        expansion_ratio::Int = 4,
        use_se::Bool = false,  # Disabled for simplicity
        use_cuda::Bool = false,
        rng = Random.GLOBAL_RNG
    )
        expanded = in_channels * expansion_ratio
        init_scale = DEFAULT_FLOAT_TYPE(1e-3)
        
        # Expansion filters (1x1 conv: height=1, width=in_channels)
        expand_filters = expansion_ratio > 1 ?
            init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, (1, in_channels, 1, expanded)) :
            nothing
        
        # Depthwise filters (kernel_size x 1 per channel)
        dw_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, 
                                         (kernel_size, 1, expanded, 1))
        
        # Projection filters (1x1 conv: height=1, width=expanded)
        project_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                              (1, expanded, 1, out_channels))
        
        if use_cuda
            expand_filters = isnothing(expand_filters) ? nothing : cu(expand_filters)
            dw_filters = cu(dw_filters)
            project_filters = cu(project_filters)
        end
        
        use_skip = (in_channels == out_channels)
        return new(expand_filters, dw_filters, project_filters, use_skip)
    end
    
    # Direct constructor for GPU/loading
    MBConvBlock(expand_filters, dw_filters, project_filters, use_skip) = 
        new(expand_filters, dw_filters, project_filters, use_skip)
end

Flux.@layer MBConvBlock

function (mb::MBConvBlock)(x)
    identity_input = x
    
    # Expansion (if applicable)
    if !isnothing(mb.expand_filters)
        x = conv(x, mb.expand_filters; pad=0, flipped=true)
        x = Flux.NNlib.relu(x)
    end
    
    # Depthwise convolution
    pad_h = size(mb.dw_filters, 1) ÷ 2
    x = depthwiseconv(x, mb.dw_filters; pad=(pad_h, 0), flipped=true)
    x = Flux.NNlib.relu(x)
    
    # Projection
    x = conv(x, mb.project_filters; pad=0, flipped=true)
    
    # Skip connection (no activation before skip)
    mb.use_skip && return x .+ identity_input
    return x
end
