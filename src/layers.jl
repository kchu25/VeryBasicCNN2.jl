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

Mobile Inverted Bottleneck Convolution (MBConv) block with SE attention.

# Fields
- `expand_filters`: Expansion filters (1, in_channels, 1, expanded)
- `dw_filters`: Depthwise filters (kernel_size, 1, 1, expanded)
- `se_w1`: SE reduction weights (se_channels, expanded, 1)
- `se_w2`: SE expansion weights (expanded, se_channels, 1)
- `project_filters`: Projection filters (1, expanded, 1, out_channels)
- `use_skip`: Whether to use skip connection
"""
struct MBConvBlock
    expand_filters::Union{AbstractArray{DEFAULT_FLOAT_TYPE, 4}, Nothing}
    dw_filters::AbstractArray{DEFAULT_FLOAT_TYPE, 4}
    se_w1::AbstractArray{DEFAULT_FLOAT_TYPE, 3}
    se_w2::AbstractArray{DEFAULT_FLOAT_TYPE, 3}
    project_filters::AbstractArray{DEFAULT_FLOAT_TYPE, 4}
    use_skip::Bool
    
    function MBConvBlock(;
        in_channels::Int,
        out_channels::Int,
        kernel_size::Int = 3,
        expansion_ratio::Int = 4,
        se_ratio::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(4),  # Standard SE reduction ratio
        use_se::Bool = true,
        use_cuda::Bool = false,
        rng = Random.GLOBAL_RNG
    )
        expanded = in_channels * expansion_ratio
        init_scale = DEFAULT_FLOAT_TYPE(1e-3)
        
        # Expansion filters (1, in_channels, 1, expanded)
        expand_filters = expansion_ratio > 1 ?
            init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, (1, in_channels, 1, expanded)) :
            nothing
        
        # Depthwise filters (kernel_size, 1, 1, expanded)
        dw_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, 
                                         (kernel_size, 1, 1, expanded))
        
        # Squeeze-Excitation weights
        se_channels = max(1, floor(Int, expanded / se_ratio))
        se_w1 = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, (se_channels, expanded, 1))
        se_w2 = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, (expanded, se_channels, 1))
        
        # Projection filters (1, expanded, 1, out_channels)
        project_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                              (1, expanded, 1, out_channels))
        
        if use_cuda
            expand_filters = isnothing(expand_filters) ? nothing : cu(expand_filters)
            dw_filters = cu(dw_filters)
            se_w1 = cu(se_w1)
            se_w2 = cu(se_w2)
            project_filters = cu(project_filters)
        end
        
        use_skip = (in_channels == out_channels)
        return new(expand_filters, dw_filters, se_w1, se_w2, project_filters, use_skip)
    end
    
    # Direct constructor for GPU/loading
    MBConvBlock(expand_filters, dw_filters, se_w1, se_w2, project_filters, use_skip) = 
        new(expand_filters, dw_filters, se_w1, se_w2, project_filters, use_skip)
end

Flux.@layer MBConvBlock

function (mb::MBConvBlock)(x)
    identity_input = x
    l, M, _, n = size(x)  # (spatial, channels, 1, batch)
    
    # Expansion
    if !isnothing(mb.expand_filters)
        x = Flux.conv(x, mb.expand_filters; pad=0, flipped=true)
        x = Flux.swish.(x)
    end
    
    edim = size(x, 3)  # expanded channels
    
    # Reshape for depthwise conv
    x = reshape(x, (l, 1, edim, n))
    
    # Depthwise convolution with groups
    pad_h = (size(mb.dw_filters, 1) - 1) ÷ 2
    x = Flux.conv(x, mb.dw_filters; pad=(pad_h, 0), flipped=true, groups=edim)
    x = Flux.swish.(x)
    
    # Squeeze-Excitation
    # Global average pooling over spatial dimension
    x_pooled = reshape(mean(x; dims=1), (edim, 1, n))
    
    # SE: reduce → swish → expand → sigmoid
    attn = Flux.NNlib.batched_mul(mb.se_w1, x_pooled)
    attn = Flux.swish.(attn)
    attn = Flux.NNlib.batched_mul(mb.se_w2, attn)
    attn = Flux.sigmoid.(attn)
    
    # Apply attention and reshape
    attn = reshape(attn, (1, 1, edim, n))
    x = x .* attn
    x = reshape(x, (l, edim, 1, n))
    
    # Projection back
    x = Flux.conv(x, mb.project_filters; pad=0, flipped=true)
    x = reshape(x, (l, size(mb.project_filters, 4), 1, n))
    
    # Skip connection
    mb.use_skip && return x .+ identity_input
    return x
end
