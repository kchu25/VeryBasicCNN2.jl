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

Mobile Inverted Bottleneck Convolution (MBConv) block for efficient feature refinement.
Inspired by EfficientNet architecture.

# Architecture
1. Expansion: 1x1 conv to expand channels (expansion_ratio × input_channels)
2. Depthwise: Depthwise conv for spatial feature extraction
3. Squeeze-Excite: Optional channel attention
4. Projection: 1x1 conv to project back to output channels
5. Skip connection if input/output channels match

# Fields
- `expand`: 1x1 expansion convolution
- `dwconv`: Depthwise convolution
- `se`: Squeeze-and-excitation (optional)
- `project`: 1x1 projection convolution
- `use_skip`: Whether to use skip connection
"""
struct MBConvBlock
    expand::Union{Flux.Chain, typeof(identity)}
    dwconv::Flux.Chain
    se::Union{Flux.Chain, typeof(identity)}
    project::Flux.Conv
    use_skip::Bool
    
    function MBConvBlock(;
        in_channels::Int,
        out_channels::Int,
        kernel_size::Int = 3,
        expansion_ratio::Int = 4,
        use_se::Bool = true,
        use_cuda::Bool = false,
        rng = Random.GLOBAL_RNG
    )
        expanded = in_channels * expansion_ratio
        
        # Expansion phase (only if expanding)
        expand = expansion_ratio > 1 ? 
            Flux.Chain(
                Flux.Conv((1, 1), in_channels => expanded; bias=true),
                Flux.BatchNorm(expanded, Flux.swish)
            ) : identity
        
        # Depthwise convolution
        dwconv = Flux.Chain(
            Flux.DepthwiseConv((kernel_size, 1), expanded => expanded; bias=true, pad=(kernel_size÷2, 0)),
            Flux.BatchNorm(expanded, Flux.swish)
        )
        
        # Squeeze-and-Excitation
        se = use_se ? 
            Flux.Chain(
                Flux.AdaptiveMeanPool((1, 1)),
                Flux.Conv((1, 1), expanded => max(1, expanded÷4); bias=true),
                x -> Flux.swish.(x),
                Flux.Conv((1, 1), max(1, expanded÷4) => expanded; bias=true),
                x -> Flux.sigmoid.(x)
            ) : identity
        
        # Projection back to out_channels
        project = Flux.Conv((1, 1), expanded => out_channels; bias=true)
        
        # Skip connection only if dimensions match
        use_skip = (in_channels == out_channels)
        
        block = new(expand, dwconv, se, project, use_skip)
        
        # Move to GPU if requested
        use_cuda && return block |> cu
        return block
    end
end

Flux.@layer MBConvBlock

function (mb::MBConvBlock)(x)
    identity_input = x
    
    # Expansion
    x = mb.expand(x)
    
    # Depthwise conv
    x = mb.dwconv(x)
    
    # Squeeze-and-Excitation
    if mb.se !== identity
        se_weights = mb.se(x)
        x = x .* se_weights
    end
    
    # Projection
    x = mb.project(x)
    
    # Skip connection
    mb.use_skip && return x .+ identity_input
    return x
end
