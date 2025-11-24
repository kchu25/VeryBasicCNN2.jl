# ============================================================================
# Hyperparameter Definitions for CNN Architecture
# ============================================================================

"""
    HyperParameters

CNN hyperparameters for biological sequence analysis.

# Architecture Fields
- `pfm_len::Int`: Length of Position Weight Matrix filters (motif length)
- `num_pfms::Int`: Number of PWM filters in base layer
- `num_img_filters::Vector{Int}`: Number of filters per convolutional layer
- `img_fil_widths::Vector{Int}`: Input channels for each conv layer
- `img_fil_heights::Vector{Int}`: Filter heights for each conv layer

# Pooling Fields  
- `pool_base::Int`: Pooling size for base layer
- `stride_base::Int`: Stride for base layer pooling
- `poolsize::Vector{Int}`: Pooling sizes for each conv layer
- `stride::Vector{Int}`: Strides for each conv layer
- `pool_lvl_top::Int`: Highest layer index that uses pooling

# Training Fields
- `softmax_strength_img_fil::Float32`: Softmax strength for filter normalization
- `batch_size::Int`: Training batch size
- `inference_code_layer::Int`: Layer to extract code from (0 = PWM layer)
- `num_mbconv::Int`: Number of MBConv blocks to add (0 = none, default)
- `mbconv_expansion::Int`: MBConv expansion ratio (default: 4)
"""
Base.@kwdef struct HyperParameters
    # Architecture
    pfm_len::Int = 10
    num_pfms::Int = 24
    num_img_filters::Vector{Int} = [65, 98, 128, 128, 76, 5]
    img_fil_widths::Vector{Int} = vcat([num_pfms], num_img_filters[1:(end-1)])
    img_fil_heights::Vector{Int} = [6, 6, 6, 6, 6, 5]
    
    # Pooling
    pool_base::Int = 2
    stride_base::Int = 1
    poolsize::Vector{Int} = [2, 2, 2, 2, 2, 1]
    stride::Vector{Int} = [1, 1, 2, 2, 2, 1]
    pool_lvl_top::Int = 5
    
    # Training
    softmax_strength_img_fil::DEFAULT_FLOAT_TYPE = 500.0
    batch_size::Int = 256
    inference_code_layer::Int = 0
    
    # MBConv (optional EfficientNet-style blocks)
    num_mbconv::Int = 0
    mbconv_expansion::Int = 4
end

# Convenience accessors
num_layers(hp::HyperParameters) = length(hp.num_img_filters)

function Base.show(io::IO, hp::HyperParameters)
    n_layers = num_layers(hp)
    
    println(io, "HyperParameters:")
    println(io, "  Architecture:")
    println(io, "    Base: $(hp.num_pfms) PWMs × $(hp.pfm_len)nt")
    println(io, "    Conv: $(n_layers) layers")
    println(io, "    Batch: $(hp.batch_size)")
    println(io, "\n  Layers:")
    println(io, "    Layer │ Filters │ Height │ Pool │ Stride")
    println(io, "    ──────┼─────────┼────────┼──────┼────────")
    println(io, "    Base  │  $(lpad(hp.num_pfms, 4)) │   N/A  │   $(hp.pool_base)  │    $(hp.stride_base)")
    
    for i in 1:n_layers
        no_pool = i > hp.pool_lvl_top ? " (skip)" : ""
        println(io, "      $(lpad(i, 2))  │  $(lpad(hp.num_img_filters[i], 4)) │    $(lpad(hp.img_fil_heights[i], 2))  │   $(hp.poolsize[i])  │    $(hp.stride[i])$no_pool")
    end
    
    println(io, "\n  Inference code layer: $(hp.inference_code_layer)")
    hp.num_mbconv > 0 && print(io, "\n  MBConv blocks: $(hp.num_mbconv) (expansion: $(hp.mbconv_expansion)x)")
end

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
    num_img_layers_range = 3:5,
    pfm_length_range = 2:6,
    num_base_filters_range = 48:24:256,
    conv_filter_range = 64:32:256,
    conv_filter_height_range = 4:8,
    pool_size_range = 1:1,
    stride_range = 1:1,
    num_no_pool_layers = 2,
    infer_base_layer_code = false,
    kwargs...
)

amino_acid_ranges_fixed_pool_stride(; kwargs...) = HyperParamRanges(;
    num_img_layers_range = 3:4,
    pfm_length_range = 5:8,
    num_base_filters_range = 64:8:128,
    conv_filter_range = 6:6,
    conv_filter_height_range = 6:8,
    pool_size_range = 1:3,
    stride_range = 1:3,
    batch_size_options = [32, 64, 128],
    num_no_pool_layers = 2,
    infer_base_layer_code = false,
    kwargs...
)

# ============================================================================
# Random Hyperparameter Generation
# ============================================================================

"""
    generate_random_hyperparameters(; batch_size=nothing, rng=Random.GLOBAL_RNG, ranges=DEFAULT_RANGES)

Generate randomized hyperparameters for architecture search.

# Arguments
- `batch_size`: Fixed batch size (defaults to random selection from ranges)
- `rng`: Random number generator for reproducibility
- `ranges`: HyperParamRanges defining valid parameter ranges

# Returns
- `HyperParameters` instance with randomized valid configuration
"""
function generate_random_hyperparameters(; 
    batch_size = nothing, 
    rng = Random.GLOBAL_RNG,
    ranges = DEFAULT_RANGES
)
    n_layers = rand(rng, ranges.num_img_layers_range)
    
    # Sample architecture parameters
    pfm_len = rand(rng, ranges.pfm_length_range)
    n_base = rand(rng, ranges.num_base_filters_range)
    
    # Conv layer filters (last layer fixed for compatibility)
    filters = [rand(rng, ranges.conv_filter_range) for _ in 1:(n_layers-1)]
    push!(filters, ranges.final_layer_filters)
    
    # Input channels track previous layer outputs
    widths = vcat([n_base], filters[1:(end-1)])
    heights = [rand(rng, ranges.conv_filter_height_range) for _ in 1:n_layers]
    
    # Pooling config: respect num_no_pool_layers, always 1 on final layer
    pools = Vector{Int}(undef, n_layers)
    strides = Vector{Int}(undef, n_layers)
    
    for i in 1:n_layers
        if i ≤ ranges.num_no_pool_layers || i == n_layers
            pools[i] = 1
            strides[i] = 1
        else
            pools[i] = rand(rng, ranges.pool_size_range)
            strides[i] = rand(rng, ranges.stride_range)
        end
    end
    
    batch = isnothing(batch_size) ? rand(rng, ranges.batch_size_options) : batch_size
    infer_layer = ranges.infer_base_layer_code ? 0 : ranges.num_no_pool_layers
    
    # MBConv configuration
    n_mbconv = rand(rng, ranges.num_mbconv_range)
    mbconv_exp = rand(rng, ranges.mbconv_expansion_options)

    return HyperParameters(
        pfm_len = pfm_len,
        num_pfms = n_base,
        num_img_filters = filters,
        img_fil_widths = widths,
        img_fil_heights = heights,
        pool_base = ranges.base_pool_size,
        stride_base = ranges.base_stride,
        poolsize = pools,
        stride = strides,
        pool_lvl_top = n_layers - 1,
        softmax_strength_img_fil = ranges.softmax_alpha,
        batch_size = batch,
        inference_code_layer = infer_layer,
        num_mbconv = n_mbconv,
        mbconv_expansion = mbconv_exp
    )
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    receptive_field(hp::HyperParameters)

Calculate receptive field size in the input sequence at the inference code layer.
This accounts for all convolutions, pooling, and strides up to that layer.
"""
function receptive_field(hp::HyperParameters)
    layer = hp.inference_code_layer
    layer == 0 && return hp.pfm_len
    
    # Start with base PWM filter
    rf = hp.pfm_len
    jump = 1  # Effective spacing in input coordinates
    
    # Base layer pooling
    rf += (hp.pool_base - 1) * jump
    jump *= hp.stride_base
    
    # Each conv layer
    for i in 1:layer
        rf += (hp.img_fil_heights[i] - 1) * jump
        
        if i ≤ hp.pool_lvl_top
            rf += (hp.poolsize[i] - 1) * jump
            jump *= hp.stride[i]
        end 
    end
    
    return rf
end

"""
    with_batch_size(hp::HyperParameters, new_batch_size::Int)

Create new HyperParameters with different batch size.
"""
function with_batch_size(hp::HyperParameters, new_batch_size::Int)
    HyperParameters(
        pfm_len = hp.pfm_len,
        num_pfms = hp.num_pfms,
        num_img_filters = hp.num_img_filters,
        img_fil_widths = hp.img_fil_widths,
        img_fil_heights = hp.img_fil_heights,
        pool_base = hp.pool_base,
        stride_base = hp.stride_base,
        poolsize = hp.poolsize,
        stride = hp.stride,
        pool_lvl_top = hp.pool_lvl_top,
        softmax_strength_img_fil = hp.softmax_strength_img_fil,
        batch_size = new_batch_size,
        inference_code_layer = hp.inference_code_layer,
        num_mbconv = hp.num_mbconv,
        mbconv_expansion = hp.mbconv_expansion
    )
end

"""
    with_mbconv(hp::HyperParameters; num_blocks=2, expansion=4)

Create new HyperParameters with MBConv blocks enabled.

# Example
```julia
hp = generate_random_hyperparameters()
hp_mbconv = with_mbconv(hp; num_blocks=3, expansion=6)
```
"""
function with_mbconv(hp::HyperParameters; num_blocks::Int=2, expansion::Int=4)
    HyperParameters(
        pfm_len = hp.pfm_len,
        num_pfms = hp.num_pfms,
        num_img_filters = hp.num_img_filters,
        img_fil_widths = hp.img_fil_widths,
        img_fil_heights = hp.img_fil_heights,
        pool_base = hp.pool_base,
        stride_base = hp.stride_base,
        poolsize = hp.poolsize,
        stride = hp.stride,
        pool_lvl_top = hp.pool_lvl_top,
        softmax_strength_img_fil = hp.softmax_strength_img_fil,
        batch_size = hp.batch_size,
        inference_code_layer = hp.inference_code_layer,
        num_mbconv = num_blocks,
        mbconv_expansion = expansion
    )
end

"""
    efficientnet_mbconv_config(phi::Int=0)

Get EfficientNet-style MBConv configuration based on compound scaling coefficient φ.

# EfficientNet Scaling
- φ=0 (B0): num_blocks=2, expansion=4  (baseline)
- φ=1 (B1): num_blocks=3, expansion=4  (1.2× depth)
- φ=2 (B2): num_blocks=3, expansion=6  (1.4× depth, wider)
- φ=3 (B3): num_blocks=4, expansion=6  (1.8× depth)
- φ=4 (B4): num_blocks=5, expansion=6  (2.2× depth)

# Returns
- `(num_blocks, expansion)`: Configuration tuple

# Example
```julia
hp = generate_random_hyperparameters()
num_blocks, expansion = efficientnet_mbconv_config(2)  # B2 config
hp_b2 = with_mbconv(hp; num_blocks=num_blocks, expansion=expansion)
```
"""
function efficientnet_mbconv_config(phi::Int=0)
    configs = [
        (2, 4),  # B0: baseline
        (3, 4),  # B1: deeper
        (3, 6),  # B2: deeper + wider
        (4, 6),  # B3: much deeper
        (5, 6),  # B4: very deep
        (6, 6),  # B5: extremely deep
        (7, 6),  # B6: ultra deep
        (8, 8),  # B7: maximum depth + width
    ]
    
    @assert 0 ≤ phi ≤ 7 "phi must be between 0 and 7 (B0-B7)"
    return configs[phi + 1]
end

"""
    with_efficientnet_mbconv(hp::HyperParameters, phi::Int=0)

Add EfficientNet-style MBConv blocks with compound scaling coefficient φ.

This uses the standard EfficientNet scaling strategy where φ controls
the depth (number of blocks) and width (expansion ratio) of MBConv layers.

# Arguments
- `hp`: Base hyperparameters
- `phi`: EfficientNet scaling coefficient (0-7 for B0-B7)

# Example
```julia
hp = generate_random_hyperparameters()
hp_b0 = with_efficientnet_mbconv(hp, 0)  # EfficientNet-B0 style
hp_b2 = with_efficientnet_mbconv(hp, 2)  # EfficientNet-B2 style
hp_b4 = with_efficientnet_mbconv(hp, 4)  # EfficientNet-B4 style
```
"""
function with_efficientnet_mbconv(hp::HyperParameters, phi::Int=0)
    num_blocks, expansion = efficientnet_mbconv_config(phi)
    return with_mbconv(hp; num_blocks=num_blocks, expansion=expansion)
end

