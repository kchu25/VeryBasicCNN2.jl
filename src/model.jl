# ============================================================================
# Main SeqCNN Model
# ============================================================================

"""
    SeqCNN

Convolutional neural network for biological sequence analysis.

# Architecture
1. **Base Layer**: Learnable PWMs for motif detection
2. **Conv Layers**: Hierarchical feature extraction with pooling
3. **MBConv Blocks** (optional): EfficientNet-style refinement
4. **Output Layer**: Linear transformation to predictions

# Fields
- `hp::HyperParameters`: Architecture configuration
- `pwms::LearnedPWMs`: Base layer PWM filters
- `conv_layers::Vector{LearnedCodeImgFilters}`: Convolutional layers
- `mbconv_blocks::Vector{MBConvBlock}`: Optional MBConv refinement blocks
- `output_weights::Array{Float32,3}`: Final linear layer (output_dim × embed_dim × 1)

# Constructor
    SeqCNN(hp, input_dims, output_dim; init_scale=0.5, use_cuda=true, rng=Random.GLOBAL_RNG)

# Arguments
- `hp`: HyperParameters specifying architecture
- `input_dims`: Tuple (alphabet_size, sequence_length)
- `output_dim`: Number of output targets
- `init_scale`: Weight initialization scale
- `use_cuda`: Whether to place on GPU
- `rng`: Random number generator

# Example
```julia
hp = generate_random_hyperparameters()
model = SeqCNN(hp, (4, 41), 244)  # DNA sequences, 244 outputs
predictions = model(sequences)
```
"""
struct SeqCNN
    hp::HyperParameters
    pwms::LearnedPWMs
    conv_layers::Vector{LearnedCodeImgFilters}
    mbconv_blocks::Vector{MBConvBlock}
    output_weights::AbstractArray{DEFAULT_FLOAT_TYPE, 3}
    final_nonlinearity::Function
    
    function SeqCNN(
        hp::HyperParameters,
        input_dims::Tuple{T, T},
        output_dim::T;
        init_scale = DEFAULT_FLOAT_TYPE(5e-1), 
        use_cuda = true,
        rng = Random.GLOBAL_RNG
    ) where T <: Integer
        
        alphabet_size, seq_len = input_dims
        
        # Base PWM layer
        pwms = LearnedPWMs(;
            filter_width = hp.pfm_len,
            filter_height = alphabet_size,
            num_filters = hp.num_pfms,
            init_scale = init_scale,
            use_cuda = use_cuda,
            rng = rng
        )

        # Convolutional layers (with LayerNorm for layers > inference_code_layer if enabled)
        conv_layers = [
            LearnedCodeImgFilters(; 
                input_channels = hp.img_fil_widths[i],
                filter_height = hp.img_fil_heights[i],
                num_filters = hp.num_img_filters[i],
                init_scale = init_scale,
                use_layernorm = (hp.use_layernorm && i > hp.inference_code_layer),
                use_cuda = use_cuda,
                rng = rng
            ) for i in 1:num_layers(hp)
        ]

        # Optional MBConv blocks (EfficientNet-style refinement)
        mbconv_blocks = MBConvBlock[]
        if hp.num_mbconv > 0
            @info "Using MBConv refinement: $(hp.num_mbconv) blocks with $(hp.mbconv_expansion)x expansion (EfficientNet-style)"
            final_channels = hp.num_img_filters[end]
            for _ in 1:hp.num_mbconv
                push!(mbconv_blocks, MBConvBlock(;
                    in_channels = final_channels,
                    out_channels = final_channels,
                    kernel_size = 3,
                    expansion_ratio = hp.mbconv_expansion,
                    use_se = true,
                    use_cuda = use_cuda,
                    rng = rng
                ))
            end
        end

        # Output layer
        final_len = final_conv_embedding_length(hp, seq_len)
        embed_dim = final_len * hp.num_img_filters[end]

        output_weights = randn(rng, DEFAULT_FLOAT_TYPE, (output_dim, embed_dim, 1))
        if use_cuda && (output_dim * embed_dim > 1)
            output_weights = cu(output_weights)
        end

        return new(hp, pwms, conv_layers, mbconv_blocks, output_weights, identity)
    end
    
    # Direct constructor for model loading/conversion
    SeqCNN(hp, pwms, conv_layers, mbconv_blocks, output_weights, final_nonlinearity=identity) = 
        new(hp, pwms, conv_layers, mbconv_blocks, output_weights, final_nonlinearity)
end

Flux.@layer SeqCNN

# Specify trainable parameters
Flux.trainable(m::SeqCNN) = (
    pwms = m.pwms, 
    conv_layers = m.conv_layers,
    mbconv_blocks = m.mbconv_blocks,
    output_weights = m.output_weights
)

# ============================================================================
# Model Properties (via getproperty)
# ============================================================================

"""
Custom property access for convenient model introspection and utilities.

# Virtual Properties
- `model.num_conv_layers`: Number of convolutional layers
- `model.receptive_field`: Receptive field at inference layer
- `model.code`: Function to extract code at inference layer
- `model.first_layer_code`: Function to extract base PWM code
- `model.linear_sum`: Function for linear output sum (optimization)
"""
function Base.getproperty(m::SeqCNN, sym::Symbol)
    if sym === :num_conv_layers
        return num_layers(m.hp)
    elseif sym === :receptive_field
        return receptive_field(m.hp)
    elseif sym === :code
        # Extract code at inference layer
        return x -> compute_code_at_layer(m, x, m.hp.inference_code_layer)
    elseif sym === :first_layer_code
        # Extract base PWM code
        return x -> m.pwms(x)
    elseif sym === :linear_sum
        # Linear sum for optimization (no nonlinearity)
        return (x; predict_position) -> sum(
            predict_from_sequences(m, x; 
                predict_position=predict_position, 
                apply_nonlinearity=false)
        )
    elseif sym === :predict_up_to_final_nonlinearity
        return  (x; kwargs...) -> VeryBasicCNN2.predict_from_code(m, x; 
            layer = m.hp.inference_code_layer, 
            apply_nonlinearity=false, kwargs...)
    else
        return getfield(m, sym)
    end
end

function predict_up_to_final_nonlinearity(m, code; predict_position=nothing)
    VeryBasicCNN2.predict_from_code(m, code; 
        layer=m.hp.inference_code_layer,
        apply_nonlinearity=false,
        predict_position=predict_position)
end

# ============================================================================
# Model Construction Helpers
# ============================================================================

"""
    create_model(input_dims, output_dim, batch_size; rng=Random.GLOBAL_RNG, use_cuda=true, ranges=DEFAULT_RANGES)

Create a SeqCNN with randomly generated hyperparameters.

# Arguments
- `input_dims`: Tuple (alphabet_size, sequence_length)
- `output_dim`: Number of output targets
- `batch_size`: Training batch size
- `rng`: Random number generator
- `use_cuda`: Whether to use GPU
- `ranges`: HyperParamRanges for architecture sampling

# Returns
- SeqCNN instance, or `nothing` if architecture is invalid

# Examples
```julia
# DNA sequences
model = create_model((4, 41), 244, 128; ranges=nucleotide_ranges())

# Protein sequences
model = create_model((20, 100), 1, 64; ranges=amino_acid_ranges())
```
"""
function create_model(input_dims, output_dim, batch_size::Int; 
                     rng = Random.GLOBAL_RNG, 
                     use_cuda::Bool = true,
                     ranges = DEFAULT_RANGES)
    hp = generate_random_hyperparameters(; batch_size=batch_size, rng=rng, ranges=ranges)
    
    # Enable LayerNorm and MBConv by default 
    # disable these two if needed for specific experiments
    hp = with_layernorm(hp, true)
    hp = with_mbconv(hp; num_blocks=2, expansion=4)
    
    # Validate architecture
    if final_conv_embedding_length(hp, input_dims[2]) < 1
        @warn "Invalid architecture: final embedding length < 1"
        return nothing
    end
    
    return SeqCNN(hp, input_dims, output_dim; use_cuda=use_cuda, rng=rng)
end

# Convenience constructors for specific domains
create_model_nucleotides(input_dims, output_dim, batch_size; kwargs...) = 
    create_model(input_dims, output_dim, batch_size; ranges=nucleotide_ranges(), kwargs...)

create_model_nucleotides_simple(input_dims, output_dim, batch_size; kwargs...) = 
    create_model(input_dims, output_dim, batch_size; ranges=nucleotide_ranges_simple(), kwargs...)

create_model_nucleotides_fixed_pool_stride(input_dims, output_dim, batch_size; kwargs...) = 
    create_model(input_dims, output_dim, batch_size; 
                ranges=nucleotide_ranges_fixed_pool_stride(), kwargs...)

create_model_aminoacids(input_dims, output_dim, batch_size; kwargs...) = 
    create_model(input_dims, output_dim, batch_size; ranges=amino_acid_ranges(), kwargs...)

create_model_aminoacids_fixed_pool_stride(input_dims, output_dim, batch_size; kwargs...) = 
    create_model(input_dims, output_dim, batch_size; 
                ranges=amino_acid_ranges_fixed_pool_stride(), kwargs...)
