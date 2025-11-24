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
