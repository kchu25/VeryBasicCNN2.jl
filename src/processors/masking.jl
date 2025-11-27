# ============================================================================
# Gumbel-Softmax Masking Logic (Shared Across Architectures)
# ============================================================================

"""
    apply_gumbel_mask(cp::CodeProcessor, x; training::Bool=true, step::Union{Nothing, Int}=nothing)

Apply hierarchical Gumbel-Softmax masking (component-wise + channel-wise).

# Arguments
- `cp`: CodeProcessor instance
- `x`: Input features (l, out_channels, 1, n)
- `training`: Whether in training mode (affects Gumbel sampling)
- `step`: Training step for temperature annealing

# Returns
- Masked features with same shape as input
"""
function apply_gumbel_mask(cp::CodeProcessor, x; training::Bool=true, step::Union{Nothing, Int}=nothing)
    # Component-wise mask: Learn mask probabilities from features
    # x shape: (l, out_channels, 1, n)
    mask_logits = Flux.conv(x, cp.mask_proj; pad=0, flipped=true)
    # Output: (l, 1, out_channels, n) -> reshape to (l, out_channels, 1, n)
    mask_logits = reshape(mask_logits, (size(mask_logits, 1), size(mask_logits, 3), 
                                        1, size(mask_logits, 4)))
    
    # Get probabilities: p_c = sigmoid(logits)
    p_c = Flux.sigmoid.(mask_logits)
    
    # Channel-wise mask: Average features across spatial dimension, then project
    # x shape: (l, out_channels, 1, n) -> average to (out_channels, 1, n)
    x_avg = reshape(mean(x; dims=1), (size(x, 2), 1, size(x, 4)))
    # Project: batched_mul(W, x_avg) where W: (out_channels, out_channels, 1)
    channel_logits = Flux.NNlib.batched_mul(cp.channel_mask_proj, x_avg)
    p_ch = Flux.sigmoid.(channel_logits)  # Shape: (out_channels, 1, n)
    
    if training
        # Temperature annealing based on training steps
        # Decay: 0.5 * 0.9995^step → reaches ~0.1 after ~3000 steps
        temp = if isnothing(step)
            cp.mask_temp
        else
            max(DEFAULT_FLOAT_TYPE(0.1), cp.mask_temp * DEFAULT_FLOAT_TYPE(0.9995)^step)
        end
        
        # Component-wise Gumbel-Softmax
        z_c = gumbel_softmax_sample(p_c, temp, cp.mask_eta, cp.mask_gamma, x)
        
        # Channel-wise Gumbel-Softmax
        z_ch = gumbel_softmax_sample(p_ch, temp, cp.mask_eta, cp.mask_gamma, x)
    else
        # Test time: deterministic hard masks
        temp = DEFAULT_FLOAT_TYPE(0.1)
        z_c = hard_threshold_mask(p_c, temp, cp.mask_eta, cp.mask_gamma)
        z_ch = hard_threshold_mask(p_ch, temp, cp.mask_eta, cp.mask_gamma)
    end
    
    # Combine masks: channel mask broadcasts over spatial dimension
    z_ch_broadcast = reshape(z_ch, (1, size(z_ch, 1), 1, size(z_ch, 3)))
    combined_mask = z_c .* z_ch_broadcast
    
    # Apply combined mask
    return x .* combined_mask
end

"""
    gumbel_softmax_sample(p, temp, eta, gamma, reference_array)

Sample from Gumbel-Softmax distribution for soft masking.

# Arguments
- `p`: Probabilities
- `temp`: Temperature
- `eta, gamma`: Stretch parameters for hard threshold
- `reference_array`: Reference for device placement (CPU/GPU)

# Returns
- Soft mask values
"""
function gumbel_softmax_sample(p, temp, eta, gamma, reference_array)
    gumbel = -log.(-log.(rand(DEFAULT_FLOAT_TYPE, size(p)...)))
    if reference_array isa CuArray
        gumbel = cu(gumbel)
    end
    
    logit_p = log.(p .+ DEFAULT_FLOAT_TYPE(1e-8)) .- 
             log.(1 .- p .+ DEFAULT_FLOAT_TYPE(1e-8))
    s = Flux.sigmoid.((logit_p .+ gumbel) ./ temp)
    
    return min.(DEFAULT_FLOAT_TYPE(1), max.(DEFAULT_FLOAT_TYPE(0), 
                s .* (eta - gamma) .+ gamma))
end

"""
    hard_threshold_mask(p, temp, eta, gamma)

Generate hard binary mask from probabilities (test time).

# Arguments
- `p`: Probabilities
- `temp`: Temperature (sharpening)
- `eta, gamma`: Stretch parameters

# Returns
- Hard binary mask (0 or 1)
"""
function hard_threshold_mask(p, temp, eta, gamma)
    logit_p = log.(p .+ DEFAULT_FLOAT_TYPE(1e-8)) .- 
             log.(1 .- p .+ DEFAULT_FLOAT_TYPE(1e-8))
    s = Flux.sigmoid.(logit_p ./ temp)
    
    z_soft = min.(DEFAULT_FLOAT_TYPE(1), max.(DEFAULT_FLOAT_TYPE(0), 
                  s .* (eta - gamma) .+ gamma))
    
    return DEFAULT_FLOAT_TYPE.(z_soft .> DEFAULT_FLOAT_TYPE(0.5))
end
