# ============================================================================
# Code Processor Networks
# ============================================================================
# Networks that process concatenated code and gradient features
# Input: code ⊕ gradient (concatenated along channel dimension)
# Output: Same dimension as code at inference_code_layer

"""
    CodeProcessorType

Type of architecture for code processing.
- `:plain`: Simple depthwise convolution
- `:resnet`: Depthwise conv with residual connection
- `:mbconv`: MBConv-style with expansion and SE attention
- `:soft_threshold`: Learnable soft thresholding with gating
- `:deep_plain`: Stacked plain layers (2 layers for more capacity)
"""
@enum CodeProcessorType plain resnet mbconv soft_threshold deep_plain

"""
    CodeProcessor

Network for processing concatenated code and gradient features.

# Fields
- `expand_filters`: Channel expansion (optional, for mbconv)
- `dw_filters`: Depthwise convolution filters
- `se_w1, se_w2`: Squeeze-excitation weights (optional, for mbconv)
- `project_filters`: Channel projection back to output size
- `gate_filters`: Gating filters (optional, for soft_threshold)
- `threshold_param`: Learnable threshold parameter (optional, for soft_threshold)
- `use_residual`: Whether to use skip connection
- `arch_type`: Architecture type (:plain, :resnet, :mbconv, :soft_threshold)
"""
struct CodeProcessor
    expand_filters::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 4}}
    dw_filters::AbstractArray{DEFAULT_FLOAT_TYPE, 4}
    se_w1::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 3}}
    se_w2::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 3}}
    project_filters::AbstractArray{DEFAULT_FLOAT_TYPE, 4}
    gate_filters::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 4}}
    threshold_param::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 1}}
    dw_filters_2::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 4}}  # Second layer for deep_plain
    project_filters_2::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 4}}  # Second projection for deep_plain
    dw_filters_3::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 4}}  # Third layer for deep_plain
    project_filters_3::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 4}}  # Third projection for deep_plain
    mask_proj::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 4}}  # Projection to get mask logits from features
    mask_temp::DEFAULT_FLOAT_TYPE  # Gumbel-Softmax temperature
    mask_eta::DEFAULT_FLOAT_TYPE   # Right stretch for hard threshold
    mask_gamma::DEFAULT_FLOAT_TYPE # Left stretch for hard threshold
    use_hard_mask::Bool  # Whether to use hard mask
    use_residual::Bool
    arch_type::CodeProcessorType
    
    function CodeProcessor(;
        in_channels::Int,        # code_channels + gradient_channels (concatenated)
        out_channels::Int,       # Same as code_channels
        kernel_size::Int = 3,
        expansion_ratio::Int = 2,
        se_ratio::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(8),
        use_se::Bool = true,     # Toggle SE attention for mbconv
        use_hard_mask::Bool = false,  # Toggle Gumbel-Softmax hard mask
        mask_temp::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(0.5),
        mask_eta::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(1.0),
        mask_gamma::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(0.0),
        arch_type::CodeProcessorType = mbconv,
        use_cuda::Bool = false,
        rng = Random.GLOBAL_RNG
    )
        init_scale = DEFAULT_FLOAT_TYPE(0.01)
        
        # Initialize mask projection if requested
        if use_hard_mask
            # 1×1 conv: out_channels -> out_channels
            # Output goes through sigmoid to get p_c ∈ [0,1]
            mask_proj = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                           (1, out_channels, 1, out_channels))
        else
            mask_proj = nothing
        end
        
        # Soft threshold architecture
        if arch_type == soft_threshold
            # Gate filters to learn importance of gradient features
            gate_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                              (1, out_channels, 1, out_channels))
            # Learnable threshold - initialize to small positive value
            threshold_param = fill(DEFAULT_FLOAT_TYPE(0.1), 1)
            
            # Simple projection from concatenated to output
            project_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                                 (1, in_channels, 1, out_channels))
            
            # No expansion, depthwise, or SE for this architecture
            expand_filters = nothing
            dw_filters = zeros(DEFAULT_FLOAT_TYPE, (1, 1, 1, 1))  # Dummy
            se_w1 = nothing
            se_w2 = nothing
            use_residual = false
            
        else
            # For mbconv, resnet, plain, and deep_plain architectures
            gate_filters = nothing
            threshold_param = nothing
            
            # Deep plain: 3 stacked layers
            if arch_type == deep_plain
                # First layer: 2C → C
                dw_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                                 (kernel_size, 1, 1, in_channels))
                project_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                                     (1, in_channels, 1, out_channels))
                # Second layer: C → C  
                dw_filters_2 = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                                   (kernel_size, 1, 1, out_channels))
                project_filters_2 = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                                       (1, out_channels, 1, out_channels))
                # Third layer: C → C
                dw_filters_3 = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                                   (kernel_size, 1, 1, out_channels))
                project_filters_3 = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                                       (1, out_channels, 1, out_channels))
                expand_filters = nothing
                se_w1 = nothing
                se_w2 = nothing
                use_residual = true  # Use residual for deep network
            else
                # For mbconv, resnet, plain
                # Expansion (only for mbconv)
                if arch_type == mbconv
                    expanded = in_channels * expansion_ratio
                    expand_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, 
                                                         (1, in_channels, 1, expanded))
                else
                    expanded = in_channels
                    expand_filters = nothing
                end
                
                # Depthwise filters
                dw_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                                 (kernel_size, 1, 1, expanded))
                
                # Squeeze-Excitation (only for mbconv if use_se=true)
                if arch_type == mbconv && use_se
                    se_channels = max(1, floor(Int, expanded / se_ratio))
                    se_w1 = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, (se_channels, expanded, 1))
                    se_w2 = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, (expanded, se_channels, 1))
                else
                    se_w1 = nothing
                    se_w2 = nothing
                end
                
                # Projection to output channels
                project_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                                     (1, expanded, 1, out_channels))
                
                # No second/third layer for these architectures
                dw_filters_2 = nothing
                project_filters_2 = nothing
                dw_filters_3 = nothing
                project_filters_3 = nothing
                
                # Residual connection only for resnet and mbconv
                use_residual = (arch_type in [resnet, mbconv])
            end
        end
        
        if use_cuda
            expand_filters = isnothing(expand_filters) ? nothing : cu(expand_filters)
            dw_filters = cu(dw_filters)
            se_w1 = isnothing(se_w1) ? nothing : cu(se_w1)
            se_w2 = isnothing(se_w2) ? nothing : cu(se_w2)
            project_filters = cu(project_filters)
            gate_filters = isnothing(gate_filters) ? nothing : cu(gate_filters)
            dw_filters_2 = isnothing(dw_filters_2) ? nothing : cu(dw_filters_2)
            project_filters_2 = isnothing(project_filters_2) ? nothing : cu(project_filters_2)
            dw_filters_3 = isnothing(dw_filters_3) ? nothing : cu(dw_filters_3)
            project_filters_3 = isnothing(project_filters_3) ? nothing : cu(project_filters_3)
            mask_proj = isnothing(mask_proj) ? nothing : cu(mask_proj)
            # Keep threshold_param on CPU for scalar access
        end
        
        # Count parameters
        num_params = 0
        num_params += isnothing(expand_filters) ? 0 : length(expand_filters)
        num_params += length(dw_filters)
        num_params += isnothing(se_w1) ? 0 : length(se_w1)
        num_params += isnothing(se_w2) ? 0 : length(se_w2)
        num_params += length(project_filters)
        num_params += isnothing(gate_filters) ? 0 : length(gate_filters)
        num_params += isnothing(threshold_param) ? 0 : length(threshold_param)
        num_params += isnothing(dw_filters_2) ? 0 : length(dw_filters_2)
        num_params += isnothing(project_filters_2) ? 0 : length(project_filters_2)
        num_params += isnothing(dw_filters_3) ? 0 : length(dw_filters_3)
        num_params += isnothing(project_filters_3) ? 0 : length(project_filters_3)
        num_params += isnothing(mask_proj) ? 0 : length(mask_proj)
        
        println("CodeProcessor ($arch_type): $num_params parameters")
        println("  - in_channels: $in_channels, out_channels: $out_channels")
        if arch_type == mbconv
            println("  - expansion_ratio: $expansion_ratio, se_ratio: $se_ratio")
        end
        
        return new(expand_filters, dw_filters, se_w1, se_w2, project_filters, 
                   gate_filters, threshold_param, dw_filters_2, project_filters_2,
                   dw_filters_3, project_filters_3, mask_proj, mask_temp, mask_eta,
                   mask_gamma, use_hard_mask, use_residual, arch_type)
    end
    
    # Positional constructor for Flux/Optimisers
    CodeProcessor(expand_filters, dw_filters, se_w1, se_w2, project_filters, 
                  gate_filters, threshold_param, dw_filters_2, project_filters_2,
                  dw_filters_3, project_filters_3, mask_proj, mask_temp, mask_eta,
                  mask_gamma, use_hard_mask, use_residual, arch_type) = 
        new(expand_filters, dw_filters, se_w1, se_w2, project_filters, 
            gate_filters, threshold_param, dw_filters_2, project_filters_2,
            dw_filters_3, project_filters_3, mask_proj, mask_temp, mask_eta,
            mask_gamma, use_hard_mask, use_residual, arch_type)
end

Flux.@layer CodeProcessor

"""
    (cp::CodeProcessor)(x; training::Bool=true, step::Union{Nothing, Int}=nothing)

Forward pass through code processor.

# Process
1. Optional expansion (mbconv only)
2. Depthwise convolution + activation
3. Optional SE attention (mbconv only)
4. Projection back to output channels
5. Optional hard mask (if enabled)
6. Optional residual connection (resnet, mbconv)

# Arguments
- `x`: Input tensor (spatial, channels, 1, batch)
- `training`: Whether in training mode (affects Gumbel sampling in hard mask)
- `step`: Training step number (for temperature annealing). If nothing, uses fixed cp.mask_temp
"""
function (cp::CodeProcessor)(x; training::Bool=true, step::Union{Nothing, Int}=nothing)
    l, M, _, n = size(x)  # (spatial, channels, 1, batch)
    
    # Soft threshold architecture
    if cp.arch_type == soft_threshold
        C = M ÷ 2  # Split point between code and gradient
        code = x[:, 1:C, :, :]
        grad = x[:, C+1:end, :, :]
        
        # Extract threshold value (kept on CPU for easy access)
        threshold = abs(cp.threshold_param[1])
        
        # Learn to gate based on CODE PATTERNS (not gradient itself!)
        # This learns which gradient features are useful given the code context
        gate_logits = Flux.conv(code, cp.gate_filters; pad=0, flipped=true)
        # After 1x1 conv: (l, 1, C, n) -> reshape to (l, C, 1, n)
        gate_logits = reshape(gate_logits, (size(gate_logits, 1), size(gate_logits, 3), 1, size(gate_logits, 4)))
        
        # Sigmoid to get gates in [0, 1]
        gates = Flux.sigmoid.(gate_logits)
        
        # Apply threshold: gates below threshold → 0 (sparse!)
        gates = gates .* (gates .> threshold)
        
        # Apply learned sparse gates to gradient
        modulated_grad = grad .* gates
        
        # Combine code with gated gradient
        combined = cat(code, modulated_grad; dims=2)
        
        # Project to output
        output = Flux.conv(combined, cp.project_filters; pad=0, flipped=true)
        output = reshape(output, (l, size(cp.project_filters, 4), 1, n))
        
        return output
    end
    
    # For residual, need to extract code portion (first half of channels)
    if cp.use_residual
        # Assume input is [code; gradient] concatenated
        # Output should match code dimensions
        out_channels = size(cp.project_filters, 4)
        identity_input = x[:, 1:out_channels, :, :]
    end
    
    # Expansion (mbconv only)
    if !isnothing(cp.expand_filters)
        x = Flux.conv(x, cp.expand_filters; pad=0, flipped=true)
        x = Flux.swish.(x)
        # After 1x1 conv, channels are in dim 3: (l, 1, expanded, n)
        # Reshape to standard format: (l, expanded, 1, n)
        l_new = size(x, 1)
        expanded = size(x, 3)
        n_new = size(x, 4)
        x = reshape(x, (l_new, expanded, 1, n_new))
    end
    
    # Get current number of channels (after optional expansion)
    current_channels = size(x, 2)
    
    # Reshape for depthwise conv
    x = reshape(x, (l, 1, current_channels, n))
    
    # Depthwise convolution
    pad_h = (size(cp.dw_filters, 1) - 1) ÷ 2
    x = Flux.conv(x, cp.dw_filters; pad=(pad_h, 0), flipped=true, groups=current_channels)
    x = Flux.swish.(x)
    
    # Reshape back for projection
    x = reshape(x, (l, current_channels, 1, n))
    
    # Squeeze-Excitation (mbconv only)
    if !isnothing(cp.se_w1)
        # Need to reshape again for SE
        x_temp = reshape(x, (l, 1, current_channels, n))
        x_pooled = reshape(mean(x_temp; dims=1), (current_channels, 1, n))
        attn = Flux.NNlib.batched_mul(cp.se_w1, x_pooled)
        attn = Flux.swish.(attn)
        attn = Flux.NNlib.batched_mul(cp.se_w2, attn)
        attn = Flux.sigmoid.(attn)
        # Reshape: (current_channels, 1, n) -> (1, current_channels, 1, n)
        attn = reshape(attn, (1, current_channels, 1, n))
        x = x .* attn
    end
    
    # Projection (1x1 conv for channel mixing - now used for all architectures)
    x = Flux.conv(x, cp.project_filters; pad=0, flipped=true)
    x = reshape(x, (l, size(cp.project_filters, 4), 1, n))
    
    # Second layer for deep_plain with lightweight channel attention
    if !isnothing(cp.dw_filters_2)
        current_channels_2 = size(x, 2)
        x = reshape(x, (l, 1, current_channels_2, n))
        
        # Second depthwise conv
        pad_h_2 = (size(cp.dw_filters_2, 1) - 1) ÷ 2
        x = Flux.conv(x, cp.dw_filters_2; pad=(pad_h_2, 0), flipped=true, groups=current_channels_2)
        x = Flux.swish.(x)
        
        # Reshape and second projection
        x = reshape(x, (l, current_channels_2, 1, n))
        x = Flux.conv(x, cp.project_filters_2; pad=0, flipped=true)
        x = reshape(x, (l, size(cp.project_filters_2, 4), 1, n))
        
        # Lightweight channel attention (no extra params!)
        attn = mean(x; dims=1)  # (1, C, 1, n) - global average pooling
        attn = Flux.sigmoid.(attn)  # Soft gating
        x = x .* attn  # Channel reweighting
    end
    
    # Third layer for deep_plain with lightweight channel attention
    if !isnothing(cp.dw_filters_3)
        current_channels_3 = size(x, 2)
        x = reshape(x, (l, 1, current_channels_3, n))
        
        # Third depthwise conv
        pad_h_3 = (size(cp.dw_filters_3, 1) - 1) ÷ 2
        x = Flux.conv(x, cp.dw_filters_3; pad=(pad_h_3, 0), flipped=true, groups=current_channels_3)
        x = Flux.swish.(x)
        
        # Reshape and third projection
        x = reshape(x, (l, current_channels_3, 1, n))
        x = Flux.conv(x, cp.project_filters_3; pad=0, flipped=true)
        x = reshape(x, (l, size(cp.project_filters_3, 4), 1, n))
        
        # Lightweight channel attention (no extra params!)
        attn = mean(x; dims=1)  # (1, C, 1, n)
        attn = Flux.sigmoid.(attn)  # Soft gating
        x = x .* attn  # Channel reweighting
    end
    
    # Residual connection FIRST (if enabled)
    if cp.use_residual
        x = x .+ identity_input
    end
    
    # Apply hard mask AFTER residual (if enabled)
    if cp.use_hard_mask && !isnothing(cp.mask_proj)
        # Learn mask probabilities from the OUTPUT features (after residual)
        # x shape: (l, out_channels, 1, n)
        mask_logits = Flux.conv(x, cp.mask_proj; pad=0, flipped=true)
        # Output: (l, 1, out_channels, n) -> reshape to (l, out_channels, 1, n)
        mask_logits = reshape(mask_logits, (size(mask_logits, 1), size(mask_logits, 3), 
                                            1, size(mask_logits, 4)))
        
        # Get probabilities: p_c = sigmoid(logits)
        # Shape: (l, out_channels, 1, n) - one probability per (spatial, channel, batch) position
        p_c = Flux.sigmoid.(mask_logits)
        
        if training
            # Temperature annealing based on training steps
            # Decay: 0.5 * 0.9995^step → reaches ~0.1 after ~3000 steps
            if isnothing(step)
                temp = cp.mask_temp  # Use default if step not provided
            else
                temp = max(DEFAULT_FLOAT_TYPE(0.01), 
                          cp.mask_temp * DEFAULT_FLOAT_TYPE(0.99)^step)
            end
            
            # Gumbel(0,1) sampling: -log(-log(uniform))
            gumbel = -log.(-log.(rand(DEFAULT_FLOAT_TYPE, size(p_c)...)))
            if x isa CuArray
                gumbel = cu(gumbel)
            end
            
            # Gumbel-Softmax: sigmoid((log(p) - log(1-p) + gumbel) / temp)
            logit_p = log.(p_c .+ DEFAULT_FLOAT_TYPE(1e-8)) .- 
                     log.(1 .- p_c .+ DEFAULT_FLOAT_TYPE(1e-8))
            s_c = Flux.sigmoid.((logit_p .+ gumbel) ./ temp)
            
            # Stretch during training for smoother gradients
            z_c = min.(DEFAULT_FLOAT_TYPE(1), max.(DEFAULT_FLOAT_TYPE(0), 
                       s_c .* (cp.mask_eta - cp.mask_gamma) .+ cp.mask_gamma))
        else
            # Test time: same pipeline as training but without Gumbel noise
            # Use minimum temperature (sharpest possible)
            temp = DEFAULT_FLOAT_TYPE(0.01)
            
            # Deterministic logit transformation (no Gumbel)
            logit_p = log.(p_c .+ DEFAULT_FLOAT_TYPE(1e-8)) .- 
                     log.(1 .- p_c .+ DEFAULT_FLOAT_TYPE(1e-8))
            s_c = Flux.sigmoid.(logit_p ./ temp)
            
            # Stretch (same as training)
            z_c_soft = min.(DEFAULT_FLOAT_TYPE(1), max.(DEFAULT_FLOAT_TYPE(0), 
                       s_c .* (cp.mask_eta - cp.mask_gamma) .+ cp.mask_gamma))
            
            # Hard cutoff: truly binary {0, 1}
            z_c = DEFAULT_FLOAT_TYPE.(z_c_soft .> DEFAULT_FLOAT_TYPE(0.95))
        end
        
        # Apply mask directly (z_c already has same shape as x)
        x = x .* z_c
    end
    
    return x
end

"""
    create_code_processor(hp::HyperParameters; 
                          arch_type=mbconv,
                          kernel_size=3,
                          expansion_ratio=2,
                          use_cuda=true,
                          rng=Random.GLOBAL_RNG)

Create a CodeProcessor network based on hyperparameters.

# Arguments
- `hp`: HyperParameters (determines input/output dimensions)
- `arch_type`: Architecture type (:plain, :resnet, or :mbconv)
- `kernel_size`: Depthwise convolution kernel size
- `expansion_ratio`: Channel expansion ratio (mbconv only)
- `use_cuda`: Whether to use GPU
- `rng`: Random number generator

# Returns
- CodeProcessor instance

# Example
```julia
hp = generate_random_hyperparameters()

# Plain depthwise conv
proc_plain = create_code_processor(hp; arch_type=plain)

# ResNet-style with residual
proc_resnet = create_code_processor(hp; arch_type=resnet)

# MBConv-style with SE attention
proc_mbconv = create_code_processor(hp; arch_type=mbconv, expansion_ratio=4)

# Forward pass: concatenate code and gradient along channel dimension
# code_and_grad = cat(code, gradient; dims=2)
# output = proc(code_and_grad)  # Same size as code
```
"""
function create_code_processor(hp::HyperParameters;
                              arch_type::CodeProcessorType = mbconv,
                              kernel_size::Int = 3,
                              expansion_ratio::Int = 2,
                              use_se::Bool = true,
                              use_hard_mask::Bool = false,
                              mask_temp::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(0.5),
                              mask_eta::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(1.0),
                              mask_gamma::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(0.0),
                              use_cuda::Bool = true,
                              rng = Random.GLOBAL_RNG)
    # Get dimensions at inference code layer
    if hp.inference_code_layer == 0
        # PWM layer
        code_channels = hp.num_pfms
    else
        # Conv layer
        code_channels = hp.num_img_filters[hp.inference_code_layer]
    end
    
    # Input: code + gradient (2× channels when concatenated)
    in_channels = 2 * code_channels
    out_channels = code_channels
    
    return CodeProcessor(;
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = kernel_size,
        expansion_ratio = expansion_ratio,
        use_se = use_se,
        use_hard_mask = use_hard_mask,
        mask_temp = mask_temp,
        mask_eta = mask_eta,
        mask_gamma = mask_gamma,
        arch_type = arch_type,
        use_cuda = use_cuda,
        rng = rng
    )
end

"""
    process_code_with_gradient(processor::CodeProcessor, code, gradient; training::Bool=true, step::Union{Nothing, Int}=nothing)

Process code and gradient features through the processor network.

# Arguments
- `processor`: CodeProcessor instance
- `code`: Code features at inference layer (l, C, 1, n)
- `gradient`: Gradient features at same layer (l, C, 1, n)
- `training`: Whether in training mode
- `step`: Current training step (for temperature annealing). Counts total batches processed.

# Returns
- Processed features (l, C, 1, n) - same size as code

# Example
```julia
# In training loop
for (step, batch) in enumerate(dataloader)
    # Temperature decays: 0.5 → 0.1 over ~3000 steps
    output = process_code_with_gradient(proc, code, grad; training=true, step=step)
end
```
"""
function process_code_with_gradient(processor::CodeProcessor, code, gradient; training::Bool=true, step::Union{Nothing, Int}=nothing)
    # Concatenate along channel dimension
    combined = cat(code, gradient; dims=2)
    
    # Process through network
    return processor(combined; training=training, step=step)
end
