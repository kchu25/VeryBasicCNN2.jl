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
"""
@enum CodeProcessorType plain resnet mbconv soft_threshold

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
    use_residual::Bool
    arch_type::CodeProcessorType
    
    function CodeProcessor(;
        in_channels::Int,        # code_channels + gradient_channels (concatenated)
        out_channels::Int,       # Same as code_channels
        kernel_size::Int = 3,
        expansion_ratio::Int = 2,
        se_ratio::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(8),
        arch_type::CodeProcessorType = mbconv,
        use_cuda::Bool = false,
        rng = Random.GLOBAL_RNG
    )
        init_scale = DEFAULT_FLOAT_TYPE(0.01)
        
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
            # For mbconv, resnet, and plain architectures
            gate_filters = nothing
            threshold_param = nothing
            
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
            
            # Squeeze-Excitation (only for mbconv)
            if arch_type == mbconv
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
            
            # Residual connection only for resnet and mbconv
            use_residual = (arch_type in [resnet, mbconv])
        end
        
        if use_cuda
            expand_filters = isnothing(expand_filters) ? nothing : cu(expand_filters)
            dw_filters = cu(dw_filters)
            se_w1 = isnothing(se_w1) ? nothing : cu(se_w1)
            se_w2 = isnothing(se_w2) ? nothing : cu(se_w2)
            project_filters = cu(project_filters)
            gate_filters = isnothing(gate_filters) ? nothing : cu(gate_filters)
            # Keep threshold_param on CPU for scalar access
        end
        
        return new(expand_filters, dw_filters, se_w1, se_w2, project_filters, 
                   gate_filters, threshold_param, use_residual, arch_type)
    end
    
    # Positional constructor for Flux/Optimisers
    CodeProcessor(expand_filters, dw_filters, se_w1, se_w2, project_filters, 
                  gate_filters, threshold_param, use_residual, arch_type) = 
        new(expand_filters, dw_filters, se_w1, se_w2, project_filters, 
            gate_filters, threshold_param, use_residual, arch_type)
end

Flux.@layer CodeProcessor

"""
    (cp::CodeProcessor)(x)

Forward pass through code processor.

# Process
1. Optional expansion (mbconv only)
2. Depthwise convolution + activation
3. Optional SE attention (mbconv only)
4. Projection back to output channels
5. Optional residual connection (resnet, mbconv)
"""
function (cp::CodeProcessor)(x)
    l, M, _, n = size(x)  # (spatial, channels, 1, batch)
    
    # Soft threshold architecture
    if cp.arch_type == soft_threshold
        C = M ÷ 2  # Split point between code and gradient
        code = x[:, 1:C, :, :]
        grad = x[:, C+1:end, :, :]
        
        # Extract threshold value (kept on CPU for easy access)
        threshold = abs(cp.threshold_param[1])
        
        # Learn gate weights from gradient features
        gate = Flux.conv(grad, cp.gate_filters; pad=0, flipped=true)
        # After 1x1 conv: (l, 1, C, n) -> reshape to (l, C, 1, n)
        gate = reshape(gate, (size(gate, 1), size(gate, 3), 1, size(gate, 4)))
        
        # Iterative soft thresholding (ISTA-style unrolling)
        # Multiple iterations to progressively sparsify
        num_iterations = 3
        step_size = DEFAULT_FLOAT_TYPE(0.3)
        
        for iter in 1:num_iterations
            # Gradient descent step toward original signal
            gate = gate .+ step_size .* (grad .- gate)
            
            # Soft thresholding operation
            gate = sign.(gate) .* max.(abs.(gate) .- threshold, 0)
        end
        
        # Normalize gate to [0, 1] range for stability
        max_gate = maximum(abs.(gate); dims=(1,2,3,4)) .+ DEFAULT_FLOAT_TYPE(1e-8)
        gate = gate ./ max_gate
        
        # Apply gating to gradient
        modulated_grad = grad .* gate
        
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
        attn = reshape(attn, (1, current_channels, 1, 1))
        x = x .* attn
    end
    
    # Projection (1x1 conv for channel mixing - now used for all architectures)
    x = Flux.conv(x, cp.project_filters; pad=0, flipped=true)
    x = reshape(x, (l, size(cp.project_filters, 4), 1, n))
    
    # Residual connection
    if cp.use_residual
        return x .+ identity_input
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
        arch_type = arch_type,
        use_cuda = use_cuda,
        rng = rng
    )
end

"""
    process_code_with_gradient(processor::CodeProcessor, code, gradient)

Process code and gradient features through the processor network.

# Arguments
- `processor`: CodeProcessor instance
- `code`: Code features at inference layer (l, C, 1, n)
- `gradient`: Gradient features at same layer (l, C, 1, n)

# Returns
- Processed features (l, C, 1, n) - same size as code
"""
function process_code_with_gradient(processor::CodeProcessor, code, gradient)
    # Concatenate along channel dimension
    combined = cat(code, gradient; dims=2)
    
    # Process through network
    return processor(combined)
end
