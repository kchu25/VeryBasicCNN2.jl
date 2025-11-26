# ============================================================================
# CPU/GPU Conversion Utilities
# ============================================================================

"""
    model2cpu(model::SeqCNN)

Convert SeqCNN model from GPU to CPU.

# Arguments
- `model`: SeqCNN instance (potentially on GPU)

# Returns
- SeqCNN instance with all arrays moved to CPU

# Example
```julia
model_cpu = model2cpu(model)
```
"""
function model2cpu(model::SeqCNN)
    # Convert conv layers - preserve LayerNorm if present
    conv_cpu = [
        if isnothing(layer.ln_gamma)
            LearnedCodeImgFilters(layer.filters |> Array)
        else
            LearnedCodeImgFilters(
                layer.filters |> Array,
                layer.ln_gamma |> Array,
                layer.ln_beta |> Array
            )
        end
        for layer in model.conv_layers
    ]
    
    # Convert PWM layer
    pwms_cpu = LearnedPWMs(
        model.pwms.filters |> Array,
        model.pwms.activation_scaler
    )
    
    # Convert MBConv blocks
    mbconv_cpu = [block |> Flux.cpu for block in model.mbconv_blocks]
    
    # Convert output weights
    weights_cpu = model.output_weights |> Array
    
    return SeqCNN(model.hp, pwms_cpu, conv_cpu, mbconv_cpu, weights_cpu, model.final_nonlinearity)
end

"""
    model2gpu(model::SeqCNN)

Convert SeqCNN model from CPU to GPU.

# Arguments
- `model`: SeqCNN instance (potentially on CPU)

# Returns
- SeqCNN instance with all arrays moved to GPU

# Example
```julia
model_gpu = model2gpu(model)
```
"""
function model2gpu(model::SeqCNN)
    # Convert conv layers - preserve LayerNorm if present
    conv_gpu = [
        if isnothing(layer.ln_gamma)
            LearnedCodeImgFilters(layer.filters |> cu)
        else
            LearnedCodeImgFilters(
                layer.filters |> cu,
                layer.ln_gamma |> cu,
                layer.ln_beta |> cu
            )
        end
        for layer in model.conv_layers
    ]
    
    # Convert PWM layer
    pwms_gpu = LearnedPWMs(
        model.pwms.filters |> cu,
        model.pwms.activation_scaler
    )
    
    # Convert MBConv blocks
    mbconv_gpu = [block |> Flux.gpu for block in model.mbconv_blocks]
    
    # Convert output weights
    weights_gpu = model.output_weights |> cu
    
    return SeqCNN(model.hp, pwms_gpu, conv_gpu, mbconv_gpu, weights_gpu, model.final_nonlinearity)
end
