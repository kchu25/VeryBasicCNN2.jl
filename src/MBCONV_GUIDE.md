# MBConv Integration - Usage Guide

The refactored VeryBasicCNN now supports **optional MBConv blocks** (Mobile Inverted Bottleneck Convolution) inspired by EfficientNet, providing more sophisticated feature refinement without breaking existing functionality.

## 🎯 Key Features

- **100% Backward Compatible**: By default, `num_mbconv=0` (disabled)
- **Minimal Code Bloat**: ~60 lines for full MBConv implementation
- **EfficientNet-style**: Expansion → Depthwise → Squeeze-Excite → Projection
- **Easy to Enable**: Simple API to add MBConv blocks

## 📖 Basic Usage

### Without MBConv (Default - Same as Before)

```julia
using VeryBasicCNN

# Standard model creation (num_mbconv=0 by default)
model = create_model_nucleotides((4, 41), 244, 128)

# Works exactly as before!
predictions = model(sequences)
```

### With MBConv Blocks

```julia
using VeryBasicCNN

# Option 1: EfficientNet-style with phi coefficient (RECOMMENDED)
hp = generate_random_hyperparameters(batch_size=128)
hp_b2 = with_efficientnet_mbconv(hp, 2)  # EfficientNet-B2 style (phi=2)
model = SeqCNN(hp_b2, (4, 41), 244)

# Option 2: Manual configuration
hp_mbconv = with_mbconv(hp; num_blocks=2, expansion=4)
model = SeqCNN(hp_mbconv, (4, 41), 244)

# Option 3: Create hyperparameters directly
hp = HyperParameters(
    pfm_len = 8,
    num_pfms = 64,
    num_img_filters = [128, 256, 128],
    # ... other params ...
    num_mbconv = 3,         # Add 3 MBConv blocks
    mbconv_expansion = 6    # Expansion ratio of 6
)
model = SeqCNN(hp, (4, 41), 244)

# Forward pass works identically
predictions = model(sequences)
```

## ⚙️ EfficientNet Compound Scaling (φ)

The φ (phi) coefficient controls MBConv depth and width following EfficientNet's scaling:

```julia
# phi = 0 (B0 - Baseline)
hp_b0 = with_efficientnet_mbconv(hp, 0)  # 2 blocks, expansion=4

# phi = 1 (B1 - Deeper)  
hp_b1 = with_efficientnet_mbconv(hp, 1)  # 3 blocks, expansion=4

# phi = 2 (B2 - Deeper + Wider)
hp_b2 = with_efficientnet_mbconv(hp, 2)  # 3 blocks, expansion=6

# phi = 3 (B3 - Much Deeper)
hp_b3 = with_efficientnet_mbconv(hp, 3)  # 4 blocks, expansion=6

# phi = 4 (B4 - Very Deep)
hp_b4 = with_efficientnet_mbconv(hp, 4)  # 5 blocks, expansion=6
```

**Scaling table:**

| φ | Model | Blocks | Expansion | Description |
|---|-------|--------|-----------|-------------|
| 0 | B0 | 2 | 4 | Baseline (fastest) |
| 1 | B1 | 3 | 4 | 1.2× depth |
| 2 | B2 | 3 | 6 | 1.4× depth, wider |
| 3 | B3 | 4 | 6 | 1.8× depth |
| 4 | B4 | 5 | 6 | 2.2× depth |
| 5 | B5 | 6 | 6 | 2.6× depth |
| 6 | B6 | 7 | 6 | 3.1× depth |
| 7 | B7 | 8 | 8 | 3.6× depth (best accuracy) |


## 🏗️ Architecture Flow

### Standard Model (num_mbconv=0)
```
Input → PWMs → Conv Layers → Output
```

### Enhanced Model (num_mbconv>0)
```
Input → PWMs → Conv Layers → MBConv Blocks → Output
                              ↑
                              Refinement stage
```

Each MBConv block:
```
Input (C channels)
  ↓
Expansion (C → C×expansion_ratio via 1×1 conv)
  ↓
Depthwise Conv (spatial features, same channels)
  ↓
Squeeze-Excite (channel attention)
  ↓
Projection (C×expansion_ratio → C via 1×1 conv)
  ↓
Add Skip Connection (if input==output dims)
  ↓
Output (C channels)
```

## 🔧 Hyperparameter Configuration

### In HyperParameters
```julia
HyperParameters(
    # Standard architecture params...
    pfm_len = 8,
    num_pfms = 64,
    # ...
    
    # MBConv configuration
    num_mbconv = 2,         # Number of MBConv blocks (0 = disabled)
    mbconv_expansion = 4    # Expansion ratio (typically 4-6)
)
```

### In HyperParamRanges
```julia
HyperParamRanges(
    # Standard ranges...
    num_img_layers_range = 3:5,
    # ...
    
    # MBConv ranges
    num_mbconv_range = 0:3,              # 0-3 blocks
    mbconv_expansion_options = [4, 6]    # Expansion options
)

# Random generation will sample from these ranges
hp = generate_random_hyperparameters(ranges=my_ranges)
```

## 💡 Advanced Examples

### Architecture Search with MBConv
```julia
# Create range that includes MBConv
ranges = nucleotide_ranges()
ranges_mbconv = HyperParamRanges(
    ranges...;  # Copy standard ranges
    num_mbconv_range = 0:3,
    mbconv_expansion_options = [4, 6, 8]
)

# Generate random architectures
for i in 1:10
    hp = generate_random_hyperparameters(batch_size=128, ranges=ranges_mbconv)
    model = create_model((4, 41), 244, 128; ranges=ranges_mbconv)
    # Train and evaluate...
end
```

### Comparing Standard vs Enhanced
```julia
# Create base hyperparameters
hp_base = generate_random_hyperparameters(batch_size=128)

# Standard model (no MBConv)
model_standard = SeqCNN(hp_base, (4, 41), 244)

# EfficientNet-B2 style (phi=2: 3 blocks, expansion=6)
hp_b2 = with_efficientnet_mbconv(hp_base, 2)
model_b2 = SeqCNN(hp_b2, (4, 41), 244)

# EfficientNet-B4 style (phi=4: 5 blocks, expansion=6)
hp_b4 = with_efficientnet_mbconv(hp_base, 4)
model_b4 = SeqCNN(hp_b4, (4, 41), 244)

# All have same API
preds_std = model_standard(sequences)
preds_b2 = model_b2(sequences)
preds_b4 = model_b4(sequences)

# Compare performance/accuracy trade-offs
```

### Architecture Search with EfficientNet Scaling
```julia
# Test different EfficientNet configurations
results = []
for phi in 0:4  # B0 to B4
    hp = generate_random_hyperparameters(batch_size=128)
    hp_scaled = with_efficientnet_mbconv(hp, phi)
    model = SeqCNN(hp_scaled, (4, 41), 244)
    
    # Train and evaluate
    accuracy = train_and_eval(model, data)
    push!(results, (phi=phi, accuracy=accuracy))
end
```

### Extract Features Before/After MBConv
```julia
# Model with MBConv
hp = with_mbconv(generate_random_hyperparameters(); num_blocks=2)
model = SeqCNN(hp, (4, 41), 244)

# Extract features at different stages
# (Note: compute_code_at_layer extracts BEFORE MBConv)
code_before_mbconv = compute_code_at_layer(model, seqs, model.num_conv_layers)

# Full features (after MBConv)
features_after_mbconv = extract_features(model, seqs)

# The MBConv blocks refine the code before final linear layer
```

## 🧪 MBConv Block Details

### Components
1. **Expansion**: `1×1 Conv` to increase channels (C → C×k)
2. **Depthwise**: `k×1 Depthwise Conv` for spatial patterns
3. **Squeeze-Excite**: Channel-wise attention mechanism
4. **Projection**: `1×1 Conv` back to original channels
5. **Skip Connection**: Residual connection when dims match

### Parameters
- `in_channels`: Input channel count
- `out_channels`: Output channel count  
- `kernel_size`: Depthwise conv kernel (default: 3)
- `expansion_ratio`: Channel expansion factor (default: 4)
- `use_se`: Enable Squeeze-Excite (default: true)

## ⚡ Performance Considerations

### When to Use MBConv

**Good for:**
- Models with sufficient data (MBConv adds parameters)
- When you want better feature refinement
- Architecture search / model comparison studies
- Tasks requiring nuanced pattern recognition

**Maybe skip for:**
- Very small datasets (may overfit)
- Rapid prototyping (adds training time)
- Resource-constrained scenarios

### Computational Cost

- Each MBConv block adds ~4× params (expansion_ratio=4)
- Depthwise conv is efficient (parameter-wise)
- Squeeze-Excite adds minimal overhead
- Skip connections cost nothing

## 🎓 Implementation Notes

### Why This Design Works

1. **Optional by default**: `num_mbconv=0` → zero overhead
2. **Applied after conv layers**: Doesn't break dimension calculations
3. **Uses Flux primitives**: `Conv`, `DepthwiseConv`, `BatchNorm`
4. **Proper @layer macro**: Integrates with Flux's parameter tracking
5. **Skip connections**: Helps gradient flow in deeper models

### Code Added
- `layers.jl`: ~90 lines for MBConvBlock
- `hyperparameters.jl`: 2 fields in HyperParameters, 2 in ranges
- `model.jl`: ~10 lines for MBConv initialization
- `forward.jl`: 3 lines in forward pass
- `convert.jl`: 2 lines for CPU/GPU conversion

**Total: ~100 lines** for full EfficientNet-style MBConv integration!

## 🔍 Backward Compatibility

All existing code works without changes:

```julia
# This still works exactly as before
model = create_model_nucleotides((4, 41), 244, 128)
predictions = model(sequences)
loss = compute_training_loss(model, sequences, targets)
model_cpu = model2cpu(model)
```

The only difference: models now have an empty `mbconv_blocks` vector by default.
