struct layer_channel_mask
    mixing_filter::AbstractArray{DEFAULT_FLOAT_TYPE, 4}
    temp::DEFAULT_FLOAT_TYPE
    eta::DEFAULT_FLOAT_TYPE
    gamma::DEFAULT_FLOAT_TYPE

    function layer_channel_mask(input_channels::Int;
                                init_scale::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(0.01),
                                temp::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(0.5),
                                eta::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(1.1),
                                gamma::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(-0.1),
                                rng = Random.GLOBAL_RNG)
        mixing_filter = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                           (1, input_channels, 1, input_channels))
        return new(mixing_filter, temp, eta, gamma)
    end

    # Direct constructor for GPU/loading
    layer_channel_mask(mixing_filter, temp, eta, gamma) =
        new(mixing_filter, temp, eta, gamma)
end

Flux.@layer layer_channel_mask
Flux.trainable(l::layer_channel_mask) = (l.mixing_filter,)

function (l::layer_channel_mask; training=true)(code)
    if size(code, 1) == 1
        in_channels = size(code, 3)
        code = reshape(code, (size(code, 2), in_channels, 1, size(code, 4)))
    else
        in_channels = size(code, 2)
    end

    z = conv(code, l.mixing_filter; pad=0, flipped=true)  # (length, 1, channels, batch)
    z = reshape(z, (size(z, 1), in_channels, 1, size(z, 4)))  # (length, channels, 1, batch)    
    z = sum(z, dims=1) # shape (1, channels, 1, batch)
    mask = sigmoid.(z)
    if training
        mask = gumbel_softmax_sample(mask, l.temp, l.eta, l.gamma) # soft mask
    else
        mask = hard_threshold_mask(mask, l.temp, l.eta, l.gamma) # hard mask
    end
    return code .* mask
end

