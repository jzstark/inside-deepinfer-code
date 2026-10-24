def is_yoco_consumer_layer(cfg: AttrDict, layer_idx: int) -> bool:
    """YOCO consumer layer: [n_prefill_layer, n_layer) in the main model."""
    return yoco_enabled(cfg) and cfg.n_prefill_layer <= layer_idx < cfg.num_hidden_layers
