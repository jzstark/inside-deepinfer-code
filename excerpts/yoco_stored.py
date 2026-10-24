self.swa_num_layer = sum(int(GPUCacheInterface._has_SWA(attn_module)) for attn_module in runtime_attn_modules)
self.swa_store_num_layer = sum(
    int(GPUCacheInterface._has_SWA(attn_module))
    for layer_id, _, attn_module in iter_runtime_attn_modules(attn_module_list, cfg.num_hidden_layers)
    if not _is_dropped_swa_layer(cfg, layer_id)
)
self.swa_drop_num_layer = self.swa_num_layer - self.swa_store_num_layer
self.swa_block_size_kvstore = get_kvstore_swa_block_size(window_size)
self.swa_block_size_inference = get_inference_swa_block_size(window_size)
