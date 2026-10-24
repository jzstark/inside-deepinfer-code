if self.use_swa:
    attn_kv_args = dict(
        key_cache=kv_cache.swa_cache,
        indices_in_kvcache=kv_swa_index,
        key_cache_extra=kv_cache.ca_cache,
        indices_in_kvcache_extra=kv_ca_index,
        index_length_extra=dense_index_len,
    )
