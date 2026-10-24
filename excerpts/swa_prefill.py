if self.use_swa:
    if input_metadata.is_prompt:
        load_paged_kv_cache(
            kv_cache.swa_cache,
            input_metadata.load_segments_swa,
            total_k_swa_cached,
            out=kv[:total_k_swa_cached],
        )  # [num_seqs * window_size, 1, qk_head_dim]
        kv_swa_full_index = input_metadata.kv_swa_full_index  # [total_q, window_size]
        # The SWA store slots may overlap the ranges loaded above, so prefill must store only after the cache load.
        store_kv_cache(kv_swa, kv_cache.swa_cache, input_metadata.slot_mapping_store_swa)
    else:
        kv_swa_index = input_metadata.kv_swa_index  # [total_q, window_size]
