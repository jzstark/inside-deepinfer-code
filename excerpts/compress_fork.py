    cast_args=CastArgs.from_cache_format(cache_format),
    out_args=KVCacheOutArgs(
        kv_cache,
        input_metadata.slot_mapping_store_ca[self.ratio_key],
        out=pre_rope_kv,
        out_stage="norm",
        layout=cache_format,
    ),
)
return pre_rope_kv[:, 0] if pre_rope_kv is not None else None
