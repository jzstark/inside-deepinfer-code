assert wait_projected_kv is not None
assert self.PackedWkv is not None
return self.PackedWkv.forward_store_kv_cache(
    wait_projected_kv(),
    kv_cache.ca_cache,
    kv_cache.ca_state,
    input_metadata,
    return_pre_rope_kv=self.has_local_index,
)
