assert self.shared_hidden_kv_proj is not None
projected_kv = self.shared_hidden_kv_proj(full_hidden)
with userbuffer.cp_session(enabled=use_cp):
    wait_projected_kv = _make_cp_gather_getter(projected_kv, use_cp)
    pre_rope_kv = self._store_main_kv(wait_projected_kv, kv_cache, full_metadata)
cross_layer_runtime.pre_rope_kv = pre_rope_kv
