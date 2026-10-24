result = fused_norm_rope_hadamard_cast_store_kvcache(
    kv_swa,
    norm_args=NormArgs(self.kv_swa_norm.weight, self.kv_swa_norm.variance_epsilon),
    rope_args=RopeArgs(
        input_metadata.positions,
        self.rotary_emb.cos_sin_cache,
        self.qk_rope_head_dim,
        self.rotary_emb.is_neox_style,
        offset=self.qk_head_dim - self.qk_rope_head_dim,
    ),
    cast_args=cast_args,
    out_args=out_args,
)
if not input_metadata.is_prompt:
    assert result is None
    return None, None
