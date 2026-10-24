self.use_fused_norm_rope_attn_rope_cast = (
    # disabled in dense packed-KV decoding
    ((not envs.DEEPINFER_BATCH_INVARIANT and cfg.engine_type == EngineType.PROMPT) or self.is_sparse or not self.has_packed_kv)
    and self.num_heads in (64, 128)
    and PLATFORM.is_gpu
)
