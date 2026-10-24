    context, _ = decode_mla_mqa_sparse(
        q_b,
        head_dim_v=self.qk_head_dim,
        sched_meta=input_metadata.flash_mla_sched_meta_dict[self.sched_meta_key],
        softmax_scale=self.softmax_scale,
        attn_sink=self.attn_sink_params,
        batch_invariant=envs.DEEPINFER_BATCH_INVARIANT,
        fused_norm_rope_cast_kwargs=fused_kwargs,
        **attn_kv_args,
    )
return context
