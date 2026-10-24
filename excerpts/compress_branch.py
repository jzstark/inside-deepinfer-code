if self.kv_compress_ratio == 1:
    assert projected_kv.dtype == torch.bfloat16 and state_cache is None
    if input_metadata.mirror_cp_reverse_index is not None:
        projected_kv = gather(projected_kv.contiguous(), input_metadata.mirror_cp_reverse_index)
    return projected_kv

assert projected_kv.dtype == torch.float32 and state_cache is not None
score, latent = projected_kv.chunk(2, dim=-1)
