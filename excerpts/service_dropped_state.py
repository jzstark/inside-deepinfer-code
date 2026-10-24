dropped_compress_bytes = cls.dropped_bytes_inference - cls.swa_drop_total_bytes_inference_pad
state_tensor[dropped_tensor_offset : dropped_tensor_offset + dropped_compress_bytes].copy_(
    state_cache[state_id, dropped_cache_offset : dropped_cache_offset + dropped_compress_bytes]
)
