if state_tensor is None:
    state_size = cls.total_bytes_inference if TO_CE else cls.state_size_to_disk
    state_tensor = torch.empty(state_size, dtype=state_cache.dtype, device=state_cache.device)

cache_views, tensor_views = cls._carve_state_views(state_id, state_cache, state_tensor, TO_CE=TO_CE)
for tensor_view, cache_view in zip(tensor_views, cache_views, strict=True):
    if tensor_view.numel() == 0:
        continue
    select_state_slice(tensor_view, cache_view, last_hit_position)
