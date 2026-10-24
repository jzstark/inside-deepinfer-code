if cls.swa_drop_num_layer > 0:
    assert envs.DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE != "off"
if not envs.DEEPINFER_PREFILL_STORE_EXTRA_STATE:
    if cls.state_size_to_disk == 0:
        return
    state_cache[state_id, : cls.state_size_to_disk] = state_tensor
    return

cache_views, tensor_views = cls._carve_state_views(state_id, state_cache, state_tensor)
for tensor_view, cache_view in zip(tensor_views, cache_views, strict=True):
    if tensor_view.numel() == 0:
        continue
    load_state_to_cache(tensor_view, cache_view, last_hit_position)
