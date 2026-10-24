def _needs_index_state(attn_module: AttrDict) -> bool:
    if attn_module.cfg.get("index_topk", 0) <= 0:
        return False
    index_cfg = attn_module.cfg.get("attn_index")
    return index_cfg is None or not index_cfg.get("use_kproj", False)


@lru_cache(8)
def get_inference_compress_block_size(compress_ratio: int) -> int:
    """Compression ring plus rollback history; environment changes require clearing the cache."""
    return compress_ratio + max(0, envs.DEEPINFER_MAX_BACKTRACKING_TOKENS - 1)
