compress_token = compress_ratio
state_cache_size = state_cache.shape[1]
assert state_cache.shape[1] >= compress_ratio + max_seqlen_q - 1

# store: each token writes to cache
for idx in range(total_q):
    sidx = state_block_idx[idx].item()
    cpos = (context_lens[idx].item() - 1) % state_cache_size
    state_cache[sidx, cpos, 0, ...] = score[idx, ...]
    state_cache[sidx, cpos, 1, ...] = latent[idx, ...]

# compute: check each token's position
kv_compressed = torch.zeros(total_q, dim, dtype=torch.bfloat16, device=score.device)
for idx in range(total_q):
    abs_pos = context_lens[idx].item() - 1
    if abs_pos % compress_ratio == compress_ratio - 1:
