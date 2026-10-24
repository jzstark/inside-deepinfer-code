seqlens_k += seqlens_q
mask = remap != -1  # [n_seq] bool
# Clamp remap to valid index (use 0 for fresh seqs, will be masked out)
remap_clamped = remap.clamp(min=0)  # [n_seq]
# Gather prev_num_accepted_tokens for all seqs (invalid ones get tokens[0], doesn't matter)
accepted = prev_num_accepted_tokens[remap_clamped.long()]  # [n_seq]
# Compute adjustment: only apply where mask is True
adjustment = torch.where(mask, nextn - accepted, torch.zeros_like(seqlens_k))
seqlens_k -= adjustment
