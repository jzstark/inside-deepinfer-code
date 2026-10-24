topk_index = topk(
    p_sum,
    self.topk_tokens,
    sorted=False,
    end=topk_lens,  # We must pass `end` to the topk kernel since OOB p_sum is not cleaned.
    indices_type=torch.int32,
    sorted_index=True,
    idx_oob_fill_value=-1,
    return_value=False,
)[1]
if candidate_blocks is not None:
    restore_candidate_indices_inplace(topk_index, candidate_blocks, candidate_block_size)
