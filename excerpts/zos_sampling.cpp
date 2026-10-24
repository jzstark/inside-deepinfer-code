metadata->seqlen_k = metadata->seqlen_q + seq.cache.num_tokens;
if (remap_ptr[i] >= 0) {
    metadata->refresh_for_sampling(seq, nextn);
}
scheduler_outputs.seqlens_q[i] = metadata->seqlen_q;
