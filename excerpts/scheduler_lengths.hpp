int seqlen_q = scheduler_outputs.seqlens_q[i];
int seqlen_k = seq->cache.num_tokens + seq->schedule.total_inflight_tokens + seqlen_q;
