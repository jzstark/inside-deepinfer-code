req = row_to_req.long()
return SamplingKernelInputs(
    LogitsTransformInputs(params.logits.temperatures[req], params.logits.top_ps[req], params.logits.top_ks[req]),
    params.seq_seeds.view(torch.int64)[req].view(torch.uint64),
    params.sample_positions[req] + local_step * DRAFT_POS_HASH_MUL,
)
