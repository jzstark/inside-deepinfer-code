if envs.DEEPINFER_ENABLE_MTP_VARLEN:
    compute_seqlens_q_by_confidence_rates(total_q, remap[:n_seq], prev.sampler_output.confidence_rates_gpu, seqlens_q[:n_seq])
if self.nextn > 0:
    compute_seqlens_k_zos(
        seqlens_k[:n_seq],
        seqlens_q[:n_seq],
        remap[:n_seq],
        prev.sampler_output.num_accepted_tokens_gpu,
        nextn=self.nextn,
    )

d2h_evt = None
if envs.DEEPINFER_ENABLE_MTP_VARLEN:
    # H2D, correction, and D2H are ordered on the current stream; only varlen rewrites this pinned source.
    seqlens_q_cpu[:n_seq].copy_(seqlens_q[:n_seq], non_blocking=True)
    d2h_evt = torch.cuda.Event()
    d2h_evt.record()
