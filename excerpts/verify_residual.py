    for j in T.Parallel(SAMPLE_BD):
        pf[j] = T.Select(
            i0 * SAMPLE_BD + j < vocab_size,
            T.max(pf[j] - T.exp(draft_lf[j] - DRAFT_PROB_PENALTY), 0.0),
            0.0,
        )
if do_sample:
    accumulate(sample_cfg, pf, sample_acc_frag)
