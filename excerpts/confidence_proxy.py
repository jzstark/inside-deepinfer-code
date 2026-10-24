if accepted_s[0]:
    # accept_probs still requires the full fixed-point residual mass.
    residual_mass = T.alloc_reducer((1,), sample_cfg.scale_dtype, replication="all")
    T.fill(residual_mass, 0)
    for j in T.Parallel(SAMPLE_BD):
        residual_mass[0] += sample_acc_frag[j]
    T.finalize_reducer(residual_mass)
    if T.get_thread_binding() == 0:
        accept_probs[row] = T.max(1.0 - T.cast(residual_mass[0], T.float) / sample_cfg.scale, 0.0)
