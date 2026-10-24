accept_lp = T.min(target_lp - draft_lp, T.cast(0, T.float))
verify_seq = sample_positions[row] * 3 + 2
T.rng_init(seed_val, verify_seq)
accepted_s[0] = T.rng_rand_float(32, "uniform") < T.exp(accept_lp)
