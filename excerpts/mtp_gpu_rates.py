# Phase 1: compute cumprod, reading confidence rates via remap (0.0 for fresh seqs)
with T.Kernel(num_seqs, threads=128) as pid:
    acc = T.alloc_var(T.float32)
    acc = 1.0
    T.pdl_sync()
    prev_idx = remap[pid]
    if prev_idx == -1:
        acc = 0.0
    for j in T.serial(nextn):
        if prev_idx != -1:
            acc = acc * prev_confidence_rates[prev_idx, j]
        cum_rates[pid, j] = acc
