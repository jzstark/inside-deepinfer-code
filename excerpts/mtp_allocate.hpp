seqlens_q.assign(num_seqs, 1);
for (int i = 0; i < best_bsz - num_seqs; ++i) {
    seqlens_q[acc_queue_for_seqs_q[i].second] += 1;
}
