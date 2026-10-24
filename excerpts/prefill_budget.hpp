if (seq_workload > (max_num_batched_tokens - num_batched_tokens) / cp_align * cp_align) {
    new_workload = std::min(seq_workload, (max_num_batched_tokens - num_batched_tokens) / cp_align * cp_align);
    seqlen_q = new_workload;
    TORCH_CHECK(
        seqlen_q < seq_workload, "_schedule: is_chunked must be true:, seqlen_q:", seqlen_q, " seq_workload: ", seq_workload);
}
