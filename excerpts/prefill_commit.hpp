num_batched_tokens += (seqlen_q + cp_align - 1) / cp_align * cp_align;

TORCH_CHECK(seq->schedule.status == SequenceStatus::WAITING, "status ", int(seq->schedule.status));
if (seq->cache.block_table.empty()) {
    acquire.allocate_seq(*seq);
}

if (!is_chunked) {
    num_curr_seqs += 1;
    seq->schedule.status = SequenceStatus::RUNNING;
    allocate_mtp_info_slot(*seq);
}
