const bool so_blocks_recompute = seq->structured_output.status != 0;
if (stop_preempted_request || so_blocks_recompute) {
    if (so_blocks_recompute && !stop_preempted_request) {
        LOG_WARN("Request {} is preempted and stops: structured output (so_status {}) can not be recomputed.",
            seq->request.request_id, seq->structured_output.status);
    } else {
        LOG_WARN("Request {} is preempted and stops.", seq->request.request_id);
    }
    seq->schedule.status = SequenceStatus::FINISHED_PREEMPTED;
    free_seq_nolock(*seq);
    ignored_seqs.push_back(seq);
} else {
    LOG_WARN("Request {} is preempted and will be recomputed.", seq->request.request_id);
