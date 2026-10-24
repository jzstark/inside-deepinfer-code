auto can_allocate = _can_allocate_with_state(seq->get_num_blocks());
if (can_allocate == AllocStatus::LATER) {
    break;
} else if (can_allocate == AllocStatus::NEVER) {
    LOG_WARN("Input prompt ({} tokens) is too long and "
             "exceeds the capacity of block_manager",
        num_prompt_tokens);
    seq->schedule.status = SequenceStatus::FINISHED_IGNORED;
    ignored_seqs.push_back(seq);
    waiting.pop_front();
    continue;
}
