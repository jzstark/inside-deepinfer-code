std::sort(running.begin(), running.end(),
    [](const SequencePtr &a, const SequencePtr &b) { return a->get_output_len() > b->get_output_len(); });
scheduled.reserve(max_seqs);
size_t l = 0, r = running.size();
while (l < r && scheduled.size() < max_seqs) {
    auto &seq = running[r - 1];
    while (!block_manager->can_allocate_single_nolock() && r - l > 1) {
        auto victim_seq = running[l];
        _preempt(victim_seq, ignored_seqs);
        l += 1;
    }
    if (block_manager->can_allocate_single_nolock()) {
        _append_slot(*seq);
        scheduled.push_back(std::move(seq));
    } else {
        _preempt(seq, ignored_seqs);
    }
