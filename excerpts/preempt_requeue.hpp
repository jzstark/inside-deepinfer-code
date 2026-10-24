    waiting.push_front(seq);
    seq->schedule.status = SequenceStatus::WAITING;
    free_seq_nolock(*seq);
}
