TORCH_CHECK(seq_ptr->schedule.status == SequenceStatus::WAITING,
    SPRINT_VARS(
        "In process_outputs, should be waiting", seq.cache.num_tokens, seq.input.tokens.get_len(), seq.get_nextn_len()));
waiting.push_front(seq_ptr);
