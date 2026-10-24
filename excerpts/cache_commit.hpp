void advance_optimistically(int token_count) {
    previous_num_tokens = num_tokens;
    num_tokens += token_count;
}

void commit_processed_tokens(int token_count) { num_tokens = previous_num_tokens + token_count; }
