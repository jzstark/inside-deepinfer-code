int get_num_blocks() const {
    return (get_len() + schedule.total_inflight_tokens + get_nextn_len() + cache.block_size - 1) / cache.block_size;
}
