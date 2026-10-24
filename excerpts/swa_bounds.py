swa_len = T.min(pos + 1, window_size)
window_start_pos = pos - swa_len + 1
swa_off = state_id * swa_block_size + window_start_pos % swa_block_size
kv_swa_index_offset_len[q_off, 0] = swa_off
kv_swa_index_offset_len[q_off, 1] = swa_len

slot_mapping_store_swa[q_off] = state_id * swa_block_size + pos % swa_block_size
