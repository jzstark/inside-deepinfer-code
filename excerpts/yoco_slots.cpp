bool in_rect = pos >= data_len - keep_tokens;
if (in_rect && q_len - j <= swa_block_size)
    p_slot_c[i_qc] = p_seq->state_id * swa_block_size + pos % swa_block_size;
else
    p_slot_c[i_qc] = -1;
