int32_t clip_prefix = compact ? std::clamp(data_len - keep_tokens - q_start, 0, q_len - 1) : 0;
clip_prefixes[i_seq] = clip_prefix;
full_bases[i_seq] = full_base;
int32_t keep_len = q_len - clip_prefix;
p_cu_c[i_seq + 1] = p_cu_c[i_seq] + keep_len;
full_base += q_len;
