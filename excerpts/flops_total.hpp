// Return (attention, total); total already includes attention. Scheduler
// calls this directly from its existing per-request metadata construction.
std::pair<int64_t, int64_t> calc_flops(int q, int k, bool is_prompt, int full_len, int swa_valid_start = 0) const {
    const auto [full, consumer] = query_ranges(q, k, is_prompt, full_len, swa_valid_start);
    const int64_t attn = calc_attn_flops(full, consumer, is_prompt);
    const int64_t non_attn = calc_non_attn_flops(full.rows(), consumer.rows());
    const int64_t logits = calc_logits_flops(q, is_prompt);
    return {attn, attn + non_attn + logits};
}
