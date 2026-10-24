QueryRange consumer = full;
if (is_prompt && (compact || has_truncated_swa)) {
    const int cutoff = full_len - keep_tokens;
    // A fully clipped compact chunk still executes its last query row.
    if (compact)
        consumer.begin = std::clamp(cutoff, start, k - 1);
    const int swa_start = std::max(swa_valid_start, cutoff);
    consumer.swa_pairs =
        window_size > 0 ? pairs(std::max(0, consumer.begin - swa_start), std::max(0, k - swa_start), 1, window_size) : 0;
}
return {full, consumer};
