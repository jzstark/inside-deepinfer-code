if (log_stats) {
    const bool is_prompt = scheduler_outputs.prompt_run;
    const int full_len = seq->input.tokens.get_len();
    const auto [attn, total] =
        flops_estimator->calc_flops(seqlen_q, seqlen_k, is_prompt, full_len, is_prompt ? seq->cache.swa_valid_start : 0);
    attn_flops += attn;
    total_flops += total;
}
