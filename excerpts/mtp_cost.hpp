int cur_bsz_idx = 0;
auto update_best = [&]() {
    while (cur_bsz_idx < num_bsz_selections && cur_bsz > _decode_perf_curve[cur_bsz_idx].first) {
        ++cur_bsz_idx;
    }
    if (cur_bsz_idx == num_bsz_selections) {
        return;
    }
    const auto &[graph_size, baseline_ms] = _decode_perf_curve[cur_bsz_idx];
    const double time_ms = baseline_ms + slope_ms * graph_size + sampler_ms;
    const double cur_throughput = cur_expected / time_ms;
    if (cur_throughput > best_throughput) {
        best_throughput = cur_throughput;
        best_bsz = cur_bsz;
    }
};
