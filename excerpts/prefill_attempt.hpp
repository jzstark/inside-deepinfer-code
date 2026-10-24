bool is_any_point = (decode_step != 0) && (decode_step % prefill_any_attempt_interval == 0);
bool is_all_point = (decode_step % prefill_attempt_interval == 0);
if (!is_any_point && !is_all_point) {
    return _schedule_decode(acquire);
}
bool want = (get_gpu_cache_usage() <= GPU_CACHE_PREFILL_WATERMARK) && _has_prefill_waiting();
