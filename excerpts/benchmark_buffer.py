stable_prefill: deque[tuple[int, float, bool]] = deque(maxlen=prefill_measure_steps + 1)
stable_decode: deque[tuple[float, bool]] = deque(maxlen=decode_measure_steps + 1)
