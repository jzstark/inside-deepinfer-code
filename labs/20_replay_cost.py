"""Count query rows and SWA pairs, not hardware time or whole-model FLOPs.

Standard-library translation of DeepInfer 774ce244:
csrc/engine/flops_estimator.hpp:237-279 (prefix sums and query_ranges).
The direct reference enumerates keys. Coefficients for projections, MLPs,
routers, HC, logits and global attention are intentionally not modelled.
All positions are zero-based; ranges are half-open. All examples are prefill.
"""


def clamp(value, lower, upper):
    return min(max(value, lower), upper)


def capped_prefix(n, window):
    stop = min(n, window)
    return stop * (stop + 1) // 2 + (n - stop) * window


def query_ranges(q, k, full_len, window, keep, mode, valid_start=0):
    assert mode in ("off", "mask", "compact")
    assert q > 0 and full_len >= k >= q and 0 <= valid_start <= k - q
    start = k - q
    full_pairs = capped_prefix(k - valid_start, window) - capped_prefix(start - valid_start, window)
    begin, swa_start = start, valid_start
    if mode != "off":
        cutoff = full_len - keep
        if mode == "compact":
            begin = clamp(cutoff, start, k - 1)
        swa_start = max(valid_start, cutoff)
    consumer_pairs = (capped_prefix(max(0, k - swa_start), window)
                      - capped_prefix(max(0, begin - swa_start), window))
    return q, full_pairs, k - begin, consumer_pairs


def enumerate_ranges(q, k, full_len, window, keep, mode, valid_start=0):
    full = list(range(k - q, k))
    selected = full
    if mode == "compact":
        selected = [i for i in full if i >= full_len - keep] or full[-1:]
    full_pairs = sum(1 for i in full for j in range(i + 1)
                     if j >= valid_start and i - j < window)
    consumer_pairs = sum(1 for i in selected for j in range(i + 1)
                         if j >= valid_start and i - j < window
                         and (mode == "off" or j >= full_len - keep))
    return len(full), full_pairs, len(selected), consumer_pairs


def check_cases():
    checks = 0
    for full_len in range(1, 9):
        for k in range(1, full_len + 1):
            for q in range(1, k + 1):
                for window in (1, 2, 3):
                    for keep in (1, 3, 10):
                        for valid_start in range(k - q + 1):
                            for mode in ("off", "mask", "compact"):
                                args = q, k, full_len, window, keep, mode, valid_start
                                assert query_ranges(*args) == enumerate_ranges(*args)
                                checks += 1
    return checks


def main():
    # Small windows make all keys countable by hand. Keep covers the window.
    cases = [
        ("short", 2, 2, 2, 0),
        ("long", 12, 12, 12, 0),
        ("clipped", 4, 4, 12, 0),
        ("crossing", 4, 10, 12, 0),
        ("valid-start", 2, 12, 12, 10),
    ]
    print("window=3, keep=4; entries are consumer rows / SWA pairs")
    print("case           off       mask     compact")
    for name, q, k, length, valid in cases:
        results = []
        for mode in ("off", "mask", "compact"):
            args = q, k, length, 3, 4, mode, valid
            result = query_ranges(*args)
            assert result == enumerate_ranges(*args)
            results.append(f"{result[2]:2}/{result[3]:-2}")
        print(f"{name:12} " + "   ".join(results))
    print("\nEqual layer costs, 20 producer + 20 consumer layers, keep=140")
    print("length    full row-work    compact row-work    ratio")
    for length in (64, 1024, 16384):
        full_work = 40 * length
        compact_work = 20 * length + 20 * min(length, 140)
        print(f"{length:6} {full_work:16} {compact_work:19} {compact_work / full_work:8.4f}")
    length, chunk = 1024, 256
    counts = [query_ranges(chunk, k, length, 128, 140, "compact")[2]
              for k in range(chunk, length + 1, chunk)]
    assert counts == [1, 1, 1, 140]
    print(f"four chunks: consumer rows {counts}, total={sum(counts)} (one chunk: 140)")
    print(f"PASS {check_cases()} range/pair cases against direct enumeration")


if __name__ == "__main__":
    main()
