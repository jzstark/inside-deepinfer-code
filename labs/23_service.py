"""A deterministic P/D queue, not measured engine performance.

Run: python3 labs/23_service.py. Four requests arrive at time zero with
1024-token prompts and the same prefix, cached only on P0 in the local case.
Matching takes one time unit and runs independently for each request. Two
prefill workers process 128 uncached tokens/unit; two decode workers each
serve one request for four units. A handoff has unlimited parallel capacity.
Prefix fetching costs zero. There is no batching, contention or cache eviction. Change the
handoff delay to see which waits move; capacity arithmetic is a separate model.
"""

from math import ceil


def simulate(shared, hit=768, transfer=1):
    prefill_free = [0, 0]
    decode_free = [0, 0]
    prefilled = []
    for request in range(4):
        candidates = range(2) if shared else [0]
        rank = min(candidates, key=lambda i: (prefill_free[i], i))
        start = max(1, prefill_free[rank])  # matching completes before routing
        end = start + (1024 - hit) // 128
        prefill_free[rank] = end
        prefilled.append((end + transfer, request, rank, start, end))
    rows = []
    decode_intervals = [[], []]
    for ready, request, rank, start, end in sorted(prefilled):
        decoder = min(range(2), key=lambda i: (decode_free[i], i))
        decode_start = max(ready, decode_free[decoder])
        done = decode_start + 4
        decode_intervals[decoder].append((decode_start, done))
        decode_free[decoder] = done
        rows.append((request, rank, start, end, decode_start, done))
    assert all(1 <= start <= end and end + transfer <= ds < done
               for _, _, start, end, ds, done in rows)
    for rank in range(2):
        prefill_intervals = sorted((r[2], r[3]) for r in rows if r[1] == rank)
        for intervals in (prefill_intervals, decode_intervals[rank]):
            assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
    if not shared:
        assert all(r[1] == 0 for r in rows)
    return sorted(rows)


def capacity(hit_fraction):
    # Assigned sustained rates at this workload, with an 80% utilisation target.
    arrivals, prompt, output = 4, 1024, 64
    prefill_rate, decode_rate, utilisation = 2048, 128, 0.8
    uncached = arrivals * prompt * (1 - hit_fraction)
    return (ceil(uncached / (prefill_rate * utilisation)),
            ceil(arrivals * output / (decode_rate * utilisation)))


def main():
    print("Assigned time units; four requests, two P and two D workers")
    cases = (("local affinity", False, 768, 1, 14),
             ("shared cache", True, 768, 1, 12),
             ("shared, no hits", True, 0, 1, 22),
             ("shared, slower handoff", True, 768, 3, 14),
             ("affinity, slower handoff", False, 768, 3, 16))
    completions = {}
    for name, shared, hit, transfer, expected in cases:
        rows = simulate(shared, hit, transfer)
        finish = [r[5] for r in rows]
        completions[shared, hit, transfer] = max(finish)
        assert max(finish) == expected
        print(f"{name}: P ends {[r[3] for r in rows]}; D ends {finish}")
    for transfer in (1, 3):
        assert completions[False, 768, transfer] - completions[True, 768, transfer] == 2
    for shared in (False, True):
        assert completions[shared, 768, 3] - completions[shared, 768, 1] == 2
    assert capacity(0) == (3, 3) and capacity(0.75) == (1, 3)
    print("Capacity model (GPU equivalents, not deployment sizes):")
    for hit in (0, 0.75):
        p, d = capacity(hit)
        print(f"reused input fraction {hit:.0%}: P={p}, D={d}")
    print("Checks: routing, stage dependencies, P ownership and capacity passed")


if __name__ == "__main__":
    main()
