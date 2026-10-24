"""Translate DeepInfer's two-queue verification budget; use synthetic costs.

Run: python3 labs/18_verification_budget.py
CPU algorithm: csrc/engine/mtp_varlen.hpp:150-268. Rates are cumulative,
formed in scheduler.hpp:1239-1250. GPU scalar reference: tilelang_ops/cuda/
schedule_mtp_varlen.py:26-81. No engine or kernel runs. Costs are invented.
The DSpark CSV transcribes the pinned deployment Q&A table (lines 249-256);
its throughput values are DeepSeek's estimates, not this lab's measurements.
"""

import csv
from fractions import Fraction as F
from itertools import accumulate, product
from operator import mul
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cumulative(raw):
    assert all(0 <= r <= 1 for row in raw for r in row)
    return [list(accumulate(row, mul)) for row in raw]


def queue(rates):
    # C++ breaks ties by request index. It allocates counts, hence prefixes.
    return sorted([(rate, i) for i, row in enumerate(rates) for rate in row],
                  key=lambda item: (-item[0], item[1]))


def allocate(rates, total):
    lengths = [1] * len(rates)
    assert len(rates) <= total <= len(rates) + sum(map(len, rates))
    for _, i in queue(rates)[:total-len(rates)]:
        lengths[i] += 1
    assert sum(lengths) == total
    assert all(1 <= n <= 1 + len(row) for n, row in zip(lengths, rates))
    return lengths


def schedule(older, newer, curve, slope=F(0), sampler=F(2)):
    """CPU search over total positions, followed by allocation with newer rates."""
    assert list(map(len, older)) == list(map(len, newer))
    expected, size, graph_idx = F(len(older)), len(older), 0
    best_rate, best_size = F(0), 0
    candidates = []
    for increment in [F(0), *(rate for rate, _ in queue(older))]:
        expected += increment
        while graph_idx < len(curve) and size > curve[graph_idx][0]:
            graph_idx += 1
        if graph_idx < len(curve):
            graph, baseline = curve[graph_idx]
            time = baseline + max(F(0), slope) * graph + sampler
            rate = expected / time
            candidates.append((size, graph, expected, time, rate))
            if rate > best_rate:  # Keep the earlier budget on equal throughput.
                best_rate, best_size = rate, size
        size += 1
    assert best_size >= len(older)
    return best_size, allocate(newer, best_size), candidates


def exhaustive(older, curve, slope=F(0), sampler=F(2)):
    """Independent oracle: enumerate every possible per-request prefix length."""
    choices = []
    for counts in product(*(range(len(row)+1) for row in older)):
        total = len(older) + sum(counts)
        bucket = next(((g, t) for g, t in curve if g >= total), None)
        if bucket is None:
            continue
        graph, baseline = bucket
        expected = len(older) + sum(sum(row[:n]) for row, n in zip(older, counts))
        rate = expected / (baseline + max(F(0), slope) * graph + sampler)
        choices.append((rate, -total))
    return -max(choices)[1]


def gpu_lengths(raw_previous, remap, total, nextn):
    """Scalar rank counting, including fresh zero rows and flat-index ties."""
    raw = [raw_previous[j] if j >= 0 else [F(0)] * nextn for j in remap]
    rates = cumulative(raw)
    flat = [r for row in rates for r in row]
    extra = total - len(remap)
    assert 0 <= extra <= len(flat)
    lengths = [1] * len(remap)
    for i, rate in enumerate(flat):
        rank = sum(other > rate or (other == rate and j < i)
                   for j, other in enumerate(flat))
        if rank < extra:
            lengths[i // nextn] += 1
    assert lengths == allocate(rates, total) and sum(lengths) == total
    return lengths


if __name__ == "__main__":
    older = cumulative([[F(9, 10), F(4, 5), F(1, 2)],
                        [F(4, 5), F(1, 2), F(1, 4)]])
    newer = cumulative([[F(2, 5), F(1, 2), F(1, 2)],
                        [F(19, 20), F(9, 10), F(4, 5)]])
    print("Cumulative confidence; two requests, three draft positions each")
    print("  N-2:", [[float(v) for v in row] for row in older])
    print("  N-1:", [[float(v) for v in row] for row in newer])
    curves = {
        "flat": [(2, 8), (4, 8), (6, 8), (8, 8)],
        "stairs": [(2, 8), (4, 10), (6, 12), (8, 28)],
        "steep": [(2, 8), (4, 10), (6, 23), (8, 38)],
        "moved": [(2, 8), (5, 10), (6, 23), (8, 38)],
    }
    destination = ROOT / "results"
    destination.mkdir(exist_ok=True)
    all_rows = {}
    for name, curve in curves.items():
        total, lengths, rows = schedule(older, newer, curve)
        all_rows[name] = rows
        assert total == exhaustive(older, curve)
        print(f"\n{name}: synthetic time units; sampler=2, slope=0")
        print("  positions graph expected time  output/time")
        for n, g, e, t, rate in rows:
            print(f"  {n:9d} {g:5d} {float(e):8.2f} {float(t):4.0f} {float(rate):12.4f}")
        print(f"  chosen total={total}, q(N-1)={lengths}")
        print(f"  allocation using N-2 instead: {allocate(older, total)}")
    with (destination / "18_budget_curve.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["positions", "expected", *[f"{name}_{metric}" for name in curves for metric in ("time", "rate")]])
        for i, row in enumerate(all_rows["flat"]):
            writer.writerow([row[0], float(row[2]),
                             *[float(all_rows[name][i][j]) for name in curves for j in (3, 4)]])
    assert [schedule(older, newer, c)[0] for c in curves.values()] == [8, 6, 4, 5]
    # Check nonzero/clamped slopes, equal scores, absent drafts and the last graph.
    for slope in (F(-1), F(1), F(3)):
        assert schedule(older, newer, curves["stairs"], slope)[0] == exhaustive(older, curves["stairs"], slope)
    assert schedule([[F(1)]], [[F(1)]], [(1, 1), (2, 2)], sampler=F(0))[0] == 1
    assert schedule([[]], [[]], [(1, 1)])[1] == [1]
    assert schedule(older, newer, [(2, 8), (4, 10)])[0] <= 4

    prev = [[F(9, 10), F(4, 5)], [F(1, 2), F(1, 2)]]
    remap = [0, -1, 1]  # Mixed batch: two survivors and one new row.
    print("\nGPU rank-count reference; remap=[0,-1,1]")
    for budget, want in [(7, [3, 1, 3]), (8, [3, 2, 3])]:
        lengths = gpu_lengths(prev, remap, budget, 2)
        assert lengths == want
        print(f"  total={budget}: q={lengths}")
    cpu_rates = [cumulative(prev)[0], [F(1), F(1)], cumulative(prev)[1]]
    assert allocate(cpu_rates, 7) == [3, 3, 1]
    print(f"  CPU missing-rate fill=1, same total=7: q={allocate(cpu_rates, 7)}")
    print("  Both conserve the budget; zero confidence can receive surplus slots.")

    print("\nDSpark: DeepSeek's reported throughput estimates, not lab timings")
    print("  concurrency fixed dynamic gain")
    with (ROOT / "labs/data/18_dspark.csv").open() as file:
        for row in csv.DictReader(file):
            fixed, dynamic = int(row["fixed_tokens_s"]), int(row["dynamic_tokens_s"])
            gain = (F(dynamic, fixed) - 1) * 100
            assert abs(gain - F(row["reported_gain_pct"])) <= F(1, 20)
            print(f"  {int(row['concurrency']):11d} {fixed:5d} {dynamic:7d} {float(gain):+5.1f}%")
