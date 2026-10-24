"""A frontier from assigned records, plus an audit of a published conversion.

Run: python3 labs/25_frontier.py. These rates are synthetic, not recovered
benchmark records. CP=1 and nextn=3. The one approximate published reading
(246 tokens/s/user at 64 requests) is kept separate from the plotted records.
Exact fractions enumerate prefix acceptance and check common rescaling.
Change one point's acceptance to see why common scaling is an assumption.
"""

import csv
from fractions import Fraction as F
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ((8, 100), (16, 90), (32, 60), (48, 35), (64, 40))


def axes(tau):
    return [(r, sps * tau, r * sps * tau) for r, sps in RECORDS]


def frontier(points):
    return [p for p in points if not any(
        q[1] >= p[1] and q[2] >= p[2] and (q[1] > p[1] or q[2] > p[2])
        for q in points)]


def expected_prefix(a, n):
    expected = F(0)
    for outcomes in product((False, True), repeat=n):
        probability = F(1)
        accepted = 0
        prefix_open = True
        for outcome in outcomes:
            probability *= a if outcome else 1 - a
            if prefix_open and outcome:
                accepted += 1
            else:
                prefix_open = False
        expected += probability * (1 + accepted)
    return expected


def main():
    a, n = F(3, 4), 3
    mean_fraction_tau = 1 + n * a
    conditional_tau = 1 + sum(a ** i for i in range(1, n + 1))
    assert expected_prefix(a, n) == conditional_tau == F(175, 64)
    points = axes(mean_fraction_tau)
    keep = {r for r, _, _ in frontier(points)}
    assert keep == {8, 16, 32, 64}
    assert keep == {r for r, _, _ in frontier(axes(conditional_tau))}
    destination = ROOT / "results"
    destination.mkdir(exist_ok=True)
    for filename, rows in (("25_points.csv", points),
                           ("25_frontier.csv", sorted(frontier(points), key=lambda p: p[1]))):
        with (destination / filename).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("requests", "user_speed", "throughput"))
            writer.writerows((r, float(x), float(y)) for r, x, y in rows)
    print("Synthetic records: requests, steps/s, user tokens/s, tokens/s/GPU")
    for (r, sps), (_, x, y) in zip(RECORDS, points):
        print(f"{r:2} {sps:3} {float(x):6.2f} {float(y):7.2f}")
    print(f"Non-dominated request counts: {sorted(keep)}")
    print(f"Mean accepted fraction 0.75: tau={float(mean_fraction_tau):.6f}")
    print(f"Conditional acceptance 0.75: tau={float(conditional_tau):.6f}")
    print(f"Common axis shrink: {float(100 * (1 - conditional_tau / mean_fraction_tau)):.4f}%")
    published_user_speed = 246  # approximate visual reading, never plotted as synthetic data
    sps = published_user_speed / mean_fraction_tau
    print(f"Published point, approximate: {published_user_speed} / 3.25 = {float(sps):.2f} steps/s")
    print(f"Implied throughput at R=64, CP=1: {published_user_speed * 64} tokens/s/GPU")
    print("Checks: exact prefix expectation and common-scale frontier passed")


if __name__ == "__main__":
    main()
