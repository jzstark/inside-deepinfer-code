"""Sweep a hypothetical batch-cost model; no device is timed.

Run: python3 labs/05_cost.py
Context traffic grows with requests; arithmetic grows with evaluated positions.
Transfer and compute overlap perfectly, after a separate serial overhead.
Change context_scale or compute_rate to move the limiting cost.
"""

import csv
from math import isclose
from pathlib import Path

SHARED_BYTES = 4_000_000
BYTES_PER_REQUEST = 100_000
OPS_PER_POSITION = 2_000_000
BANDWIDTH = 100_000_000_000  # bytes/s; hypothetical
COMPUTE_RATE = 1_000_000_000_000  # operations/s; hypothetical
OVERHEAD = 20e-6  # seconds; serial in this model


def cost(requests, positions, context_scale=1, compute_rate=COMPUTE_RATE):
    traffic = SHARED_BYTES + requests * BYTES_PER_REQUEST * context_scale
    operations = positions * OPS_PER_POSITION
    memory_s = traffic / BANDWIDTH
    compute_s = operations / compute_rate
    step_s = OVERHEAD + max(memory_s, compute_s)
    return memory_s, compute_s, step_s


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "results"
    destination.mkdir(exist_ok=True)
    print("Hypothetical model; times in microseconds")
    print("R=N  transfer  compute     step   tokens/s")
    with (destination / "05_cost_curve.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["requests", "positions", "transfer_us", "compute_us", "step_us", "tokens_s"])
        for requests in range(1, 129):
            positions = requests  # Ordinary decode: one position per request.
            memory_s, compute_s, step_s = cost(requests, positions)
            row = (requests, positions, memory_s * 1e6, compute_s * 1e6,
                   step_s * 1e6, requests / step_s)
            writer.writerow(row)
            if requests in (1, 4, 8, 16, 32, 40, 64, 128):
                print(f"{requests:3d} {row[2]:9.1f} {row[3]:8.1f}"
                      f" {row[4]:8.1f} {row[5]:10.0f}")
    assert isclose(cost(40, 40)[0], cost(40, 40)[1])
    assert isclose(cost(16, 16)[2], cost(16, 16, compute_rate=2 * COMPUTE_RATE)[2])
    print("\nOne change at a time (microseconds)")
    print(f"R=N=16, double compute rate: {cost(16, 16)[2] * 1e6:.1f}"
          f" -> {cost(16, 16, compute_rate=2 * COMPUTE_RATE)[2] * 1e6:.1f}")
    print(f"R=N=32, four times context traffic: {cost(32, 32)[2] * 1e6:.1f}"
          f" -> {cost(32, 32, context_scale=4)[2] * 1e6:.1f}")
    print("\nSame 16 requests, more positions (microseconds)")
    print(" R   N  transfer  compute     step")
    for positions in (16, 28, 96):
        memory_s, compute_s, step_s = cost(16, positions)
        print(f"16 {positions:3d} {memory_s * 1e6:9.1f}"
              f" {compute_s * 1e6:8.1f} {step_s * 1e6:8.1f}")
    assert isclose(cost(16, 16)[2], cost(16, 28)[2])
    print("\nAssigned useful-output counts; rate = U / step_s")
    for positions, useful in [(32, 24), (64, 28)]:
        step_s = cost(16, positions)[2]
        print(f"R=16, N={positions}, U={useful}:"
              f" {step_s * 1e6:.0f} us, {useful / step_s:.0f} useful tokens/s")
