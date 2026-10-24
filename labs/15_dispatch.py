"""Dispatch token-expert assignments to ranks, compute them and combine results.

Run: python3 labs/15_dispatch.py
Experts are small integer functions and routing weights are fractions, so the
distributed result can be checked exactly against direct evaluation. Each rank
owns its own tokens, as attention data parallelism does. The replica rule copies
DeepInfer's gate kernels: (rank + token index * 23333) mod the replica count.
This is a teaching model of data movement, not DeepInfer's kernels.
"""

from collections import Counter
from fractions import Fraction as F

EXPERTS = {"A": lambda x: 2 * x + 1, "B": lambda x: 3 * x,
           "C": lambda x: x - 4, "D": lambda x: 5 * x + 2}


def shared(x):  # the always-used shared expert, computed on the token's rank
    return x + 10


# (origin rank, index in that rank's batch, value, routed selections)
TOKENS = [
    (0, 0, 3, [("A", F(3, 4)), ("B", F(1, 4))]),
    (0, 1, 5, [("A", F(1, 2)), ("C", F(1, 2))]),
    (0, 2, 2, [("D", F(2, 3)), ("A", F(1, 3))]),
    (1, 0, 7, [("A", F(5, 6)), ("D", F(1, 6))]),
    (1, 1, 1, [("B", F(1, 2)), ("C", F(1, 2))]),
    (1, 2, 4, [("A", F(2, 5)), ("B", F(3, 5))]),
]


def direct(token, top_k):
    _, _, x, picks = token
    weights = [w for _, w in picks[:top_k]]
    norm = sum(weights)  # top-1 keeps only the first choice, with weight one
    return shared(x) + sum(w / norm * EXPERTS[e](x) for (e, _), w in zip(picks, weights))


def run(title, slots, top_k):
    """slots: physical slot -> logical expert; slots are split evenly over two ranks."""
    per_rank = len(slots) // 2
    replicas = {e: [s for s, name in enumerate(slots) if name == e] for e in EXPERTS}
    inbox, sent = {0: [], 1: []}, 0
    for token in TOKENS:  # dispatch: each selection goes to one physical slot
        rank, index, x, picks = token
        norm = sum(w for _, w in picks[:top_k])
        for expert, weight in picks[:top_k]:
            copies = replicas[expert]
            slot = copies[(rank + index * 23333) % len(copies)]
            owner = slot // per_rank
            inbox[owner].append(((rank, index), slot, x, weight / norm))
            sent += owner != rank
    partial = {}  # compute on the owner; the weight is applied before return
    for owner, items in inbox.items():
        for key, slot, x, weight in items:
            partial.setdefault(key, []).append(weight * EXPERTS[slots[slot]](x))
    combined = {(r, i): shared(x) + sum(partial[(r, i)]) for r, i, x, _ in TOKENS}
    expected = {(r, i): direct((r, i, x, p), top_k) for r, i, x, p in TOKENS}
    loads = [len(inbox[0]), len(inbox[1])]
    print(f"{title}: slots={slots}, top-{top_k}")
    print(f"  assignments per rank={loads}, busiest={max(loads)}, sent to another rank={sent}")
    counts = Counter((slot, slots[slot]) for items in inbox.values() for _, slot, _, _ in items)
    print("  per slot:", ", ".join(f"{s}:{e}={n}" for (s, e), n in sorted(counts.items())))
    assert combined == expected
    return loads, combined


if __name__ == "__main__":
    one, _ = run("1. one expert per token", ["A", "B", "C", "D"], 1)
    two, result = run("2. two weighted experts", ["A", "B", "C", "D"], 2)
    moved, _ = run("3. same experts, other placement", ["A", "C", "B", "D"], 2)
    copies, same = run("4. replicas of A and B", ["A", "B", "C", "D", "A", "B"], 2)
    assert result == same and two == [8, 4] and moved == [7, 5] and copies == [6, 6]
    print("token results:", ", ".join(f"{k}={v}" for k, v in sorted(result.items())))
