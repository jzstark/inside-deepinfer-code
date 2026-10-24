"""Run DeepInfer's pinned expert-placement policy on small CPU load histories.

Run: python3 labs/16_placement.py   (requires PyTorch; a CPU is enough)
Imports labs/vendor/eplb_policy.py, an unmodified copy of
deepinfer/model_executor/eplb/policy.py checked by `make check`. Only that file
is loaded; the DeepInfer package and engine are not imported or run. The score
is the policy's load estimate, not a measured step time.
"""

import importlib.util
from pathlib import Path

import torch

POLICY = Path(__file__).resolve().parent / "vendor" / "eplb_policy.py"
NAMES = "ABCD"


def load_policy():
    spec = importlib.util.spec_from_file_location("eplb_policy", POLICY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(history, slots, per_rank):
    """Sum over steps of the busiest rank; a replica takes an equal share."""
    copies = {e: slots.count(e) for e in set(slots)}
    total = 0.0
    for step in history:
        ranks = [0.0] * (len(slots) // per_rank)
        for slot, expert in enumerate(slots):
            ranks[slot // per_rank] += step[expert] / copies[expert]
        total += max(ranks)
    return total


def show(label, slots, per_rank):
    groups = ["".join(NAMES[e] for e in slots[r:r + per_rank]) for r in range(0, len(slots), per_rank)]
    return f"{label} {'/'.join(groups)}"


def place(policy, history, num_physical, per_rank):
    tokens = torch.tensor([[step] for step in history], dtype=torch.int)  # [steps, layers=1, experts]
    p2l, l2p, count = policy.rebalance_experts_global(tokens, num_physical, per_rank)
    slots, l2p, count = p2l[0].tolist(), l2p[0].tolist(), count[0].tolist()
    for expert, physical in enumerate(l2p):  # check both maps and the replica counts
        valid = [p for p in physical if p >= 0]
        assert len(valid) == count[expert] == slots.count(expert)
        assert all(slots[p] == expert for p in valid)
    assert len(slots) == num_physical and sum(count) == num_physical
    return slots, l2p, count


if __name__ == "__main__":
    policy = load_policy()
    history = [[10, 10, 0, 0], [0, 0, 10, 10]]
    print("History by step:", history)
    for label, slots in (("candidate", [0, 1, 2, 3]), ("candidate", [0, 2, 1, 3])):
        print(f"  {show(label, slots, 2)}: score={score(history, slots, 2):.0f}")
    chosen, _, _ = place(policy, history, 4, 2)
    print(f"  {show('policy', chosen, 2)}: score={score(history, chosen, 2):.0f}")
    assert chosen == [0, 2, 1, 3]

    skewed = [[30, 6, 4, 0], [20, 10, 0, 6], [40, 2, 6, 2]]
    print("\nSkewed history by step:", skewed)
    initial = [0, 1, 2, 3, 0, 1]  # get_initial_global_physical_to_logical_map(4, 2)
    print(f"  {show('initial', initial, 3)}: score={score(skewed, initial, 3):.2f}")
    slots, l2p, count = place(policy, skewed, 6, 3)
    print(f"  {show('policy', slots, 3)}: score={score(skewed, slots, 3):.2f}")
    print("  physical->logical:", [NAMES[e] for e in slots])
    print("  logical->physical:", {NAMES[e]: [p for p in row if p >= 0] for e, row in enumerate(l2p)})
    print("  replicas:", dict(zip(NAMES, count)))
    assert count == [3, 1, 1, 1] and score(skewed, slots, 3) < score(skewed, initial, 3)

    contrasting = [[10, 0], [10, 18]]
    print("\nReplica-choice history by step:", contrasting)
    # Two logical experts, one spare slot, three ranks with one slot each.
    _, _, count = place(policy, contrasting, 3, 1)
    print("  replicas:", ", ".join(f"{NAMES[e]}={n}" for e, n in enumerate(count)))
    assert count == [1, 2]
