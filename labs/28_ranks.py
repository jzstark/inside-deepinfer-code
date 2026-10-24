"""Translate rank grouping for two single-node examples; start no processes.

Run: python3 labs/28_ranks.py. Four devices, no DEVICE_OFFSET, engine_rank equals
the example launcher DP rank, and no extra Engram replicas. The equations come
from WorkerBase.global_rank and ParallelProcessGroup.create_process_groups.
Groups are memberships over the same workers, not additional process pools.
"""


def layout(dp, cp, ep):
    world = dp * cp
    assert world % ep == 0
    workers = [(e * cp + c, e, c, "driver" if c == 0 else "child")
               for e in range(dp) for c in range(cp)]
    dp_groups = [[e * cp + c for e in range(dp)] for c in range(cp)]
    cp_groups = [[e * cp + c for c in range(cp)] for e in range(dp)]
    ep_groups = [list(range(start, start + ep)) for start in range(0, world, ep)]
    for groups in (dp_groups, cp_groups, ep_groups):
        assert sorted(rank for group in groups for rank in group) == list(range(world))
    assert len(workers) == world and sum(c == 0 for _, _, c, _ in workers) == dp
    return workers, dp_groups, cp_groups, ep_groups


def main():
    for dp, cp in ((4, 1), (2, 2)):
        workers, dg, cg, eg = layout(dp, cp, 4)
        print(f"DP={dp}, CP={cp}, EP=4: {len(workers)} device workers")
        print("(global rank, engine rank, CP rank, process role):", workers)
        print("DP groups:", dg)
        print("CP groups:", cg)
        print("EP groups:", eg)
    assert layout(2, 2, 4)[1:] == ([[0, 2], [1, 3]], [[0, 1], [2, 3]], [[0, 1, 2, 3]])
    print("Checks: each grouping partitions the same four ranks")


if __name__ == "__main__":
    main()
