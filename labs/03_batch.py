"""Compare fixed batches with refilling empty slots after each round.

Run: python3 labs/03_batch.py
All requests arrive at round zero. There are two slots. Each active request
gets one token per round; every round costs one abstract time unit. No prefill,
memory limits, host overhead or variation in device time is modelled.
"""

from collections import deque

JOBS = [("A", 2), ("B", 4), ("C", 1)]


def simulate(refill):
    waiting = deque(JOBS)
    active = [None, None]
    finished = {}
    rounds = 0
    unused = 0
    while waiting or any(job is not None for job in active):
        if refill or all(job is None for job in active):
            for slot, job in enumerate(active):
                if job is None and waiting:
                    name, length = waiting.popleft()
                    active[slot] = [name, length]
        rounds += 1
        names = [job[0] if job else "-" for job in active]
        print(f"round {rounds}: {' '.join(names)}")
        for slot, job in enumerate(active):
            if job is None:
                unused += 1
                continue
            job[1] -= 1
            if job[1] == 0:
                finished[job[0]] = rounds
                active[slot] = None
    print(f"rounds: {rounds} | unused slot-rounds: {unused}")
    print("finish:", ", ".join(f"{name}={finished[name]}" for name, _ in JOBS))
    return rounds, unused, finished


if __name__ == "__main__":
    print("Fixed batches")
    fixed = simulate(refill=False)
    print("\nRefill after each round")
    rolling = simulate(refill=True)
    assert fixed == (5, 3, {"A": 2, "B": 4, "C": 5})
    assert rolling == (4, 1, {"A": 2, "C": 3, "B": 4})
