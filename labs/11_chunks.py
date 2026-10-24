"""Separate prompt chunk size from the rule for choosing the next request.

Run: python3 labs/11_chunks.py
One prompt chunk runs at a time. Its assigned duration is 1 + query positions,
in abstract time units. The first output arrives when the prompt completes.
No decode work, memory limits or device overlap is modelled. Rotation is an
alternative teaching policy, not a claim about DeepInfer's waiting queue.
"""

from collections import deque

JOBS = (("A", 0, 12), ("B", 3, 2), ("C", 3, 2))


def simulate(budget, rotate):
    arrivals = deque(JOBS)
    waiting = deque()
    remaining = {name: length for name, _, length in JOBS}
    first = {}
    now = steps = work = 0

    def admit():
        while arrivals and arrivals[0][1] <= now:
            name, _, _ = arrivals.popleft()
            waiting.append(name)

    while arrivals or waiting:
        if not waiting:
            now = max(now, arrivals[0][1])
        admit()
        name = waiting.popleft()
        query = min(budget, remaining[name])
        start = now
        now += 1 + query
        remaining[name] -= query
        steps += 1
        work += query
        admit()  # Arrivals during this chunk precede a rotated continuation.
        if remaining[name]:
            if rotate:
                waiting.append(name)
            else:
                waiting.appendleft(name)
        else:
            first[name] = now
        print(f"  {start:2d}->{now:2d}: {name} q={query:2d} waiting={list(waiting)}")

    delays = {name: first[name] - arrival for name, arrival, _ in JOBS}
    print("  first-token delays:", ", ".join(f"{k}={v}" for k, v in delays.items()))
    print(f"  work={work} positions; steps={steps}; finish={now}")
    assert work == sum(length for _, _, length in JOBS)
    return delays, steps, now


if __name__ == "__main__":
    print("Whole prompts, FIFO (budget 12)")
    whole = simulate(12, rotate=False)
    print("\nChunks, continuation at front (budget 4)")
    front = simulate(4, rotate=False)
    print("\nChunks, continuation at back (budget 4)")
    back = simulate(4, rotate=True)
    assert whole == ({"A": 13, "B": 13, "C": 16}, 3, 19)
    assert front == ({"A": 15, "B": 15, "C": 18}, 5, 21)
    assert back == ({"A": 21, "B": 5, "C": 8}, 5, 21)
