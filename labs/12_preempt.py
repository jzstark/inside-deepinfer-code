"""Trace one memory-pressure decision and three preemption outcomes.

Run: python3 labs/12_preempt.py
This synchronous teaching model uses private four-position blocks. The next
reservation for B and C crosses a block boundary. There is no shared prefix,
device work or timing model. The branch and ordering rules are adapted from
DeepInfer scheduler.hpp:624-640 and 770-788; only selected state is represented.
"""

from collections import deque
from dataclasses import dataclass


@dataclass
class Request:
    name: str
    history: tuple
    output: int
    processed: int
    blocks: list
    state_id: int
    so_status: int = 0
    status: str = "RUNNING"


def preempt(seq, waiting, ignored, free, stop):
    if stop or seq.so_status != 0:
        seq.status = "FINISHED_PREEMPTED"
        ignored.append(seq.name)
    else:
        seq.processed = 0
        waiting.appendleft(seq.name)
        seq.status = "WAITING"
    free.extend(reversed(seq.blocks))
    seq.blocks.clear()
    seq.state_id = -1


def run(stop, so_status):
    a = Request("A", tuple(range(20)), 8, 19, list(range(1, 6)), 1,
                so_status=so_status)
    b = Request("B", tuple(range(45)), 2, 44, list(range(6, 17)), 2)
    c = Request("C", tuple(range(13)), 4, 12, list(range(17, 20)), 3)
    running = sorted([a, b, c], key=lambda seq: seq.output, reverse=True)
    waiting, ignored, free = deque(["D"]), [], deque()
    history = a.history
    print(f"stop={stop}, A.so_status={so_status}")
    print("  output order:", [(s.name, s.output) for s in running])
    candidate = running[-1]
    victim = running[0]
    print(f"  candidate={candidate.name}; victim={victim.name}; "
          f"blocks A={len(a.blocks)}, B={len(b.blocks)}, C={len(c.blocks)}")
    preempt(victim, waiting, ignored, free, stop)
    # In this chosen case, each survivor needs one more block.
    scheduled = list(reversed(running[1:]))
    for seq in scheduled:
        need = (len(seq.history) + 3) // 4
        assert need == len(seq.blocks) + 1
        seq.blocks.append(free.popleft())
    print(f"  A: {a.status}, processed={a.processed}, "
          f"blocks={a.blocks}, state={a.state_id}")
    print(f"  waiting={list(waiting)}, ignored={ignored}, free={list(free)}")
    print(f"  history retained={a.history == history}; "
          f"scheduled={[s.name for s in scheduled]}")
    assert a.history == history and a.output == 8
    assert not a.blocks and a.state_id == -1
    owned = b.blocks + c.blocks
    assert sorted(owned + list(free)) == list(range(1, 20))
    assert [s.name for s in scheduled] == ["B", "C"]
    if stop or so_status != 0:
        assert a.status == "FINISHED_PREEMPTED" and ignored == ["A"]
        assert list(waiting) == ["D"]
    else:
        assert a.status == "WAITING" and list(waiting) == ["A", "D"]
        assert a.processed == 0 and not ignored


if __name__ == "__main__":
    run(stop=False, so_status=0)
    print()
    run(stop=True, so_status=0)
    print()
    run(stop=False, so_status=1)
