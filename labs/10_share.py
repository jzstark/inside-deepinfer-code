"""Share full prefix blocks, copy a partial tail, and release a pin safely.

Run: python3 labs/10_share.py
Blocks hold token IDs in place of KV rows; four positions per block, block 0
means "none". The copy queue stands for device work that runs after the host
has moved on. The new writer is assigned to write without waiting for it.
"""

from collections import deque

BLOCK_SIZE = 4


class Pool:
    def __init__(self, num_blocks):
        self.refs = [0] * num_blocks
        self.rows = [[] for _ in range(num_blocks)]
        self.free = deque(range(1, num_blocks))

    def allocate(self):
        block = self.free.popleft()
        self.refs[block] = 1
        return block

    def pin(self, blocks):
        for block in blocks:
            assert self.refs[block] > 0, "cannot share a free block"
            self.refs[block] += 1

    def release(self, blocks):
        for block in reversed(blocks):
            self.refs[block] -= 1
            assert self.refs[block] >= 0
            if self.refs[block] == 0:
                self.free.append(block)


def fork(pool, parent, processed, copies):
    child = list(parent)
    pool.pin(child)
    keep = processed // BLOCK_SIZE
    unpin = child[keep:]
    child[keep:] = []
    if processed % BLOCK_SIZE:
        tail = pool.allocate()
        copies.append((unpin[0], tail))
        child.append(tail)
    return child, unpin


def run(release_before_copy, processed):
    pool, copies = Pool(5), []
    prompt = list(range(100, 100 + processed))
    parent = [pool.allocate() for _ in range(-(-processed // BLOCK_SIZE))]
    for i, block in enumerate(parent):
        pool.rows[block] = prompt[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
    child, unpin = fork(pool, parent, len(prompt), copies)
    refs = " ".join(f"{b}:{pool.refs[b]}" for b in range(1, 5))
    print(f"  fork: parent={parent} child={child} refs {refs} copy={copies}")
    pool.release(parent)
    assert pool.refs[child[0]] == pool.refs[child[1]] == 1
    print(f"  parent finishes: free={list(pool.free)}; shared {child[:2]} kept")
    if release_before_copy:
        pool.release(unpin)
    writer = pool.allocate() if pool.free else None
    if writer is not None:
        pool.rows[writer] = [900, 901]
        print(f"  writer receives block {writer} before the copy runs")
    for source, destination in copies:
        pool.rows[destination] = list(pool.rows[source])
    if not release_before_copy:
        pool.release(unpin)
    if writer is None:
        writer = pool.allocate()
        pool.rows[writer] = [900, 901]
        print(f"  writer waits; receives block {writer} after the copy")
    rows = [row for block in child for row in pool.rows[block]][:len(prompt)]
    print(f"  child reads {rows[:2]}...{rows[-2:]}: correct={rows == prompt}")
    return rows == prompt, bool(copies)


if __name__ == "__main__":
    PROCESSED = 10
    print("Release the tail pin before the queued copy")
    early, copied = run(release_before_copy=True, processed=PROCESSED)
    print("\nRelease the tail pin after the queued copy")
    late, _ = run(release_before_copy=False, processed=PROCESSED)
    # Early release corrupts the child only when a tail copy was queued.
    assert late and early == (not copied)
