"""Give growing requests fixed-size blocks from one finite pool.

Run: python3 labs/09_blocks.py
Blocks hold four token positions (illustrative). Block 0 is reserved to mean
"no block". Each block has an owner count and returns to the free list at zero.
"""

from collections import deque

BLOCK_SIZE = 4


class Pool:
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.refs = [0] * num_blocks
        self.free = deque(range(1, num_blocks))

    def check(self, count):
        if count >= self.num_blocks:
            return "NEVER"
        return "OK" if len(self.free) >= count else "LATER"

    def allocate(self, count):
        assert self.check(count) == "OK"
        blocks = [self.free.popleft() for _ in range(count)]
        for block in blocks:
            self.refs[block] = 1
        return blocks

    def release(self, blocks):
        for block in reversed(blocks):
            self.refs[block] -= 1
            assert self.refs[block] >= 0
            if self.refs[block] == 0:
                self.free.append(block)


def blocks_needed(tokens, reserved=0):
    return -(-(tokens + reserved) // BLOCK_SIZE)


def resize(pool, table, tokens, reserved=0):
    need = blocks_needed(tokens, reserved)
    if need > len(table):
        table += pool.allocate(need - len(table))
    elif need < len(table):
        pool.release(table[need:])
        del table[need:]


def show(label, pool, tables):
    owned = " ".join(f"{name}={table}" for name, table in tables.items())
    print(f"{label:30s} {owned:28s} free={list(pool.free)}")
    used = [block for block in range(1, pool.num_blocks) if pool.refs[block]]
    assert sorted(used + list(pool.free)) == list(range(1, pool.num_blocks))


if __name__ == "__main__":
    pool = Pool(8)
    tables = {"A": [], "B": []}
    resize(pool, tables["A"], 6)
    resize(pool, tables["B"], 9)
    show("admit A(6), B(9)", pool, tables)
    resize(pool, tables["A"], 8)
    show("A reaches 8 tokens", pool, tables)
    resize(pool, tables["A"], 9)
    show("A reaches 9 tokens", pool, tables)
    resize(pool, tables["A"], 11, reserved=2)
    show("A at 11, two drafts reserved", pool, tables)
    resize(pool, tables["A"], 12)
    show("drafts rejected; A at 12", pool, tables)
    print("admission check for 4 blocks:", pool.check(4))
    pool.release(tables.pop("B"))
    show("B finishes", pool, {"A": tables["A"]})
    print("admission check for 4 blocks:", pool.check(4))
    print("admission check for 8 blocks:", pool.check(8))
    assert tables["A"] == [1, 2, 6] and list(pool.free) == [7, 5, 4, 3]
