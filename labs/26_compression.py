"""C2 boundaries and rollback state, using exact rational arithmetic.

Run: python3 labs/26_compression.py. Positive integer weights stand for exp(score)
per channel. The weighted mean translates compress.py's softmax-over-positions
reference formula before BF16 conversion, norm and RoPE. Chunk and ring checks
model its boundaries, not a kernel launch or full-model equivalence.
"""

from fractions import Fraction as F
from itertools import product

# Two channels; different gates make this more than an ordinary token average.
ROWS = [((1, 3), (0, 10)), ((3, 1), (8, 2)),
        ((2, 1), (4, 6)), ((2, 3), (12, 2)),
        ((1, 1), (1, 3)), ((1, 1), (5, 7)), ((2, 2), (9, 11))]


def compress(rows):
    return tuple(sum(F(w[d]) * v[d] for w, v in rows)
                 / sum(w[d] for w, _ in rows) for d in range(2))


def chunked(cuts):
    pending, output, counts = [], [], []
    start = 0
    for end in cuts:
        before = len(output)
        for row in ROWS[start:end]:
            pending.append(row)
            if len(pending) == 2:
                output.append(compress(pending))
                pending.clear()
        count = end // 2 - start // 2
        assert len(output) - before == count
        counts.append((count, max(1, count)))  # valid rows / reserved rows
        start = end
    return output, pending, counts


def decode_ring(size):
    # Position 0 came from an earlier step. Write all six new positions first.
    ring = [None] * size
    ring[0] = ROWS[0]
    for position in range(1, 7):
        ring[position % size] = ROWS[position]
    return [compress([ring[(p - 1) % size], ring[p % size]])
            for p in (1, 3, 5)]


def main():
    expected = [compress(ROWS[i:i + 2]) for i in (0, 2, 4)]
    checked = 0
    for splits in product((False, True), repeat=6):
        cuts = [i + 1 for i, split in enumerate(splits) if split] + [7]
        result, pending, _ = chunked(cuts)
        assert result == expected and pending == ROWS[-1:]
        checked += 1
    result, pending, counts = chunked([1, 4, 7])
    print("C2 rows (before BF16/norm/RoPE):", [tuple(map(str, r)) for r in result])
    print("Chunks [0,1), [1,4), [4,7): valid/reserved rows", counts)
    print("Pending token positions:", [len(ROWS) - len(pending)])
    assert decode_ring(7) == expected
    wrong = decode_ring(6)
    assert wrong != expected
    print("Six-position decode: ring 7 preserves every completed pair")
    print("Ring 6 corrupts first pair:", tuple(map(str, wrong[0])))
    print(f"Checks: {checked} chunk partitions agree exactly; undersized ring fails")
    print("Default C2 state payload: 7 * 2 * 512 * 4 =", 7 * 2 * 512 * 4, "bytes")


if __name__ == "__main__":
    main()
