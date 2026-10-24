"""Indexer scoring, candidate restriction and address spaces.

Run: python3 labs/27_indexing.py. Integer dot products implement the mathematical
sum_h weight_h * relu(dot(q_h, k)); projections, RoPE, quantisation and GPU kernels
are omitted. Ties break by index in this lab, not a claimed production tie rule.
The block table uses eight original tokens per block and C2, hence four rows.
"""

from itertools import combinations

Q = ((1, 0), (0, 1))
WEIGHTS = (1, 2)
KEYS = ((2, 0), (0, 3), (1, 1), (-2, -1), (100, 100))


def score(key):
    return sum(w * max(0, sum(a * b for a, b in zip(q, key)))
               for w, q in zip(WEIGHTS, Q))


def select(scores, candidates, k):
    chosen = sorted(candidates, key=lambda j: (-scores[j], j))[:k]
    return sorted(chosen) + [-1] * (k - len(chosen))


def physical(index, blocks, rows_per_block):
    if index == -1:
        return -1
    return blocks[index // rows_per_block] * rows_per_block + index % rows_per_block


def main():
    scores = list(map(score, KEYS))
    valid = 4  # four completed C2 rows; the fifth key is future/dirty storage
    dense = select(scores, range(valid), 2)
    assert scores == [2, 6, 3, 0, 300] and dense == [1, 2]
    assert 4 in select(scores, range(5), 2)  # bounds must precede selection
    block_scores = [max(scores[i:i + 2]) for i in range(0, valid, 2)]
    candidate_block = max(range(len(block_scores)), key=block_scores.__getitem__)
    candidates = range(candidate_block * 2, candidate_block * 2 + 2)
    restricted = select(scores, candidates, 2)
    assert restricted == [0, 1] and restricted != dense
    assert sum(scores[i] for i in dense) == max(
        sum(scores[i] for i in pair) for pair in combinations(range(valid), 2))
    assert select(scores, range(0), 2) == [-1, -1]
    assert select(scores, range(1), 3) == [0, -1, -1]
    blocks = [7, 2]
    logical = [1, 2, 4, -1]
    slots = [physical(i, blocks, 4) for i in logical]
    assert slots == [29, 30, 8, -1]
    for i, slot in zip(logical, slots):
        if i >= 0:
            assert slot // 4 == blocks[i // 4] and slot % 4 == i % 4
    print("Scores:", scores, "| valid prefix:", valid)
    print("Top-2 over valid rows:", dense)
    print("Candidate block scores:", block_scores, "| retained block:", candidate_block)
    print("Top-2 within retained block:", restricted)
    print("One valid row, three output slots:", select(scores, range(1), 3))
    print("Prefill at gathered offset 10:", [10 + i for i in dense])
    print("Decode logical rows:", logical, "-> physical slots:", slots)
    print("Payload bytes/row: main FP8=528; main FP4=288; index FP4=68")
    assert 512 + 512 // 32 == 528
    assert 512 // 2 + 512 // 16 == 288
    assert 128 // 2 + 128 // 32 == 68
    print("Checks: bounds, candidate counterexample, sentinel and address mapping passed")


if __name__ == "__main__":
    main()
