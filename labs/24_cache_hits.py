"""Translate the Q&A's hit policy, not the unreleased cache driver.

Run: python3 labs/24_cache_hits.py. The cache contains a contiguous prefix of
cached tokens. A boolean represents qualifying SWA availability. The policy
caps block reuse, aligns it, then backs off one block if SWA is missing.
Monotonicity is checked with cache extent and SWA availability held fixed.
This does not resolve the 134/140 integration gap described in chapter 9.
"""

BLOCK = 128
RESERVE = 134


def hit_lengths(length, cached, has_swa):
    assert length >= 0 and cached >= 0
    cap = max(0, length - RESERVE)
    block_hit = min(cap, cached) // BLOCK * BLOCK
    compute_hit = block_hit if has_swa else max(0, block_hit - BLOCK)
    return block_hit, compute_hit


def check():
    cases = 0
    for cached in range(0, 2049, BLOCK):
        for has_swa in (False, True):
            previous = (0, 0)
            for length in range(2049):
                block_hit, compute_hit = hit_lengths(length, cached, has_swa)
                valid = [n for n in range(0, min(length, cached) + 1, BLOCK)
                         if n == 0 or n <= length - RESERVE]
                assert block_hit == max(valid)
                assert 0 <= compute_hit <= block_hit <= max(0, length - RESERVE)
                assert block_hit % BLOCK == compute_hit % BLOCK == 0
                assert block_hit - compute_hit == (0 if has_swa or not block_hit else BLOCK)
                assert block_hit >= previous[0] and compute_hit >= previous[1]
                if length < RESERVE + BLOCK:
                    assert block_hit == compute_hit == 0
                previous = block_hit, compute_hit
                cases += 1
    return cases


def main():
    print("Assigned full block-prefix availability; columns count tokens")
    print("L     block hit L1   compute hit: SWA present / missing")
    for length in (128, 261, 262, 390, 902, 1024):
        block_hit, present = hit_lengths(length, length, True)
        _, missing = hit_lengths(length, length, False)
        print(f"{length:<5} {block_hit:<14} {present:>4} / {missing}")
    print(f"Checks: {check()} cases; bounds, alignment, backoff, monotonicity")


if __name__ == "__main__":
    main()
