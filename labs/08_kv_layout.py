"""KV capacity arithmetic and a scalar transcription of DeepInfer SWA indices.

The generic separate-K/V budget is hypothetical. The 512+16 byte format and
128/134 window/ring use the inspected format and default environment settings.
This lab does not import DeepInfer or execute its device kernels.
"""


def kv_bytes(layers, kv_heads, width, length, bytes_per_element):
    return 2 * layers * kv_heads * width * length * bytes_per_element


def swa_slots(position, state_id, window, ring):
    length = min(position + 1, window)
    start = position - length + 1
    offset = state_id * ring + start % ring
    slots = [offset // ring * ring + (offset + j) % ring
             if j < length else -1 for j in range(window)]
    write_slot = state_id * ring + position % ring
    return start, slots, write_slot


def main():
    full = kv_bytes(32, 8, 128, 4096, 2)
    windowed = kv_bytes(32, 8, 128, 128, 2)
    assert full == 536_870_912 and windowed == 16_777_216
    print(f"generic full KV: {full:,} B = {full // 2**20} MiB")
    print(f"generic window KV: {windowed:,} B = {windowed // 2**20} MiB")
    window, nextn, override = 128, 5, 0
    default_ring = window + max(override, nextn + 1)
    payload, scales = 512, 512 // 32
    token_bytes = payload + scales
    print(f"SWA row: {payload} data B + {scales} scale B = {token_bytes} B")
    print(f"128 visible rows, one layer: {128 * token_bytes:,} B")
    print(f"{default_ring} allocated rows, one layer: {default_ring * token_bytes:,} B")
    print("\nposition | logical range | first/last read slot | write slot")
    for position in (127, 128, 133, 134):
        start, slots, write = swa_slots(position, 2, window, default_ring)
        print(f"{position:8} | {start:3}..{position:<3}       | "
              f"{slots[0]:3} / {slots[-1]:3}            | {write}")
    # Check partial windows, eviction, wraparound and isolation between states.
    for window, ring in ((1, 1), (3, 5), (128, 134)):
        for state_id in (1, 2):
            for position in range(3 * ring):
                start, slots, write = swa_slots(position, state_id, window, ring)
                expected = [state_id * ring + p % ring
                            for p in range(start, position + 1)]
                assert slots == expected + [-1] * (window - len(expected))
                assert write == expected[-1]
                assert len(set(expected)) == len(expected)
                assert all(state_id * ring <= s < (state_id + 1) * ring
                           for s in expected)
    print("checks passed: partial windows, eviction, wraparound, state isolation")
    separate = kv_bytes(1, 8, 128, 1, 2)
    assert separate == 4096 and token_bytes == 528
    print(f"\nper layer/token: separate={separate:,} B, SWA={token_bytes} B")
    print(f"default ring: {window} + max({override}, {nextn} + 1) = {default_ring}")
    # Assigned six-query step: check logical row identities after all writes.
    position = 200
    for ring_size in (128, 133, default_ring):
        cache = {p % ring_size: p for p in range(position - ring_size, position)}
        for p in range(position, position + nextn + 1):
            cache[p % ring_size] = p
        damaged = [sum(cache[p % ring_size] != p
                       for p in range(q - window + 1, q + 1))
                   for q in range(position, position + nextn + 1)]
        assert damaged == ([5, 4, 3, 2, 1, 0] if ring_size == 128 else [0] * 6)
        print(f"S={ring_size}: overwritten visible rows per query = {damaged}")


if __name__ == "__main__":
    main()
