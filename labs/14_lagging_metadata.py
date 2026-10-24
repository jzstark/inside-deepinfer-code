"""Correct one-step-old decode metadata with DeepInfer's matching and formula.

Run: python3 labs/14_lagging_metadata.py
A list translation of prepare_zos_inputs (csrc/engine/zos.cpp:142-196), the
token gather (gather_input_ids_zos.py:21-29) and compute_seqlens_k_zos_ref
(compute_seqlens_k_zos.py:35-52). No kernel runs; token IDs are invented.
"""

from bisect import bisect_left
from dataclasses import dataclass

EOS = 2


@dataclass
class Prev:  # device outputs of step N, one row of batch N
    seq_id: int
    sampled: int  # token sampled after the accepted drafts
    drafts: tuple  # draft tokens proposed for step N+1
    accepted: int  # drafts accepted in step N, 0..nextn


@dataclass
class Row:  # host metadata while scheduling step N+1
    seq_id: int
    processed: int  # committed cache.num_tokens; step N is not included
    inflight: int  # positions reserved for step N; 0 if absent from batch N
    q: int
    fresh: tuple  # host-known inputs, used only for unmatched rows


def graph_pad(n_seq, total_q, max_size, interval=8):
    sizes, size = [], 1  # build_graph_batch_sizes, cuda_graph_runner.py:41-51
    while size < min(interval, max_size):
        sizes.append(size)
        size *= 2
    sizes += list(range(min(interval, max_size), max_size, interval)) + [max_size]
    return n_seq + sizes[bisect_left(sizes, total_q)] - total_q


def prepare(rows, prev, n_pad, nextn):
    remap = [-1] * len(rows)
    same = len(rows) == len(prev)
    if same:
        for i, (row, old) in enumerate(zip(rows, prev)):
            if row.seq_id != old.seq_id:
                same = False
                break
            remap[i] = i
    overlap = same and bool(rows)
    if not same:
        where = {old.seq_id: j for j, old in enumerate(prev)}
        remap = [where.get(row.seq_id, -1) for row in rows]
        overlap = any(r >= 0 for r in remap)
    if not overlap:
        return None
    pad = n_pad - len(rows)
    k_cpu = [r.processed + r.inflight + r.q for r in rows]  # scheduler.hpp:697
    k = [kc - (r.q if nextn else 0) for r, kc in zip(rows, k_cpu)]  # zos.cpp:188
    return remap + [-1] * pad, [r.q for r in rows] + [1] * pad, k + [1] * pad, k_cpu, same


def correct(rows, prev, remap, q, k, nextn):
    inputs = []
    for i, row in enumerate(rows):
        old = prev[remap[i]] if remap[i] >= 0 else None
        inputs.append([row.fresh[j] if old is None else
                       old.sampled if j == 0 else old.drafts[j - 1] for j in range(q[i])])
        if nextn:  # worker.py:220-227; padded rows are left unchanged
            k[i] += q[i] - (nextn - old.accepted if old else 0)
    return inputs


def truth(row, prev):
    old = next((p for p in prev if p.seq_id == row.seq_id), None)
    if old is None:
        return list(row.fresh[:row.q]), row.processed + row.q
    done = row.processed + 1 + old.accepted
    return [old.sampled, *old.drafts][:row.q], done + row.q


def scenario(title, rows, prev, nextn, max_seqs=8):
    total_q = sum(r.q for r in rows)
    n_pad = graph_pad(len(rows), total_q, max_seqs * (1 + nextn))
    prepared = prepare(rows, prev, n_pad, nextn)
    assert prepared is not None  # each scenario overlaps the previous batch
    remap, q, k, k_cpu, same = prepared
    base = list(k)
    inputs = correct(rows, prev, remap, q, k, nextn)
    print(f"{title}: nextn={nextn}, total_q={total_q}, rows with padding={n_pad},",
          "same-order path" if same else "seq_id map")
    print("  id remap q k_cpu base corrected true  inputs")
    for i, row in enumerate(rows):
        want_inputs, want_k = truth(row, prev)
        assert inputs[i] == want_inputs and k[i] == want_k
        print(f"  {row.seq_id:2d} {remap[i]:5d} {q[i]} {k_cpu[i]:5d} {base[i]:4d}"
              f" {k[i]:9d} {want_k:4d}  {inputs[i]}")
    assert remap[len(rows):] == [-1] * (n_pad - len(rows))
    assert q[len(rows):] == k[len(rows):] == [1] * (n_pad - len(rows))
    finished = [p.seq_id for p in prev if p.sampled == EOS and p.seq_id in
                [r.seq_id for r in rows]]
    if finished:
        print(f"  evaluated although step N sampled EOS: {finished};",
              "their step N+1 output is not published")
    return remap


if __name__ == "__main__":
    prev0 = [Prev(11, 501, (), 0), Prev(12, 502, (), 0), Prev(13, 503, (), 0), Prev(14, 504, (), 0)]
    rows0 = [Row(s, 20 + s, 1, 1, ()) for s in (11, 12, 13, 14)]
    scenario("A. missing token, same order", rows0, prev0, 0)
    print()
    rows1 = [Row(13, 33, 1, 1, ()), Row(11, 31, 1, 1, ()), Row(15, 40, 0, 1, (515,)), Row(12, 32, 1, 1, ())]
    assert scenario("B. reordered survivors and a new request", rows1, prev0, 0) == [2, 0, -1, 1]
    print()
    prev2 = [Prev(11, 601, (611, 612), 0), Prev(12, EOS, (621, 622), 1),
             Prev(13, 603, (631, 632), 2), Prev(14, 604, (641, 642), 2)]
    rows2 = [Row(s, 20 + s, 3, 3, ()) for s in (11, 12, 13, 14)]
    scenario("C. missing accepted length, EOS and graph padding", rows2, prev2, 2)
