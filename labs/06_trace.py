"""Analyse assigned request timestamps, not a measured engine trace.

Run: python3 labs/06_trace.py
All events use one clock. Token timestamps mean delivery at the observer.
Try delaying an arrival or a token; preserve the event order in each request.
"""

import json
from math import ceil
from pathlib import Path
from statistics import mean


def analyse(request):
    arrival, start, end = (request[key] for key in ("arrival", "prompt_start", "prompt_end"))
    tokens = request["tokens"]
    assert tokens and arrival <= start <= end <= tokens[0]
    assert tokens == sorted(tokens)
    queue, prompt, publish = start - arrival, end - start, tokens[0] - end
    ttft = tokens[0] - arrival
    gaps = [later - earlier for earlier, later in zip(tokens, tokens[1:])]
    assert ttft == queue + prompt + publish
    return queue, prompt, publish, ttft, gaps


def percentile(values, fraction):
    """Nearest-rank percentile of a non-empty sample, for 0 < fraction <= 1."""
    return sorted(values)[ceil(fraction * len(values)) - 1]


if __name__ == "__main__":
    source = Path(__file__).with_name("data") / "06_request_trace.json"
    trace = json.loads(source.read_text())
    assert trace["unit"] == "ms"
    requests = trace["requests"]
    ttfts, all_gaps = [], []
    print(trace["kind"])
    print("All times in ms; gaps exclude the wait for the first token")
    print("id  queue  prompt  publish  TTFT  gaps")
    for request in requests:
        queue, prompt, publish, ttft, gaps = analyse(request)
        ttfts.append(ttft)
        all_gaps.extend(gaps)
        gap_text = ",".join(map(str, gaps)) if gaps else "-"
        print(f"{request['id']:2s} {queue:6d} {prompt:7d} {publish:8d} {ttft:5d}  {gap_text}")
    print(f"TTFT: n={len(ttfts)}, mean={mean(ttfts):.1f},"
          f" p50={percentile(ttfts, 0.50)}, p95={percentile(ttfts, 0.95)}")
    print(f"Token gaps: n={len(all_gaps)}, mean={mean(all_gaps):.1f},"
          f" p95={percentile(all_gaps, 0.95)}")
    elapsed = max(r["tokens"][-1] for r in requests) - min(r["arrival"] for r in requests)
    total_tokens = sum(len(r["tokens"]) for r in requests)
    print(f"Whole trace: {total_tokens} tokens / {elapsed} ms"
          f" = {total_tokens / (elapsed / 1000):.1f} tokens/s")
    assert ttfts == [10, 10, 19, 15, 42]
    assert len(all_gaps) == 8 and mean(all_gaps) == 5.5
    assert total_tokens == 13 and elapsed == 65
