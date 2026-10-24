clean = [ts - prev[0] for prev, (ts, flushed) in pairwise(stable_decode) if not flushed]
if not clean:
    return {}
avg_sps = len(clean) / sum(clean)
