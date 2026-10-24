# DP group
dp_ranks = rearrange(ranks, "(replica dp cp) -> (replica cp) dp", dp=dp_size, cp=cp_size)
self.dp = _split_group(dp_ranks.tolist(), backend=backend, pg_options=options)
self.cpu_dp_group = _split_group(dp_ranks.tolist(), backend="gloo")

# CP group
cp_ranks = rearrange(ranks, "(dp cp) -> dp cp", cp=cp_size)
self.cp = _split_group(cp_ranks.tolist(), backend=backend, pg_options=options)
self.cpu_cp_group = _split_group(cp_ranks.tolist(), backend="gloo")

# EP groups: contiguous groups of ep_size ranks.
if ep_size == ranks.numel():
    self.ep = self.world
else:
    ep_ranks = rearrange(ranks, "(edp ep) -> edp ep", ep=ep_size)
    self.ep = _split_group(ep_ranks.tolist(), backend=backend, pg_options=options)
