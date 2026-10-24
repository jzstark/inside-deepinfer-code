times_ms.sort()
trimmed = times_ms[NUM_TRIM : NUM_MEASURE - NUM_TRIM]
avg_time_ms = sum(trimmed) / len(trimmed)
sum_time = torch.full((1,), avg_time_ms, dtype=torch.float32, device="cuda")
dist.all_reduce(sum_time, op=dist.ReduceOp.SUM, group=group)
avg_time_ms = sum_time.item() / group.size()
results.append((batch_size, avg_time_ms))
