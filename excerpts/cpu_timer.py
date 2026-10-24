t0 = time.perf_counter()
with torch.profiler.record_function(f"deepinfer:{record_name or name}"):
    yield
elapsed = time.perf_counter() - t0
self.add_time(name, elapsed)
