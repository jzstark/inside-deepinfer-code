xs = [p["decode_avg_sps"] * tau for p in points]
ys = [p["decode_avg_sps"] * p["decode_num_requests"] * tau / p["cp"] for p in points]
return xs, ys, [f"{p['decode_num_requests']:,}" for p in points]
