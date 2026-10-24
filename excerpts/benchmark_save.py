if args.result_jsonl is not None:
    result = benchmark_config | prefill_metrics | decode_metrics
    args.result_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.result_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, default=str) + "\n")
    print(f"Benchmark result appended to {args.result_jsonl.resolve()}")
