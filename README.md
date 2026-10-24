# Inside DeepInfer: companion code

Experiments and source checks for the book *Inside DeepInfer: How an LLM
Inference Engine Works* by Jianxin Zhao.

This is an independent project. It is not affiliated with, sponsored by or
endorsed by DeepSeek. The book describes DeepInfer at commit
[`774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01).

## Run the experiments

The experiments need Python 3.10 or later. All but lab 16 use only the standard
library; lab 16 also needs PyTorch, and a CPU is enough.

```bash
python3 labs/01_generate.py      # run one experiment
make labs                        # run all, rewriting results/ and provenance
make test                        # check the experiment runner
```

Each script's docstring states its assumptions. The experiments are teaching
models: none of them runs the DeepInfer engine. Lab 16 imports one unmodified
DeepInfer file, the EPLB placement policy, and runs it on CPU tensors. Assigned
costs and simulated clocks expose dependencies and trade-offs; they do not
measure accelerator performance.

`results/` holds the outputs printed in the book. `results/provenance.json`
records each script's hash, its output hash, the Python version and platform.
If PyTorch is missing, `make labs` skips lab 16 and keeps its saved output.

## Check a DeepInfer checkout against the book

```bash
git clone https://github.com/deepseek-ai/DeepInfer.git
git -C DeepInfer checkout 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01
make check DEEPINFER_SOURCE=DeepInfer
```

`make check` compares the hashes of every inspected DeepInfer file with
`source-manifest.json` and reproduces each printed excerpt in `excerpts/` from
its recorded line range. A mismatch shows that a checkout differs from the
book's edition; it does not say which explanations still hold.

## Experiments by chapter

| Lab | Chapter | What it shows |
|---|---|---|
| `01_generate.py` | 1 | Generate tokens with a fixed score table |
| `02_reuse.py` | 1 | Reuse a prefix sum instead of recomputing the history |
| `03_batch.py` | 1 | Fixed batches versus refilling empty slots |
| `04_window.py` | 1 | Rebuild a three-integer window from the processed suffix |
| `05_cost.py` | 2 | A hypothetical batch-cost model |
| `06_trace.py` | 2 | Latency boundaries in an assigned request trace |
| `07_attention.py` | 3 | One causal attention layer, cached and recomputed |
| `08_kv_layout.py` | 3 | KV capacity and DeepInfer's SWA ring indices |
| `09_blocks.py` | 4 | Fixed-size blocks from one finite pool |
| `10_share.py` | 4 | Shared prefix blocks, a copied tail and a safe release |
| `11_chunks.py` | 5 | Chunk size versus the choice of the next request |
| `12_preempt.py` | 5 | A memory-pressure decision and three preemption outcomes |
| `13_overlap.py` | 6 | Synchronous and ZOS host/device timelines |
| `14_lagging_metadata.py` | 6 | Correcting one-step-old decode metadata |
| `15_dispatch.py` | 7 | Expert dispatch, computation and combine |
| `16_placement.py` | 7 | DeepInfer's expert-placement policy on small histories |
| `17_verification.py` | 8 | Standard and penalised speculative sampling, exactly |
| `18_verification_budget.py` | 8 | The two-queue verification budget on synthetic costs |
| `19_replay.py` | 9 | Bounded SWA replay with shared and per-layer inputs |
| `20_replay_cost.py` | 9 | Query rows and SWA pairs after truncation |
| `21_equality.py` | 10 | Reduction order and numerical checks |
| `22_sampling_contracts.py` | 10 | Random keys, verification budgets and PIT masses |
| `23_service.py` | 11 | A prefill/decode queue with shared and local caches |
| `24_cache_hits.py` | 11 | The documented prefix-hit policy |
| `25_frontier.py` | 11 | A throughput frontier and an acceptance-rate audit |
| `26_compression.py` | A | C2 boundaries and rollback state |
| `27_indexing.py` | A | Index scoring, candidate restriction and addresses |
| `28_ranks.py` | B | DP, CP and EP rank grouping |

## Layout

- `labs/`: the experiments, `run_all.py`, input data and the vendored
  DeepInfer policy file used by lab 16.
- `results/`: saved outputs and run provenance.
- `excerpts/`: the DeepInfer source ranges printed in the book.
- `source-manifest.json`: pinned source edition, file hashes and excerpt ranges.
- `tools/check_sources.py`: the checker behind `make check`.
- `tests/`: tests for the experiment runner.

## Licence

The code in this repository is distributed under the Apache License 2.0; see
`LICENSE`. `excerpts/` and `labs/vendor/eplb_policy.py` are taken from DeepInfer,
which is also distributed under the Apache License 2.0; see
`excerpts/NOTICE.txt` and `labs/vendor/NOTICE.txt` for their origin and
notices. The book's text is not included here and remains © 2026 Jianxin Zhao.
