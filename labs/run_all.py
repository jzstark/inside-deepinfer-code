"""Run the book's CPU labs and save actual output and execution provenance."""

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "01_generate.py", "02_reuse.py", "03_batch.py", "04_window.py",
    "05_cost.py", "06_trace.py", "07_attention.py", "08_kv_layout.py",
    "09_blocks.py", "10_share.py", "11_chunks.py", "12_preempt.py",
    "13_overlap.py", "14_lagging_metadata.py", "15_dispatch.py", "16_placement.py",
    "17_verification.py", "18_verification_budget.py",
    "19_replay.py", "20_replay_cost.py",
    "21_equality.py", "22_sampling_contracts.py",
    "23_service.py", "24_cache_hits.py", "25_frontier.py",
    "26_compression.py", "27_indexing.py", "28_ranks.py",
)
INPUTS = {
    "06_trace.py": ("labs/data/06_request_trace.json",),
    "16_placement.py": ("labs/vendor/eplb_policy.py",),
    "18_verification_budget.py": ("labs/data/18_dspark.csv",),
}
# Optional third-party packages; a lab is skipped, keeping its saved output, if absent.
REQUIRES = {"16_placement.py": ("torch",)}
ARTIFACTS = {
    "05_cost.py": ("results/05_cost_curve.csv",),
    "25_frontier.py": ("results/25_points.csv", "results/25_frontier.csv"),
    "13_overlap.py": ("results/13_timeline.tex",),
    "18_verification_budget.py": ("results/18_budget_curve.csv",),
}


def main():
    destination = ROOT / "results"
    destination.mkdir(exist_ok=True)
    provenance_path = destination / "provenance.json"
    previous = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    saved_runs = {run["script"]: run for run in previous.get("runs", ()) if "output_sha256" in run}
    execution = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    records = []
    skipped = []
    for name in SCRIPTS:
        script = ROOT / "labs" / name
        missing = [pkg for pkg in REQUIRES.get(name, ()) if importlib.util.find_spec(pkg) is None]
        if missing:
            script_path = str(script.relative_to(ROOT))
            skipped.append({"script": script_path, "reason": f"missing {', '.join(missing)}"})
            saved = saved_runs.get(script_path)
            if saved is not None:
                # Older records inherited their generating environment from the top level.
                saved.setdefault("execution", {key: previous[key] for key in execution})
                records.append(saved)
            status = "saved output and provenance kept" if saved is not None else "no saved run record"
            print(f"SKIP {name}: missing {', '.join(missing)}; {status}")
            continue
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        output = destination / f"{script.stem}.txt"
        output.write_text(result.stdout, encoding="utf-8")
        records.append({
            "script": str(script.relative_to(ROOT)),
            "sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "output": str(output.relative_to(ROOT)),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "exit_code": result.returncode,
            "stderr": result.stderr,
            "execution": execution,
        })
        for key, paths in (("inputs", INPUTS), ("artifacts", ARTIFACTS)):
            if name in paths:
                records[-1][key] = [
                    {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
                    for path in paths[name]
                ]
        if name in REQUIRES:
            records[-1]["dependencies"] = {pkg: importlib.metadata.version(pkg) for pkg in REQUIRES[name]}
        print(f"PASS {name} -> {output.relative_to(ROOT)}")
    provenance = {
        **execution,
        "kind": "teaching experiments; standard library unless dependencies are listed; not hardware benchmarks",
        "runs": records,
        "skipped": skipped,
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
