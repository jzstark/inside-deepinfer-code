"""Check local DeepInfer source hashes and reproduce every printed excerpt."""

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_sources(source_root):
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    sources = {}
    for record in manifest["files"]:
        path = record["path"]
        data = (source_root / path).read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"source hash mismatch: {path}")
        sources[path] = data.decode("utf-8")

    for record in manifest["excerpts"]:
        path = record["path"]
        lines = sources[record["source"]].splitlines(keepends=True)
        first, last = record["first_line"], record["last_line"]
        if not 1 <= first <= last <= len(lines):
            raise ValueError(f"invalid source range: {path}: {first}--{last}")
        selected = "".join(lines[first - 1:last])
        expected = textwrap.dedent(selected)
        original_line = next(line for line in selected.splitlines() if line.strip())
        dedented_line = next(line for line in expected.splitlines() if line.strip())
        if len(original_line) - len(dedented_line) != record["removed_leading_spaces"]:
            raise ValueError(f"indentation mismatch: {path}")
        actual = (ROOT / path).read_bytes()
        if actual != expected.encode("utf-8"):
            raise ValueError(f"excerpt differs from source range: {path}")
        if hashlib.sha256(actual).hexdigest() != record["sha256"]:
            raise ValueError(f"excerpt hash mismatch: {path}")
        print(f"PASS {path}")

    print(f"PASS {len(sources)} source hashes and {len(manifest['excerpts'])} excerpts"
          f" for {manifest['revision']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="local DeepInfer source directory")
    args = parser.parse_args()
    try:
        check_sources(args.source_root)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        parser.exit(1, f"FAIL: {error}\n")
