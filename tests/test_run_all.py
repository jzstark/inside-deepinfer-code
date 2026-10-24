"""Check that optional lab skips preserve the evidence for saved results."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from labs import run_all


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "labs").mkdir()
        (self.root / "labs/example.py").write_text('print("saved result")\n')
        (self.root / "labs/input.txt").write_text("input evidence\n")
        overrides = patch.multiple(
            run_all, ROOT=self.root, SCRIPTS=("example.py",),
            REQUIRES={"example.py": ("optional_package",)},
            INPUTS={"example.py": ("labs/input.txt",)}, ARTIFACTS={},
        )
        overrides.start()
        self.addCleanup(overrides.stop)

    def run_labs(self, available):
        with patch.object(run_all.importlib.util, "find_spec", return_value=object() if available else None), \
                patch.object(run_all.importlib.metadata, "version", return_value="1.2.3"), \
                contextlib.redirect_stdout(io.StringIO()):
            run_all.main()
        return json.loads((self.root / "results/provenance.json").read_text())

    def test_repeated_skips_preserve_legacy_record_and_original_environment(self):
        original = self.run_labs(available=True)
        run = original["runs"][0]
        del run["execution"]  # the previous schema stored the environment only at the top
        environment = {"executed_at_utc": "2000-01-01T00:00:00+00:00",
                       "python": "original Python", "platform": "original platform"}
        original.update(environment)
        (self.root / "results/provenance.json").write_text(json.dumps(original))
        output = (self.root / run["output"]).read_bytes()
        for _ in range(2):
            with patch.object(run_all.subprocess, "run") as subprocess_run:
                skipped = self.run_labs(available=False)
                subprocess_run.assert_not_called()
            self.assertEqual(skipped["runs"], [{**run, "execution": environment}])
            self.assertEqual(skipped["skipped"], [
                {"script": "labs/example.py", "reason": "missing optional_package"}])
            self.assertEqual((self.root / run["output"]).read_bytes(), output)
        rerun = self.run_labs(available=True)
        self.assertEqual(rerun["skipped"], [])
        self.assertEqual(rerun["runs"][0]["execution"]["python"], rerun["python"])
        self.assertNotEqual(rerun["runs"][0]["execution"], environment)

    def test_skip_without_a_previous_run_does_not_invent_evidence(self):
        skipped = self.run_labs(available=False)
        self.assertEqual(skipped["runs"], [])
        self.assertEqual(len(skipped["skipped"]), 1)
        self.assertFalse((self.root / "results/example.txt").exists())


if __name__ == "__main__":
    unittest.main()
