"""Unit tests for diff parsing, AST syntax guard, Gemma 4 thought stripping, and deterministic patch ranking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gemma4_swe_agent.diff_engine import (
    apply_unified_diff,
    parse_search_replace_blocks,
    search_replace_to_unified_diff,
)
from gemma4_swe_agent.gemma4_runtime import extract_diff_payload, strip_gemma4_thoughts
from gemma4_swe_agent.patch_pipeline import (
    CandidatePatch,
    DeterministicPatchRanker,
    TestExecutionReport,
)


class TestDiffEngineAndRanking(unittest.TestCase):
    def test_gemma4_thought_stripping_and_search_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "calc.py").write_text(
                "def add(a, b):\n    return a - b\n", encoding="utf-8"
            )
            raw_model_out = (
                "<|channel>thought\nNeed to change minus to plus.<channel|>\n"
                "```text\n"
                "FILE: calc.py\n"
                "<<<SEARCH\n"
                "    return a - b\n"
                "===\n"
                "    return a + b\n"
                ">>>REPLACE\n"
                "```\n"
            )
            cleaned, thoughts = strip_gemma4_thoughts(raw_model_out)
            self.assertIn("Need to change minus to plus", thoughts)
            self.assertNotIn("<|channel>thought", cleaned)
            edits = parse_search_replace_blocks(cleaned)
            self.assertEqual(len(edits), 1)
            udiff = search_replace_to_unified_diff(repo, edits)
            res = apply_unified_diff(repo, udiff, write_to_disk=True)
            self.assertTrue(res.success)
            self.assertIn(
                "return a + b", (repo / "calc.py").read_text(encoding="utf-8")
            )
            self.assertIn("--- a/calc.py", extract_diff_payload(udiff))

    def test_syntax_guard_test_policy_and_line_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "mod.py").write_text(
                "# header 1\n# header 2\ndef safe_div(a, b):\n    return a / b\n",
                encoding="utf-8",
            )
            (repo / "tests").mkdir()
            (repo / "tests" / "test_mod.py").write_text(
                "assert True\n", encoding="utf-8"
            )

            # 1. Line drift (says line 1 instead of line 3) -> deterministic recovery
            drifted = (
                "--- a/mod.py\n"
                "+++ b/mod.py\n"
                "@@ -1,2 +1,4 @@\n"
                " def safe_div(a, b):\n"
                "+    if b == 0:\n"
                "+        return 0\n"
                "     return a / b\n"
            )
            res_drift = apply_unified_diff(repo, drifted, write_to_disk=False)
            self.assertTrue(res_drift.success, res_drift.errors)
            self.assertEqual(res_drift.blast_radius.lines_added, 2)

            # 2. Invalid Python syntax -> blocked
            bad_syntax = (
                "--- a/mod.py\n"
                "+++ b/mod.py\n"
                "@@ -3,2 +3,2 @@\n"
                " def safe_div(a, b):\n"
                "-    return a / b\n"
                "+    return a / (\n"
            )
            res_syn = apply_unified_diff(repo, bad_syntax, write_to_disk=False)
            self.assertFalse(res_syn.success)
            self.assertFalse(res_syn.syntax_valid)

            # 3. Editing test files -> blocked by policy
            test_cheat = (
                "--- a/tests/test_mod.py\n"
                "+++ b/tests/test_mod.py\n"
                "@@ -1 +1 @@\n"
                "-assert True\n"
                "+assert 1 == 1\n"
            )
            res_cheat = apply_unified_diff(repo, test_cheat, write_to_disk=False)
            self.assertFalse(res_cheat.success)
            self.assertTrue(res_cheat.blast_radius.touches_tests)

            # 4. Path traversal -> blocked
            traversal = (
                "--- a/../../evil.py\n"
                "+++ b/../../evil.py\n"
                "@@ -1 +1 @@\n"
                "-x = 1\n"
                "+x = 2\n"
            )
            res_trav = apply_unified_diff(repo, traversal, write_to_disk=False)
            self.assertFalse(res_trav.success)

    def test_deterministic_patch_ranking_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "m.py").write_text("def f(x):\n    return x\n", encoding="utf-8")

            small_clean = apply_unified_diff(
                repo,
                "--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,2 @@\n def f(x):\n-    return x\n+    return x + 1\n",
            )
            large_clean = apply_unified_diff(
                repo,
                "--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,4 @@\n def f(x):\n-    return x\n+    y = x\n+    z = y + 1\n+    return z\n",
            )

            pass_rep = TestExecutionReport(True, 0, 5, 0, "ok", "", 0.05)
            fail_rep = TestExecutionReport(
                False, 1, 5, 1, "fail", "", 0.05, "AssertionError"
            )

            ev_reg_broken = DeterministicPatchRanker.score_evaluation(
                CandidatePatch("c_reg", "s1", small_clean.canonical_diff),
                small_clean,
                pass_rep,
                fail_rep,
                policy_compliant=True,
                localization_aligned=True,
            )
            ev_large_clean = DeterministicPatchRanker.score_evaluation(
                CandidatePatch("c_large", "s2", large_clean.canonical_diff),
                large_clean,
                pass_rep,
                pass_rep,
                policy_compliant=True,
                localization_aligned=True,
            )
            ev_small_clean = DeterministicPatchRanker.score_evaluation(
                CandidatePatch("c_small", "s3", small_clean.canonical_diff),
                small_clean,
                pass_rep,
                pass_rep,
                policy_compliant=True,
                localization_aligned=True,
            )

            ranked = DeterministicPatchRanker.rank(
                [ev_reg_broken, ev_large_clean, ev_small_clean]
            )
            self.assertEqual(
                [r.candidate.candidate_id for r in ranked],
                ["c_small", "c_large", "c_reg"],
            )


if __name__ == "__main__":
    unittest.main()
