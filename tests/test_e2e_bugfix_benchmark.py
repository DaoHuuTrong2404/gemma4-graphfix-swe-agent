"""End-to-end SWE bug-fixing benchmark verifying all 5 Gemma 4 agent stages locally."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gemma4_swe_agent.agent_orchestrator import AutonomousSWEAgent
from gemma4_swe_agent.diff_engine import search_replace_to_unified_diff
from gemma4_swe_agent.gemma4_runtime import DeterministicRuleRuntime, SynthesisProposal


class TestEndToEndSWEBenchmark(unittest.TestCase):
    def test_case1_mutable_default_and_zero_division_autonomous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "src_pkg").mkdir()
            (repo / "tests").mkdir()
            (repo / "src_pkg" / "__init__.py").write_text("", encoding="utf-8")
            (repo / "src_pkg" / "tags.py").write_text(
                """def attach_tags(item_name: str, tags: list[str] = []) -> list[str]:
    tags.append(item_name)
    return tags
""",
                encoding="utf-8",
            )
            (repo / "src_pkg" / "stats.py").write_text(
                """def mean_latency(samples: list[float]) -> float:
    return sum(samples) / len(samples)
""",
                encoding="utf-8",
            )
            (repo / "tests" / "test_tags.py").write_text(
                """import unittest
from src_pkg.tags import attach_tags

class TestTags(unittest.TestCase):
    def test_isolated_calls(self):
        self.assertEqual(attach_tags("a"), ["a"])
        self.assertEqual(attach_tags("b"), ["b"])
""",
                encoding="utf-8",
            )
            (repo / "tests" / "test_stats.py").write_text(
                """import unittest
from src_pkg.stats import mean_latency

class TestStats(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(mean_latency([]), 0.0)
    def test_non_empty(self):
        self.assertEqual(mean_latency([2.0, 4.0]), 3.0)
""",
                encoding="utf-8",
            )

            agent = AutonomousSWEAgent(runtime=DeterministicRuleRuntime())

            res_tags = agent.solve_issue(
                repo_root=repo,
                issue_text="attach_tags leaks previous item_name across calls when tags omitted",
                target_cmd=[sys.executable, "-m", "unittest", "tests/test_tags.py"],
                regression_cmd=[sys.executable, "-m", "unittest", "tests/test_tags.py"],
                apply_best=True,
            )
            self.assertTrue(res_tags.solved, res_tags.ranked_candidates)
            self.assertEqual(
                res_tags.telemetry["top_localized_file"], "src_pkg/tags.py"
            )

            res_stats = agent.solve_issue(
                repo_root=repo,
                issue_text="mean_latency raises ZeroDivisionError on empty samples",
                traceback_text='File "src_pkg/stats.py", line 2, in mean_latency\nZeroDivisionError: division by zero',
                target_cmd=[sys.executable, "-m", "unittest", "tests/test_stats.py"],
                regression_cmd=[
                    sys.executable,
                    "-m",
                    "unittest",
                    "tests/test_stats.py",
                ],
                apply_best=True,
            )
            self.assertTrue(res_stats.solved, res_stats.ranked_candidates)
            self.assertEqual(
                res_stats.telemetry["top_localized_file"], "src_pkg/stats.py"
            )

    def test_case2_self_refinement_and_regression_guard_selection(self) -> None:
        """Verify Round 0 blocks test-cheating and regressive patch, then Round 1 self-refinement succeeds."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "limiter.py").write_text(
                """def is_within_quota(used: int, limit: int) -> bool:
    if used < 0 or limit < 0:
        return False
    if used < limit:
        return True
    return False
""",
                encoding="utf-8",
            )
            (repo / "tests").mkdir()
            (repo / "tests" / "test_target.py").write_text(
                """import unittest
from limiter import is_within_quota

class TestTarget(unittest.TestCase):
    def test_exact_limit_allowed(self):
        self.assertTrue(is_within_quota(10, 10))
""",
                encoding="utf-8",
            )
            (repo / "tests" / "test_regression.py").write_text(
                """import unittest
from limiter import is_within_quota

class TestRegression(unittest.TestCase):
    def test_negative_rejected(self):
        self.assertFalse(is_within_quota(-1, 10))
    def test_over_limit_rejected(self):
        self.assertFalse(is_within_quota(11, 10))
""",
                encoding="utf-8",
            )

            # Candidate 1: cheats by changing test file (must be blocked)
            cheat_diff = search_replace_to_unified_diff(
                repo,
                [
                    (
                        "tests/test_target.py",
                        "self.assertTrue(is_within_quota(10, 10))",
                        "self.assertFalse(is_within_quota(10, 10))",
                    )
                ],
            )
            # Candidate 2: passes target test by always returning True (breaks regression test!)
            regressive_diff = search_replace_to_unified_diff(
                repo,
                [
                    (
                        "limiter.py",
                        "    if used < 0 or limit < 0:\n        return False",
                        "    return True",
                    )
                ],
            )
            # Candidate 3 (Round 1 after failure feedback): exact inclusive boundary fix
            clean_diff = search_replace_to_unified_diff(
                repo,
                [("limiter.py", "    if used < limit:", "    if used <= limit:")],
            )

            scripted = {
                0: [
                    SynthesisProposal("cheat_test_edit", cheat_diff, 0.7),
                    SynthesisProposal("overly_permissive", regressive_diff, 0.35),
                ],
                1: [
                    SynthesisProposal("refined_inclusive_bound", clean_diff, 0.0),
                ],
            }
            agent = AutonomousSWEAgent(
                runtime=DeterministicRuleRuntime(scripted), max_refine_rounds=2
            )
            res = agent.solve_issue(
                repo_root=repo,
                issue_text="is_within_quota rejects used == limit boundary",
                target_cmd=[sys.executable, "-m", "unittest", "tests/test_target.py"],
                regression_cmd=[
                    sys.executable,
                    "-m",
                    "unittest",
                    "tests/test_regression.py",
                ],
                apply_best=True,
            )
            self.assertTrue(res.solved)
            self.assertEqual(res.rounds_executed, 2)
            self.assertEqual(
                res.selected_evaluation.candidate.strategy, "refined_inclusive_bound"
            )
            self.assertTrue(res.selected_evaluation.policy_compliant)
            self.assertTrue(res.selected_evaluation.regression_report.passed)
            self.assertIn(
                "if used <= limit:", (repo / "limiter.py").read_text(encoding="utf-8")
            )


if __name__ == "__main__":
    unittest.main()
