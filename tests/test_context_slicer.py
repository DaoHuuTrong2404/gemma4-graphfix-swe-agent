"""Unit tests for token-budgeted MinimalContextSlicer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gemma4_swe_agent.ast_localizer import ASTFaultLocalizer
from gemma4_swe_agent.context_slicer import MinimalContextSlicer


class TestMinimalContextSlicer(unittest.TestCase):
    def test_slice_respects_budget_and_includes_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "app").mkdir()
            (repo / "app" / "core.py").write_text(
                '''import math

def helper_scale(val: float) -> float:
    """Scale value cleanly."""
    return val * 2.0

class Processor:
    def compute(self, x: float) -> float:
        scaled = helper_scale(x)
        return math.floor(scaled)

    def unrelated_big(self) -> int:
        # filler lines
        total = 0
        for i in range(40):
            total += i
        return total
''',
                encoding="utf-8",
            )
            loc = ASTFaultLocalizer(repo)
            res = loc.localize("Processor.compute returns wrong floor value", top_k=5)
            slicer = MinimalContextSlicer(repo, loc)

            sl = slicer.build_slice(res, max_tokens=1200, max_focus_entities=3)
            self.assertIn("L0008:     def compute", sl.prompt_context)
            self.assertIn("class Processor:", sl.prompt_context)
            self.assertIn(
                "-> calls app/core.py:3 `def helper_scale(val) -> float:`",
                sl.prompt_context,
            )
            self.assertLessEqual(sl.estimated_tokens, 1200)

            tight = slicer.build_slice(res, max_tokens=120, max_focus_entities=3)
            self.assertTrue(tight.truncated or len(tight.included_entities) <= 1)


if __name__ == "__main__":
    unittest.main()
