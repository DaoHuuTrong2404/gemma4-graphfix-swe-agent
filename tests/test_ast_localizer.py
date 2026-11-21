"""Unit tests for ASTFaultLocalizer and call-graph fault propagation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gemma4_swe_agent.ast_localizer import ASTFaultLocalizer, split_identifier_tokens


def _build_sample_repo(root: Path) -> None:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "rates.py").write_text(
        '''def compute_tier_rate(amount: float, vip_tier: int) -> float:
    """Calculate discount rate for VIP tier."""
    if vip_tier >= 3:
        return amount * 0.85
    return amount
''',
        encoding="utf-8",
    )
    (root / "pkg" / "billing.py").write_text(
        '''from pkg.rates import compute_tier_rate

class InvoiceEngine:
    """Core invoice pricing engine."""

    def apply_invoice_total(self, subtotal: float, vip_tier: int) -> float:
        discounted = compute_tier_rate(subtotal, vip_tier)
        return round(discounted, 2)

    def format_receipt(self, total: float) -> str:
        return f"TOTAL={total:.2f}"
''',
        encoding="utf-8",
    )
    (root / "tests" / "test_billing.py").write_text(
        """import unittest
from pkg.billing import InvoiceEngine

class TestBilling(unittest.TestCase):
    def test_vip_discount(self):
        eng = InvoiceEngine()
        self.assertEqual(eng.apply_invoice_total(100.0, 3), 85.0)
""",
        encoding="utf-8",
    )


class TestASTFaultLocalizer(unittest.TestCase):
    def test_identifier_splitting(self) -> None:
        toks = split_identifier_tokens("apply_invoice_total InvoiceEngine HTTPStatus")
        self.assertIn("apply", toks)
        self.assertIn("invoice", toks)
        self.assertIn("total", toks)
        self.assertIn("engine", toks)
        self.assertIn("http", toks)

    def test_call_graph_propagation_and_test_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _build_sample_repo(repo)
            loc = ASTFaultLocalizer(repo)
            res = loc.localize(
                issue_text="InvoiceEngine.apply_invoice_total returns wrong VIP tier discount",
                traceback_text=(
                    "Traceback (most recent call last):\n"
                    '  File "tests/test_billing.py", line 7, in test_vip_discount\n'
                    '  File "pkg/billing.py", line 7, in apply_invoice_total\n'
                    "AssertionError: 80.0 != 85.0"
                ),
                top_k=6,
            )
            ranked_names = [item.entity.name for item in res.ranked_entities]
            self.assertIn("apply_invoice_total", ranked_names[:2])
            self.assertIn("compute_tier_rate", ranked_names[:3])
            # Test files must not pollute production fault candidates
            self.assertTrue(
                all(not item.entity.is_test for item in res.ranked_entities)
            )
            self.assertIn("tests/test_billing.py", res.linked_tests)


if __name__ == "__main__":
    unittest.main()
