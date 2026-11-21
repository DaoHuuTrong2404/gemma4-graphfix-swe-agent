"""Multi-candidate patch synthesis, test-execution self-refinement, regression guard, and deterministic ranking."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .ast_localizer import ASTFaultLocalizer, LocalizationResult
from .diff_engine import BlastRadius, DiffApplyResult, apply_unified_diff, diff_sha256

UNITTEST_RAN_RE = re.compile(r"Ran\s+(\d+)\s+tests?\s+in")
PYTEST_PASSED_RE = re.compile(r"(\d+)\s+passed")
PYTEST_FAILED_RE = re.compile(r"(\d+)\s+failed")
ERROR_EXC_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\s*:\s*(.+)")


@dataclass
class CandidatePatch:
    """Single patch candidate synthesized by a Gemma 4 strategy/temperature arm."""

    candidate_id: str
    strategy: str
    diff_text: str
    temperature: float = 0.0
    refine_round: int = 0
    raw_thoughts: str = ""


@dataclass
class TestExecutionReport:
    """Structured test execution & traceback summary from sandbox validation."""

    passed: bool
    returncode: int
    total_run: int
    failed_count: int
    stdout: str
    stderr: str
    duration_sec: float
    failure_summary: str = ""
    timed_out: bool = False


@dataclass
class CandidateEvaluation:
    """Complete evaluation & ranking record for one candidate patch."""

    candidate: CandidatePatch
    apply_result: DiffApplyResult
    target_report: TestExecutionReport | None
    regression_report: TestExecutionReport | None
    policy_compliant: bool
    localization_aligned: bool
    rank_score: float
    rank_tuple: tuple
    canonical_hash: str
    rejection_reason: str = ""


def _parse_failure_summary(stdout: str, stderr: str) -> str:
    combined = f"{stdout}\n{stderr}".strip()
    lines = [ln for ln in combined.splitlines() if ln.strip()]
    exc_hits = ERROR_EXC_RE.findall(combined)
    exc_line = f"{exc_hits[-1][0]}: {exc_hits[-1][1]}" if exc_hits else ""
    tb_tail = "\n".join(lines[-10:])
    if exc_line and exc_line not in tb_tail:
        return f"{exc_line}\n{tb_tail}"
    return tb_tail[:900]


def _parse_test_counts(stdout: str, stderr: str, passed: bool) -> tuple[int, int]:
    combined = f"{stdout}\n{stderr}"
    m_unit = UNITTEST_RAN_RE.search(combined)
    if m_unit:
        total = int(m_unit.group(1))
        failed = (
            0 if passed else max(1, combined.count("FAIL:") + combined.count("ERROR:"))
        )
        return total, min(total, failed)
    m_p = PYTEST_PASSED_RE.search(combined)
    m_f = PYTEST_FAILED_RE.search(combined)
    p_cnt = int(m_p.group(1)) if m_p else 0
    f_cnt = int(m_f.group(1)) if m_f else (0 if passed else 1)
    total = p_cnt + f_cnt
    return max(total, 1 if not passed else total), f_cnt


def _ignore_copy(_dir: str, names: list[str]) -> list[str]:
    return [
        n
        for n in names
        if n in {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}
    ]


class RegressionGuard:
    """Runs target + regression tests inside an isolated sandbox copy with strict anti-tamper checks."""

    def __init__(
        self,
        repo_root: str | Path,
        allow_test_edits: bool = False,
        timeout_seconds: int = 30,
        python_executable: str | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.allow_test_edits = allow_test_edits
        self.timeout_seconds = timeout_seconds
        self.python_exe = python_executable or sys.executable

    def _run_suite_in_dir(
        self, workdir: Path, cmd: Sequence[str]
    ) -> TestExecutionReport:
        t0 = time.perf_counter()
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run(
                list(cmd),
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
            )
            dur = round(time.perf_counter() - t0, 4)
            passed = proc.returncode == 0
            total, failed = _parse_test_counts(proc.stdout, proc.stderr, passed)
            summary = "" if passed else _parse_failure_summary(proc.stdout, proc.stderr)
            return TestExecutionReport(
                passed=passed,
                returncode=proc.returncode,
                total_run=total,
                failed_count=failed,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_sec=dur,
                failure_summary=summary,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            dur = round(time.perf_counter() - t0, 4)
            return TestExecutionReport(
                passed=False,
                returncode=124,
                total_run=1,
                failed_count=1,
                stdout=str(exc.stdout or ""),
                stderr=str(exc.stderr or ""),
                duration_sec=dur,
                failure_summary=f"TimeoutExpired after {self.timeout_seconds}s",
                timed_out=True,
            )

    def evaluate_candidate(
        self,
        candidate: CandidatePatch,
        target_cmd: Sequence[str],
        regression_cmd: Sequence[str] | None = None,
        localization: LocalizationResult | None = None,
    ) -> CandidateEvaluation:
        """Apply candidate to isolated temp sandbox, validate syntax/policy, and run tests."""
        apply_res = apply_unified_diff(
            self.repo_root,
            candidate.diff_text,
            write_to_disk=False,
            allow_test_edits=self.allow_test_edits,
        )

        if not apply_res.success or not apply_res.syntax_valid:
            reason = (
                "; ".join(apply_res.errors)
                or "Diff application or AST syntax check failed"
            )
            policy_ok = not (
                apply_res.blast_radius
                and apply_res.blast_radius.touches_tests
                and not self.allow_test_edits
            )
            return CandidateEvaluation(
                candidate=candidate,
                apply_result=apply_res,
                target_report=None,
                regression_report=None,
                policy_compliant=policy_ok,
                localization_aligned=False,
                rank_score=-1000.0,
                rank_tuple=(0, 0, 0, 0, 0.0, 0, -9999.0, ""),
                canonical_hash=diff_sha256(candidate.diff_text),
                rejection_reason=reason,
            )

        blast = apply_res.blast_radius
        policy_ok = bool(
            blast is not None and (self.allow_test_edits or not blast.touches_tests)
        )

        # Check alignment with top-3 localized production files
        loc_aligned = False
        if localization and localization.ranked_files and blast:
            top_files = {f for f, _ in localization.ranked_files[:3]}
            loc_aligned = bool(top_files & set(blast.modified_files))

        with tempfile.TemporaryDirectory(prefix="gemma4_sandbox_") as tmp:
            sandbox_dir = Path(tmp) / "repo"
            shutil.copytree(self.repo_root, sandbox_dir, ignore=_ignore_copy)
            for rel_path, new_content in apply_res.updated_files.items():
                dst = sandbox_dir / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(new_content, encoding="utf-8")

            target_rep = self._run_suite_in_dir(sandbox_dir, target_cmd)
            if regression_cmd is not None:
                reg_rep = self._run_suite_in_dir(sandbox_dir, regression_cmd)
            else:
                reg_rep = target_rep

        return DeterministicPatchRanker.score_evaluation(
            candidate=candidate,
            apply_result=apply_res,
            target_report=target_rep,
            regression_report=reg_rep,
            policy_compliant=policy_ok,
            localization_aligned=loc_aligned,
        )


class DeterministicPatchRanker:
    """Deterministic lexicographic + composite patch selector with SHA-256 tie-break."""

    @staticmethod
    def score_evaluation(
        candidate: CandidatePatch,
        apply_result: DiffApplyResult,
        target_report: TestExecutionReport | None,
        regression_report: TestExecutionReport | None,
        policy_compliant: bool,
        localization_aligned: bool,
    ) -> CandidateEvaluation:
        blast: BlastRadius | None = apply_result.blast_radius
        blast_penalty = blast.penalty if blast else 999.0
        canon_hash = diff_sha256(apply_result.canonical_diff or candidate.diff_text)

        t_pass = bool(target_report and target_report.passed)
        r_pass = bool(regression_report and regression_report.passed)

        total_tests = 0
        passed_tests = 0
        if target_report:
            total_tests += target_report.total_run
            passed_tests += max(0, target_report.total_run - target_report.failed_count)
        if regression_report and regression_report is not target_report:
            total_tests += regression_report.total_run
            passed_tests += max(
                0, regression_report.total_run - regression_report.failed_count
            )
        pass_ratio = round(passed_tests / max(1, total_tests), 6)

        # Lexicographic priority guarantees zero-regression + target-pass beats smaller partial fixes
        # Negate hex prefix int so smaller hash wins deterministically if literally every metric ties
        hash_tie = -int(canon_hash[:12], 16)
        rank_tuple = (
            1 if policy_compliant else 0,
            1 if apply_result.syntax_valid else 0,
            1 if t_pass else 0,
            1 if r_pass else 0,
            pass_ratio,
            1 if localization_aligned else 0,
            round(-blast_penalty, 6),
            hash_tie,
        )

        composite = (
            (1000.0 if policy_compliant else -1000.0)
            + (200.0 if apply_result.syntax_valid else -500.0)
            + (500.0 if t_pass else 0.0)
            + (400.0 if r_pass else -250.0)
            + (100.0 * pass_ratio)
            + (25.0 if localization_aligned else 0.0)
            - blast_penalty
        )

        rejection = ""
        if not policy_compliant:
            rejection = "Modified forbidden test file"
        elif not apply_result.syntax_valid:
            rejection = "; ".join(apply_result.errors)
        elif not t_pass:
            rejection = f"Target tests failed: {target_report.failure_summary if target_report else 'no run'}"
        elif not r_pass:
            rejection = f"Regression tests failed: {regression_report.failure_summary if regression_report else 'no run'}"

        return CandidateEvaluation(
            candidate=candidate,
            apply_result=apply_result,
            target_report=target_report,
            regression_report=regression_report,
            policy_compliant=policy_compliant,
            localization_aligned=localization_aligned,
            rank_score=round(composite, 4),
            rank_tuple=rank_tuple,
            canonical_hash=canon_hash,
            rejection_reason=rejection,
        )

    @staticmethod
    def rank(evaluations: Sequence[CandidateEvaluation]) -> list[CandidateEvaluation]:
        """Sort candidates strictly from best to worst."""
        return sorted(evaluations, key=lambda ev: ev.rank_tuple, reverse=True)
