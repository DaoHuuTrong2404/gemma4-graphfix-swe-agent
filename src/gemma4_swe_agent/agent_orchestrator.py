"""End-to-end 5-stage Autonomous SWE Agent orchestrator optimized for Gemma 4."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .ast_localizer import ASTFaultLocalizer, LocalizationResult
from .context_slicer import ContextSlice, MinimalContextSlicer
from .diff_engine import apply_unified_diff
from .gemma4_runtime import DeterministicRuleRuntime, RuntimeBackend
from .patch_pipeline import (
    CandidateEvaluation,
    CandidatePatch,
    DeterministicPatchRanker,
    RegressionGuard,
)


@dataclass
class AgentRunResult:
    """Full structured trace and selected patch from AutonomousSWEAgent."""

    solved: bool
    selected_patch: str
    selected_evaluation: CandidateEvaluation | None
    ranked_candidates: list[CandidateEvaluation]
    localization: LocalizationResult
    context_slice: ContextSlice
    rounds_executed: int
    telemetry: dict[str, object] = field(default_factory=dict)


class AutonomousSWEAgent:
    """5-Stage Gemma 4 SWE Bug-Fixing Agent:
    1. AST Fault Localization (call graph + traceback + BM25)
    2. Minimal Context Slicing (token-budgeted AST window)
    3. Multi-Candidate Patch Synthesis (temperature/strategy arms)
    4. Test-Execution Self-Refinement & Regression Guard (isolated sandbox + anti-test-tamper)
    5. Deterministic Patch Ranking (lexicographic pass + zero-regression + blast radius + sha256)
    """

    def __init__(
        self,
        runtime: RuntimeBackend | None = None,
        max_context_tokens: int = 2800,
        max_refine_rounds: int = 2,
        timeout_seconds: int = 30,
        allow_test_edits: bool = False,
    ) -> None:
        self.runtime: RuntimeBackend = runtime or DeterministicRuleRuntime()
        self.max_context_tokens = max_context_tokens
        self.max_refine_rounds = max_refine_rounds
        self.timeout_seconds = timeout_seconds
        self.allow_test_edits = allow_test_edits

    def solve_issue(
        self,
        repo_root: str | Path,
        issue_text: str,
        target_cmd: Sequence[str],
        regression_cmd: Sequence[str] | None = None,
        traceback_text: str = "",
        apply_best: bool = False,
    ) -> AgentRunResult:
        root = Path(repo_root).resolve()

        # Stage 1: AST Fault Localization
        localizer = ASTFaultLocalizer(root)
        localizer.index_repository()
        localization = localizer.localize(
            issue_text=issue_text,
            traceback_text=traceback_text,
            top_k=8,
        )

        # Stage 2: Minimal Context Slicing
        slicer = MinimalContextSlicer(root, localizer)
        ctx_slice = slicer.build_slice(
            localization=localization,
            max_tokens=self.max_context_tokens,
            max_focus_entities=4,
        )

        # Stage 3 & 4: Multi-Candidate Synthesis + Test-Execution Self-Refinement + Regression Guard
        guard = RegressionGuard(
            repo_root=root,
            allow_test_edits=self.allow_test_edits,
            timeout_seconds=self.timeout_seconds,
        )
        all_evaluations: list[CandidateEvaluation] = []
        seen_hashes: set[str] = set()
        failure_feedback = traceback_text.strip()
        rounds_executed = 0

        for round_idx in range(self.max_refine_rounds + 1):
            rounds_executed = round_idx + 1
            proposals = self.runtime.synthesize_candidates(
                repo_root=str(root),
                issue_text=issue_text,
                context_slice=ctx_slice,
                failure_feedback=failure_feedback,
                round_idx=round_idx,
            )
            if not proposals:
                break

            round_evals: list[CandidateEvaluation] = []
            for cand_idx, prop in enumerate(proposals):
                cand = CandidatePatch(
                    candidate_id=f"r{round_idx}_c{cand_idx}_{prop.strategy}",
                    strategy=prop.strategy,
                    diff_text=prop.diff_text,
                    temperature=prop.temperature,
                    refine_round=round_idx,
                    raw_thoughts=prop.raw_thoughts,
                )
                ev = guard.evaluate_candidate(
                    candidate=cand,
                    target_cmd=target_cmd,
                    regression_cmd=regression_cmd,
                    localization=localization,
                )
                if ev.canonical_hash not in seen_hashes:
                    seen_hashes.add(ev.canonical_hash)
                    all_evaluations.append(ev)
                round_evals.append(ev)

            # Check if a policy-compliant, zero-regression solution was found
            passing = [
                ev
                for ev in round_evals
                if ev.policy_compliant
                and ev.apply_result.syntax_valid
                and ev.target_report
                and ev.target_report.passed
                and ev.regression_report
                and ev.regression_report.passed
            ]
            if passing:
                break

            # Prepare compact failure delta for next refinement round
            best_round = (
                DeterministicPatchRanker.rank(round_evals)[0] if round_evals else None
            )
            if best_round is not None:
                failure_feedback = (
                    best_round.rejection_reason or "No candidate passed tests"
                )

        # Stage 5: Deterministic Patch Ranking
        ranked = DeterministicPatchRanker.rank(all_evaluations)
        best = ranked[0] if ranked else None
        solved = bool(
            best is not None
            and best.policy_compliant
            and best.apply_result.syntax_valid
            and best.target_report
            and best.target_report.passed
            and best.regression_report
            and best.regression_report.passed
        )
        selected_patch = (
            (best.apply_result.canonical_diff or best.candidate.diff_text)
            if (best is not None and solved)
            else ""
        )

        if apply_best and solved and selected_patch:
            apply_unified_diff(
                root,
                selected_patch,
                write_to_disk=True,
                allow_test_edits=self.allow_test_edits,
            )

        return AgentRunResult(
            solved=solved,
            selected_patch=selected_patch,
            selected_evaluation=best,
            ranked_candidates=ranked,
            localization=localization,
            context_slice=ctx_slice,
            rounds_executed=rounds_executed,
            telemetry={
                "entities_indexed": localization.total_entities_indexed,
                "files_indexed": localization.total_files_indexed,
                "slice_tokens": ctx_slice.estimated_tokens,
                "candidates_evaluated": len(all_evaluations),
                "top_localized_file": localization.ranked_files[0][0]
                if localization.ranked_files
                else None,
            },
        )
