"""Gemma4-GraphFix: Autonomous SWE Bug-Fixing Agent optimized for Gemma 4."""

from .ast_localizer import (
    ASTFaultLocalizer,
    CodeEntity,
    LocalizationResult,
    SuspiciousEntity,
)
from .context_slicer import ContextSlice, MinimalContextSlicer
from .diff_engine import (
    BlastRadius,
    DiffApplyResult,
    FileDiff,
    Hunk,
    apply_unified_diff,
    compute_blast_radius,
    parse_search_replace_blocks,
    parse_unified_diff,
    search_replace_to_unified_diff,
    validate_python_syntax,
)
from .gemma4_runtime import (
    DeterministicRuleRuntime,
    Gemma4TransformersRuntime,
    Gemma4VLLMRuntime,
    extract_diff_payload,
    strip_gemma4_thoughts,
)
from .patch_pipeline import (
    CandidateEvaluation,
    CandidatePatch,
    DeterministicPatchRanker,
    RegressionGuard,
    TestExecutionReport,
)
from .agent_orchestrator import AgentRunResult, AutonomousSWEAgent

__all__ = [
    "ASTFaultLocalizer",
    "AgentRunResult",
    "AutonomousSWEAgent",
    "BlastRadius",
    "CandidateEvaluation",
    "CandidatePatch",
    "CodeEntity",
    "ContextSlice",
    "DeterministicPatchRanker",
    "DeterministicRuleRuntime",
    "DiffApplyResult",
    "FileDiff",
    "Gemma4TransformersRuntime",
    "Gemma4VLLMRuntime",
    "Hunk",
    "LocalizationResult",
    "MinimalContextSlicer",
    "RegressionGuard",
    "SuspiciousEntity",
    "TestExecutionReport",
    "apply_unified_diff",
    "compute_blast_radius",
    "extract_diff_payload",
    "parse_search_replace_blocks",
    "parse_unified_diff",
    "search_replace_to_unified_diff",
    "strip_gemma4_thoughts",
    "validate_python_syntax",
]
