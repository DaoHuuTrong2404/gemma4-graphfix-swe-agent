"""Gemma 4 runtime adapters: QAT vLLM, Transformers+PEFT, channel-thought stripping, and CPU deterministic benchmark runtime."""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .context_slicer import ContextSlice
from .diff_engine import parse_search_replace_blocks, search_replace_to_unified_diff

THOUGHT_CHANNEL_RE = re.compile(r"<\|channel>thought\b.*?<channel\|>", re.DOTALL)
THOUGHT_TAG_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL)
FENCED_DIFF_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)


def strip_gemma4_thoughts(raw_output: str) -> tuple[str, str]:
    """Separate Gemma 4 internal thought channel from user/tool output cleanly."""
    thoughts: list[str] = []
    for regex in (THOUGHT_CHANNEL_RE, THOUGHT_TAG_RE):
        for m in regex.findall(raw_output):
            thoughts.append(m.strip())
    cleaned = THOUGHT_CHANNEL_RE.sub("", raw_output)
    cleaned = THOUGHT_TAG_RE.sub("", cleaned).strip()
    return cleaned, "\n\n".join(thoughts)


def extract_diff_payload(text: str, repo_root: str | None = None) -> str:
    """Extract unified diff or SEARCH/REPLACE edit from Gemma 4 model output."""
    cleaned, _ = strip_gemma4_thoughts(text)
    if "<<<SEARCH" in cleaned and repo_root is not None:
        edits = parse_search_replace_blocks(cleaned)
        if edits:
            return search_replace_to_unified_diff(repo_root, edits)

    for block in FENCED_DIFF_RE.findall(cleaned):
        if "--- " in block and "+++ " in block and "@@ " in block:
            return block.strip() + "\n"

    if "--- " in cleaned and "+++ " in cleaned and "@@ " in cleaned:
        idx = cleaned.find("--- ")
        return cleaned[idx:].strip() + "\n"
    return ""


@dataclass
class SynthesisProposal:
    """Single patch generation output from a runtime backend."""

    strategy: str
    diff_text: str
    temperature: float
    raw_thoughts: str = ""


class RuntimeBackend(Protocol):
    """Protocol implemented by vLLM, HF Transformers, and DeterministicRuleRuntime."""

    def synthesize_candidates(
        self,
        repo_root: str,
        issue_text: str,
        context_slice: ContextSlice,
        failure_feedback: str = "",
        round_idx: int = 0,
    ) -> list[SynthesisProposal]: ...


class Gemma4VLLMRuntime:
    """Client for Gemma 4 QAT (`w4a16-ct`) served via vLLM OpenAI endpoint on Kaggle L4/H100."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model_id: str = "gemma-4-31b-it-qat-w4a16-ct",
        max_tokens: int = 2048,
        timeout_seconds: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def _chat(self, messages: list[dict[str, str]], temperature: float) -> str:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"] or ""

    def synthesize_candidates(
        self,
        repo_root: str,
        issue_text: str,
        context_slice: ContextSlice,
        failure_feedback: str = "",
        round_idx: int = 0,
    ) -> list[SynthesisProposal]:
        arms = [
            (
                "surgical_greedy",
                0.0,
                "Produce the smallest possible unified diff fixing the root cause.",
            ),
            (
                "boundary_guard",
                0.35,
                "Check boundary/off-by-one/None/empty-input invariants and emit a minimal unified diff.",
            ),
            (
                "state_invariant",
                0.65,
                "Check state mutation, aliasing, and type invariants and emit a minimal unified diff.",
            ),
        ]
        proposals: list[SynthesisProposal] = []
        for strategy, temp, hint in arms:
            sys_prompt = (
                "You are an autonomous Gemma 4 SWE bug-fixing agent. "
                "NEVER modify test files. Output ONLY a valid unified diff inside ```diff fences."
            )
            user_prompt = (
                f"ISSUE:\n{issue_text}\n\n"
                f"AST CONTEXT SLICE:\n{context_slice.prompt_context}\n\n"
                f"STRATEGY HINT: {hint}\n"
            )
            if failure_feedback:
                user_prompt += f"\nPREVIOUS TEST FAILURE DELTA (ROUND {round_idx}):\n{failure_feedback}\n"
            raw = self._chat(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temp,
            )
            _, thoughts = strip_gemma4_thoughts(raw)
            diff_txt = extract_diff_payload(raw, repo_root=repo_root)
            if diff_txt:
                proposals.append(
                    SynthesisProposal(
                        strategy=strategy,
                        diff_text=diff_txt,
                        temperature=temp,
                        raw_thoughts=thoughts,
                    )
                )
        return proposals


class Gemma4TransformersRuntime:
    """In-process HuggingFace Transformers + PEFT LoRA inference wrapper for Gemma 4."""

    def __init__(self, model: Any, tokenizer: Any, max_new_tokens: int = 1536) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def synthesize_candidates(
        self,
        repo_root: str,
        issue_text: str,
        context_slice: ContextSlice,
        failure_feedback: str = "",
        round_idx: int = 0,
    ) -> list[SynthesisProposal]:
        import torch

        arms = [("hf_greedy", 0.0), ("hf_sampled", 0.35)]
        proposals: list[SynthesisProposal] = []
        prompt = (
            f"Fix the following Python bug without modifying tests. Output ```diff unified diff.\n\n"
            f"ISSUE:\n{issue_text}\n\nCONTEXT:\n{context_slice.prompt_context}\n"
        )
        if failure_feedback:
            prompt += f"\nTEST FAILURE FEEDBACK:\n{failure_feedback}\n"
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        for strategy, temp in arms:
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": temp > 0.0,
            }
            if temp > 0.0:
                gen_kwargs["temperature"] = temp
                gen_kwargs["top_p"] = 0.95
                gen_kwargs["top_k"] = 64
            with torch.inference_mode():
                out_ids = self.model.generate(**inputs, **gen_kwargs)
            gen_only = out_ids[0][inputs["input_ids"].shape[1] :]
            raw = self.tokenizer.decode(gen_only, skip_special_tokens=False)
            _, thoughts = strip_gemma4_thoughts(raw)
            diff_txt = extract_diff_payload(raw, repo_root=repo_root)
            if diff_txt:
                proposals.append(
                    SynthesisProposal(
                        strategy=strategy,
                        diff_text=diff_txt,
                        temperature=temp,
                        raw_thoughts=thoughts,
                    )
                )
        return proposals


class DeterministicRuleRuntime:
    """Zero-GPU candidate generator for local CPU verification of the agent loop & ranking pipeline.

    Accepts injected candidate trajectories or derives targeted structural bug-fix hypotheses
    (e.g., mutable default argument, off-by-one slice/comparison, zero-division/empty guard)
    from the AST-localized context slice.
    """

    def __init__(
        self, scripted_rounds: dict[int, list[SynthesisProposal]] | None = None
    ) -> None:
        self.scripted_rounds = scripted_rounds or {}

    def synthesize_candidates(
        self,
        repo_root: str,
        issue_text: str,
        context_slice: ContextSlice,
        failure_feedback: str = "",
        round_idx: int = 0,
    ) -> list[SynthesisProposal]:
        if round_idx in self.scripted_rounds:
            return list(self.scripted_rounds[round_idx])

        proposals: list[SynthesisProposal] = []
        # Synthesize structural hypotheses directly from the top localized files
        for rel_path in context_slice.included_files:
            if rel_path.startswith("tests/") or "test_" in rel_path:
                continue
            fpath = Path(repo_root) / rel_path
            if not fpath.is_file():
                continue
            src = fpath.read_text(encoding="utf-8", errors="replace")

            # Hypothesis A: Mutable default list/dict in def signature -> None default
            m_mut = re.search(
                r"(def\s+\w+\([^)]*?)(\w+)\s*:\s*[^=,\)]+=\s*(\[\]|\{\})|"
                r"(def\s+\w+\([^)]*?)(\w+)\s*=\s*(\[\]|\{\})",
                src,
            )
            if m_mut:
                full_line_match = re.search(
                    r"^([ \t]*def\s+\w+\([^\n]*=\s*(\[\]|\{\})[^\n]*\)[^\n:]*:\n)([ \t]+)",
                    src,
                    re.MULTILINE,
                )
                if full_line_match:
                    hdr = full_line_match.group(1)
                    lit = full_line_match.group(2)
                    indent = full_line_match.group(3)
                    param_m = re.search(r"(\w+)\s*(?::[^,=()]+)?=\s*(?:\[\]|\{\})", hdr)
                    if param_m:
                        pname = param_m.group(1)
                        new_hdr = re.sub(
                            rf"(\b{re.escape(pname)}\s*(?::[^,=()]+)?=)\s*(?:\[\]|\{{\}})",
                            r"\1 None",
                            hdr,
                            count=1,
                        )
                        init_line = f"{indent}if {pname} is None:\n{indent}    {pname} = {lit}\n"
                        udiff = search_replace_to_unified_diff(
                            repo_root,
                            [(rel_path, hdr + indent, new_hdr + init_line + indent)],
                        )
                        proposals.append(
                            SynthesisProposal(
                                strategy="mutable_default_guard",
                                diff_text=udiff,
                                temperature=0.0,
                            )
                        )

            # Hypothesis B: Empty sequence guard before division by len(...)
            if (
                "ZeroDivisionError" in (issue_text + failure_feedback)
                or "/ len(" in src
            ):
                m_div = re.search(
                    r"^([ \t]+)return\s+sum\((\w+)\)\s*/\s*len\(\2\)\s*$",
                    src,
                    re.MULTILINE,
                )
                if m_div:
                    indent = m_div.group(1)
                    var = m_div.group(2)
                    old_blk = m_div.group(0)
                    new_blk = (
                        f"{indent}if not {var}:\n{indent}    return 0.0\n{old_blk}"
                    )
                    udiff = search_replace_to_unified_diff(
                        repo_root, [(rel_path, old_blk, new_blk)]
                    )
                    proposals.append(
                        SynthesisProposal(
                            strategy="empty_division_guard",
                            diff_text=udiff,
                            temperature=0.2,
                        )
                    )

            # Hypothesis C: Strict '<' vs '<=' inclusive upper bound / off-by-one candidate
            m_cmp = re.search(
                r"^([ \t]+if\s+[^:\n]+?)<([^=][^:\n]*:)$", src, re.MULTILINE
            )
            if m_cmp and (
                "inclusive" in issue_text.lower()
                or "boundary" in issue_text.lower()
                or "AssertionError" in failure_feedback
            ):
                old_ln = m_cmp.group(0)
                new_ln = f"{m_cmp.group(1)}<={m_cmp.group(2)}"
                udiff = search_replace_to_unified_diff(
                    repo_root, [(rel_path, old_ln, new_ln)]
                )
                proposals.append(
                    SynthesisProposal(
                        strategy="inclusive_boundary_fix",
                        diff_text=udiff,
                        temperature=0.35,
                    )
                )

        return proposals
