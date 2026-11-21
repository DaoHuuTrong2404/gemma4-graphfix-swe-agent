# Gemma4-GraphFix: Autonomous SWE Developer Agent

**Competition:** [Google - The Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent)  
**Critical Deadlines:**
- **Entry & Rules Acceptance Deadline:** **25/11/2026** (MUST accept rules on Kaggle before this date!)
- **Final Submission Deadline:** **02/12/2026 23:59 UTC**

## Overview
`Gemma4-GraphFix` is a 5-stage autonomous bug-fixing SWE agent engineered specifically for Google's open-weight **Gemma 4** QAT models (`gemma-4-31b-it-qat-w4a16-ct` and `gemma-4-e4b-it-qat-w4a16-ct`) inside Kaggle's `swegemma` / Google ADK evaluation harness:

1. **AST Fault Localization (`ast_localizer.py`)**: Static Python `ast` symbol + call-graph extractor paired with traceback innermost-frame weighting, BM25 sub-token IDF scoring, and 2-hop call-graph propagation while keeping test files strictly separated.
2. **Minimal Context Slicing (`context_slicer.py`)**: Budgeted line-numbered (`L0001:`) AST window emitting imports, enclosing class headers, focal function bodies, and 1-hop caller/callee signatures within tight 32K/128K token limits.
3. **Multi-Candidate Patch Synthesis (`gemma4_runtime.py`)**: Strips Gemma 4 internal thought channels (`<|channel>thought...<channel|>`), extracts unified diffs or `SEARCH/REPLACE` blocks across strategy/temperature arms (`surgical_greedy`, `boundary_guard`, `state_invariant`).
4. **Test-Execution Self-Refinement & Regression Guard (`diff_engine.py`, `patch_pipeline.py`)**: Enforces path-traversal protection, pre-apply AST syntax validation, strict anti-test-tampering (`allow_test_edits=False`), isolated sandbox testing (`target_cmd` + `regression_cmd`), and traceback-delta refinement.
5. **Deterministic Patch Ranking (`patch_pipeline.py`)**: Lexicographic ranking (`policy_ok > syntax_valid > target_pass > zero_regression > pass_ratio > loc_alignment > -blast_radius > sha256_tiebreak`).

## Project Structure
- `submission_bundle/` & `submission.zip`: Ready-to-upload ADK bundle (`agent.yaml` at archive root, `agent_e4b_fast.yaml`, `eval_config.yaml`, `configs/sampling.yaml`, `prompts/`, `skills/ast_swe_toolkit/`).
- `gemma4_swe_agent_kaggle.ipynb` & `kernel-metadata.json`: Self-contained Kaggle GPU Notebook (runs verification, QLoRA/vLLM setup, and builds `submission.zip`).
- `src/gemma4_swe_agent/`: Clean modular Python package with zero third-party runtime dependencies for core AST/diff/ranking logic.
- `tests/`: Unit & end-to-end SWE bug-fixing benchmark suite.
- `architecture_dossier.pdf` / `architecture_dossier.png`: Competition submission & architecture dossier.

## Quick Verification
```powershell
& "C:\Users\huutr\.local\bin\python.exe" -m unittest discover -s tests -v
& "C:\Users\huutr\.local\bin\python.exe" build_submission.py
```
