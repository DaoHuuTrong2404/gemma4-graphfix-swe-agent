---
name: ast_swe_toolkit
description: Static AST Fault Localization, Minimal Context Slicing, and Patch Syntax/Guard validation for Python repositories inside the Kaggle sandbox.
---

# AST SWE Toolkit Skill

Use `scripts/run_localize.py` via `run_command` before making edits on unfamiliar or multi-file Python repositories:

```bash
python skills/ast_swe_toolkit/scripts/run_localize.py --repo . --issue "<issue summary>" --max-tokens 2600
```

It returns:
1. Top suspicious production entities ranked by AST call-graph + traceback + BM25 IDF overlap.
2. Line-numbered minimal context slice (`L0001: ...`) ready for exact `edit_file` calls.
3. Linked read-only test files to execute after patching.
