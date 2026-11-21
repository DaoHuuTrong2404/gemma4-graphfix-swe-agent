# Self-Refinement Delta Protocol

When a candidate patch fails verification:
1. Classify the failure: `SYNTAX_ERROR`, `ASSERTION_MISMATCH`, `REGRESSION_BROKEN`, `EDGE_CASE_EXCEPTION`, or `TIMEOUT`.
2. Inspect only the innermost traceback frame and compare expected vs actual values.
3. If `REGRESSION_BROKEN`, inspect `git diff` via `get_status` and tighten the conditional guard instead of broadening behavior.
4. Keep total blast radius minimal (minimum files, hunks, and added/removed lines).
