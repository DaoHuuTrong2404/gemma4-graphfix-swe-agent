"""Self-contained AST fault localizer + minimal slicer CLI bundled inside ADK skills/."""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
from collections import defaultdict
from pathlib import Path

IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "build", "dist"}


def is_test_path(rel_path: str) -> bool:
    norm = rel_path.replace("\\", "/").lower()
    parts = norm.split("/")
    if any(p in ("tests", "test", "testing") for p in parts[:-1]):
        return True
    name = parts[-1]
    return name.startswith("test_") or name.endswith("_test.py")


def tokenize(text: str) -> set[str]:
    out: set[str] = set()
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        for part in word.split("_"):
            for cp in re.findall(r"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z]+|[0-9]+", part):
                low = cp.lower()
                if len(low) >= 2:
                    out.add(low)
    return out


def localize_repo(repo_root: Path, query: str, top_k: int = 5) -> str:
    q_tokens = tokenize(query)
    exact = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", query))
    candidates: list[tuple[float, str, str, int, int]] = []
    tests: list[str] = []

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS and not d.startswith("."))
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            fpath = Path(root) / fname
            rel = fpath.relative_to(repo_root).as_posix()
            text = fpath.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            lines = text.splitlines()
            if is_test_path(rel):
                if tokenize(text) & q_tokens:
                    tests.append(rel)
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    end_ln = getattr(node, "end_lineno", node.lineno) or node.lineno
                    body = "\n".join(lines[node.lineno - 1 : end_ln])
                    overlap = tokenize(body) & q_tokens
                    score = len(overlap) / (1.0 + 0.1 * math.log1p(max(1, end_ln - node.lineno + 1)))
                    if node.name in exact:
                        score += 3.0
                    if rel in query:
                        score += 2.0
                    if score > 0:
                        candidates.append((round(score, 3), rel, node.name, node.lineno, end_ln))

    candidates.sort(key=lambda x: (-x[0], x[1], x[3]))
    out = ["[AST_LOCALIZER_RESULTS]"]
    for score, rel, name, s_ln, e_ln in candidates[:top_k]:
        out.append(f"- {rel}:{s_ln}-{e_ln} `{name}` (score={score})")
        lines = (repo_root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
        for ln in range(s_ln, min(len(lines), e_ln) + 1):
            out.append(f"  L{ln:04d}: {lines[ln - 1]}")
    if tests:
        out.append("[LINKED_TESTS_DO_NOT_EDIT]")
        out.extend(f"- {t}" for t in tests[:5])
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run local AST fault localization inside sandbox")
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--issue", required=True)
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()
    print(localize_repo(args.repo.resolve(), args.issue, top_k=args.top_k))


if __name__ == "__main__":
    main()
