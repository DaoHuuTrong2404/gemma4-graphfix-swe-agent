"""Unified diff & SEARCH/REPLACE parser, atomic hunk applier, AST syntax guard, and blast-radius metrics."""

from __future__ import annotations

import ast
import difflib
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .ast_localizer import is_test_path

HUNK_HEADER_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(.*)$")
SEARCH_REPLACE_RE = re.compile(
    r"(?:^|\n)(?:FILE:\s*(?P<file>[^\n]+)\n)?<<<\s*SEARCH\s*\n(?P<search>.*?)\n===\s*\n(?P<replace>.*?)\n>>>\s*REPLACE",
    re.DOTALL,
)


@dataclass
class Hunk:
    """Single unified diff hunk."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]  # prefixed with ' ', '-', or '+'
    header_context: str = ""

    @property
    def expected_old_lines(self) -> list[str]:
        return [ln[1:] for ln in self.lines if ln.startswith((" ", "-"))]

    @property
    def replacement_lines(self) -> list[str]:
        return [ln[1:] for ln in self.lines if ln.startswith((" ", "+"))]

    @property
    def added_count(self) -> int:
        return sum(1 for ln in self.lines if ln.startswith("+"))

    @property
    def removed_count(self) -> int:
        return sum(1 for ln in self.lines if ln.startswith("-"))


@dataclass
class FileDiff:
    """Parsed unified diff for a single file."""

    old_path: str
    new_path: str
    hunks: list[Hunk] = field(default_factory=list)
    is_new_file: bool = False
    is_deleted: bool = False

    @property
    def target_path(self) -> str:
        if self.new_path and self.new_path != "/dev/null":
            return self.new_path
        return self.old_path


@dataclass(frozen=True)
class BlastRadius:
    """Quantitative size & risk footprint of a candidate patch."""

    files_touched: int
    hunks_count: int
    lines_added: int
    lines_removed: int
    net_churn: int
    touches_tests: bool
    modified_files: tuple[str, ...]

    @property
    def penalty(self) -> float:
        return round(
            (self.files_touched * 1.5)
            + (self.hunks_count * 0.6)
            + ((self.lines_added + self.lines_removed) * 0.08),
            6,
        )


@dataclass
class DiffApplyResult:
    """Result of validating and applying a unified diff to an in-memory or on-disk repo."""

    success: bool
    updated_files: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    syntax_valid: bool = False
    blast_radius: BlastRadius | None = None
    canonical_diff: str = ""


def _clean_diff_path(raw: str) -> str:
    raw = raw.strip()
    if raw == "/dev/null":
        return "/dev/null"
    if raw.startswith(("a/", "b/")):
        raw = raw[2:]
    return raw.replace("\\", "/").lstrip("./")


def _validate_safe_rel_path(rel_path: str) -> str:
    if not rel_path or rel_path == "/dev/null":
        raise ValueError(f"Invalid relative diff path: {rel_path!r}")
    p = Path(rel_path)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Path traversal blocked in diff: {rel_path!r}")
    return p.as_posix()


def parse_unified_diff(diff_text: str) -> list[FileDiff]:
    """Parse standard unified/git diff text into FileDiff objects."""
    lines = diff_text.splitlines()
    file_diffs: list[FileDiff] = []
    current_file: FileDiff | None = None
    current_hunk: Hunk | None = None
    pending_old: str | None = None

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            pending_old = None
            current_hunk = None
            i += 1
            continue
        if line.startswith("--- "):
            pending_old = _clean_diff_path(line[4:].split("\t")[0])
            current_hunk = None
            i += 1
            continue
        if line.startswith("+++ "):
            new_path = _clean_diff_path(line[4:].split("\t")[0])
            old_path = pending_old if pending_old is not None else new_path
            current_file = FileDiff(
                old_path=old_path,
                new_path=new_path,
                is_new_file=(old_path == "/dev/null"),
                is_deleted=(new_path == "/dev/null"),
            )
            file_diffs.append(current_file)
            current_hunk = None
            pending_old = None
            i += 1
            continue

        m = HUNK_HEADER_RE.match(line)
        if m:
            if current_file is None:
                raise ValueError(
                    f"Hunk header found before file header at line {i + 1}: {line}"
                )
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) is not None else 1
            current_hunk = Hunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                lines=[],
                header_context=m.group(5).strip(),
            )
            current_file.hunks.append(current_hunk)
            i += 1
            continue

        if current_hunk is not None:
            if line.startswith((" ", "-", "+")):
                current_hunk.lines.append(line)
            elif line.startswith("\\ No newline at end of file"):
                pass
            elif line == "" and (
                len(current_hunk.expected_old_lines) < current_hunk.old_count
                or len(current_hunk.replacement_lines) < current_hunk.new_count
            ):
                # Tolerate empty line whose leading space was stripped by markdown formatting
                current_hunk.lines.append(" ")
            else:
                current_hunk = None
        i += 1

    return [fd for fd in file_diffs if fd.hunks]


def parse_search_replace_blocks(
    text: str, default_file: str | None = None
) -> list[tuple[str, str, str]]:
    """Extract (file_path, search_block, replace_block) tuples from Gemma 4 edit blocks."""
    results: list[tuple[str, str, str]] = []
    for match in SEARCH_REPLACE_RE.finditer(text):
        fpath = (match.group("file") or default_file or "").strip()
        if not fpath:
            raise ValueError("SEARCH/REPLACE block missing FILE path")
        results.append(
            (_clean_diff_path(fpath), match.group("search"), match.group("replace"))
        )
    return results


def search_replace_to_unified_diff(
    repo_root: str | Path,
    edits: list[tuple[str, str, str]],
) -> str:
    """Convert exact SEARCH/REPLACE edits against files in repo_root into canonical unified diff."""
    root = Path(repo_root).resolve()
    grouped: dict[str, list[tuple[str, str]]] = {}
    for rel, search, replace in edits:
        safe_rel = _validate_safe_rel_path(rel)
        grouped.setdefault(safe_rel, []).append((search, replace))

    diff_chunks: list[str] = []
    for rel, file_edits in sorted(grouped.items()):
        fpath = root / rel
        if not fpath.is_file():
            raise FileNotFoundError(f"Target file not found for edit: {rel}")
        orig_text = fpath.read_text(encoding="utf-8", errors="replace")
        new_text = orig_text
        for search, replace in file_edits:
            count = new_text.count(search)
            if count == 0:
                raise ValueError(f"SEARCH block not found in {rel}")
            if count > 1:
                raise ValueError(f"Ambiguous SEARCH block ({count} matches) in {rel}")
            new_text = new_text.replace(search, replace, 1)

        orig_lines = orig_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        udiff = list(
            difflib.unified_diff(
                orig_lines,
                new_lines,
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                n=3,
            )
        )
        if udiff:
            diff_chunks.append("".join(udiff))
    return "\n".join(diff_chunks)


def compute_blast_radius(file_diffs: Iterable[FileDiff]) -> BlastRadius:
    """Compute files touched, hunk counts, line churn, and test-touch guard flag."""
    diffs = list(file_diffs)
    files: list[str] = []
    hunks_count = 0
    added = 0
    removed = 0
    touches_tests = False

    for fd in diffs:
        tpath = fd.target_path
        files.append(tpath)
        if is_test_path(tpath):
            touches_tests = True
        for h in fd.hunks:
            hunks_count += 1
            added += h.added_count
            removed += h.removed_count

    return BlastRadius(
        files_touched=len(set(files)),
        hunks_count=hunks_count,
        lines_added=added,
        lines_removed=removed,
        net_churn=abs(added - removed),
        touches_tests=touches_tests,
        modified_files=tuple(sorted(set(files))),
    )


def validate_python_syntax(rel_path: str, content: str) -> str | None:
    """Return error string if Python file has SyntaxError, else None."""
    if not rel_path.endswith(".py"):
        return None
    try:
        ast.parse(content, filename=rel_path)
        return None
    except SyntaxError as exc:
        return f"{rel_path}:{exc.lineno}: {exc.msg}"


def _locate_hunk_index(
    file_lines: list[str], hunk: Hunk, max_drift: int = 20
) -> int | None:
    """Find deterministic 0-based index where hunk.expected_old_lines matches verbatim."""
    expected = hunk.expected_old_lines
    if not expected:
        return max(0, min(len(file_lines), hunk.old_start - 1))

    exp_len = len(expected)
    nominal_idx = max(0, hunk.old_start - 1)
    if (
        nominal_idx + exp_len <= len(file_lines)
        and file_lines[nominal_idx : nominal_idx + exp_len] == expected
    ):
        return nominal_idx

    # Search bounded window around nominal_idx; accept only if unique match
    lo = max(0, nominal_idx - max_drift)
    hi = min(len(file_lines) - exp_len, nominal_idx + max_drift)
    matches = [
        idx for idx in range(lo, hi + 1) if file_lines[idx : idx + exp_len] == expected
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def apply_unified_diff(
    repo_root: str | Path,
    diff_text: str,
    write_to_disk: bool = False,
    allow_test_edits: bool = False,
) -> DiffApplyResult:
    """Validate, apply, and AST-check a unified diff without leaving partial edits on error."""
    root = Path(repo_root).resolve()
    try:
        file_diffs = parse_unified_diff(diff_text)
    except ValueError as exc:
        return DiffApplyResult(success=False, errors=[f"Diff parse error: {exc}"])

    if not file_diffs:
        return DiffApplyResult(success=False, errors=["Empty or unparseable diff"])

    blast = compute_blast_radius(file_diffs)
    if not allow_test_edits and blast.touches_tests:
        return DiffApplyResult(
            success=False,
            errors=["Policy violation: candidate patch modifies test files"],
            blast_radius=blast,
        )

    updated_files: dict[str, str] = {}
    canonical_parts: list[str] = []
    errors: list[str] = []

    for fd in file_diffs:
        try:
            rel_path = _validate_safe_rel_path(fd.target_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        fpath = root / rel_path
        if fd.is_new_file:
            orig_lines: list[str] = []
            had_trailing_newline = True
        else:
            if not fpath.is_file():
                errors.append(f"Target file does not exist: {rel_path}")
                continue
            raw_text = fpath.read_text(encoding="utf-8", errors="replace")
            had_trailing_newline = raw_text.endswith("\n") or raw_text == ""
            orig_lines = raw_text.splitlines()

        new_lines = list(orig_lines)
        offset_shift = 0
        hunk_failed = False

        for h_idx, hunk in enumerate(fd.hunks):
            if fd.is_new_file:
                new_lines.extend(hunk.replacement_lines)
                continue
            shifted_hunk = Hunk(
                old_start=hunk.old_start + offset_shift,
                old_count=hunk.old_count,
                new_start=hunk.new_start,
                new_count=hunk.new_count,
                lines=hunk.lines,
            )
            idx = _locate_hunk_index(new_lines, shifted_hunk)
            if idx is None:
                errors.append(
                    f"Hunk #{h_idx + 1} mismatch in {rel_path} near line {hunk.old_start}"
                )
                hunk_failed = True
                break
            exp_len = len(hunk.expected_old_lines)
            rep = hunk.replacement_lines
            new_lines[idx : idx + exp_len] = rep
            offset_shift += len(rep) - exp_len

        if hunk_failed:
            continue

        new_content = "\n".join(new_lines)
        if had_trailing_newline and new_lines:
            new_content += "\n"

        syntax_err = validate_python_syntax(rel_path, new_content)
        if syntax_err:
            errors.append(f"AST syntax check failed: {syntax_err}")
            continue

        updated_files[rel_path] = new_content
        old_for_canon = (
            "\n".join(orig_lines) + ("\n" if orig_lines else "")
        ).splitlines(keepends=True)
        new_for_canon = new_content.splitlines(keepends=True)
        canonical_parts.append(
            "".join(
                difflib.unified_diff(
                    old_for_canon,
                    new_for_canon,
                    fromfile=f"a/{rel_path}",
                    tofile=f"b/{rel_path}",
                    n=3,
                )
            )
        )

    if errors:
        return DiffApplyResult(
            success=False,
            errors=errors,
            syntax_valid=False,
            blast_radius=blast,
        )

    canonical_diff = "\n".join(p for p in canonical_parts if p)
    if write_to_disk:
        for rel_path, content in updated_files.items():
            target = root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    return DiffApplyResult(
        success=True,
        updated_files=updated_files,
        errors=[],
        syntax_valid=True,
        blast_radius=blast,
        canonical_diff=canonical_diff,
    )


def diff_sha256(canonical_diff: str) -> str:
    """Stable content hash for deterministic tie-breaking."""
    return hashlib.sha256(canonical_diff.encode("utf-8")).hexdigest()
