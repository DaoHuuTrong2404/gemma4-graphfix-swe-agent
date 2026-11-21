"""Token-budgeted Minimal Context Slicing for Gemma 4 32K/128K context windows."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from .ast_localizer import ASTFaultLocalizer, CodeEntity, LocalizationResult


@dataclass
class ContextSlice:
    """Compact line-numbered repository slice tailored for Gemma 4 patch synthesis."""

    prompt_context: str
    estimated_tokens: int
    included_entities: list[str] = field(default_factory=list)
    included_files: list[str] = field(default_factory=list)
    truncated: bool = False


def estimate_tokens(text: str) -> int:
    """Conservative token estimator (~3.6 chars/token for Python code)."""
    return max(1, (len(text) + 3) // 4)


def _format_numbered_lines(lines: list[str], start_line: int, end_line: int) -> str:
    lo = max(1, start_line)
    hi = min(len(lines), end_line)
    out = []
    for ln in range(lo, hi + 1):
        out.append(f"L{ln:04d}: {lines[ln - 1]}")
    return "\n".join(out)


def _extract_import_block(lines: list[str], max_lines: int = 24) -> str:
    """Extract top-level import header lines from a Python module."""
    source = "\n".join(lines)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "\n".join(
            f"L{idx:04d}: {ln}"
            for idx, ln in enumerate(lines[:max_lines], start=1)
            if ln.strip().startswith(("import ", "from "))
        )
    import_lines: list[int] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            end_ln = getattr(node, "end_lineno", node.lineno) or node.lineno
            import_lines.extend(range(node.lineno, end_ln + 1))
    if not import_lines:
        return ""
    picked = sorted(set(import_lines))[:max_lines]
    return "\n".join(
        f"L{ln:04d}: {lines[ln - 1]}" for ln in picked if 1 <= ln <= len(lines)
    )


class MinimalContextSlicer:
    """Assemble minimal high-signal AST slices within strict token budgets."""

    def __init__(self, repo_root: str | Path, localizer: ASTFaultLocalizer) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.localizer = localizer
        self._file_cache: dict[str, list[str]] = {}

    def _get_lines(self, rel_path: str) -> list[str]:
        if rel_path not in self._file_cache:
            fp = self.repo_root / rel_path
            if fp.is_file():
                self._file_cache[rel_path] = fp.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            else:
                self._file_cache[rel_path] = []
        return self._file_cache[rel_path]

    def build_slice(
        self,
        localization: LocalizationResult,
        max_tokens: int = 2800,
        max_focus_entities: int = 4,
        include_neighbor_signatures: bool = True,
        include_test_snippet: bool = True,
    ) -> ContextSlice:
        """Build a surgical context bundle for Gemma 4 without dumping whole files."""
        sections: list[str] = []
        included_entities: list[str] = []
        included_files: list[str] = []
        used_tokens = 0
        truncated = False
        seen_imports: set[str] = set()
        seen_class_headers: set[tuple[str, str]] = set()

        # Focus on concrete methods/functions first; fallback to classes if needed
        candidates = [
            item for item in localization.ranked_entities if item.entity.kind != "class"
        ]
        if not candidates:
            candidates = list(localization.ranked_entities)

        for susp in candidates[:max_focus_entities]:
            ent: CodeEntity = susp.entity
            lines = self._get_lines(ent.rel_path)
            if not lines:
                continue

            parts: list[str] = [
                f"### TARGET `{ent.rel_path}` :: `{ent.qualname}` "
                f"(score={susp.score:.2f}, lines {ent.start_line}-{ent.end_line})"
            ]
            if susp.reasons:
                parts.append(f"# localization_evidence: {'; '.join(susp.reasons)}")

            if ent.rel_path not in seen_imports:
                imp = _extract_import_block(lines)
                if imp:
                    parts.append("# Module imports:\n" + imp)
                seen_imports.add(ent.rel_path)

            # Include full body of focal AST node (+1 line margin if small)
            pad_start = max(1, ent.start_line - 1)
            pad_end = min(len(lines), ent.end_line + 1)

            if (
                ent.parent_class
                and (ent.rel_path, ent.parent_class) not in seen_class_headers
            ):
                cls_entities = [
                    e
                    for e in self.localizer.file_entities.get(ent.rel_path, ())
                    if e.kind == "class" and e.name == ent.parent_class
                ]
                if cls_entities:
                    cls_ent = cls_entities[0]
                    if cls_ent.start_line < pad_start:
                        parts.append(
                            f"L{cls_ent.start_line:04d}: {lines[cls_ent.start_line - 1]}"
                        )
                seen_class_headers.add((ent.rel_path, ent.parent_class))

            parts.append(_format_numbered_lines(lines, pad_start, pad_end))

            if include_neighbor_signatures:
                neighbor_lines: list[str] = []
                for callee_id in self.localizer.call_graph.get(ent.entity_id, ())[:4]:
                    callee = self.localizer.entities.get(callee_id)
                    if callee:
                        doc = f" # {callee.docstring}" if callee.docstring else ""
                        neighbor_lines.append(
                            f"  -> calls {callee.rel_path}:{callee.start_line} `{callee.signature}`{doc}"
                        )
                for caller_id in self.localizer.reverse_graph.get(ent.entity_id, ())[
                    :3
                ]:
                    caller = self.localizer.entities.get(caller_id)
                    if caller and not caller.is_test:
                        neighbor_lines.append(
                            f"  <- called_by {caller.rel_path}:{caller.start_line} `{caller.signature}`"
                        )
                if neighbor_lines:
                    parts.append("# 1-hop AST neighbors:\n" + "\n".join(neighbor_lines))

            block = "\n".join(parts)
            block_tokens = estimate_tokens(block)
            if used_tokens + block_tokens > max_tokens and sections:
                truncated = True
                break
            sections.append(block)
            used_tokens += block_tokens
            included_entities.append(ent.entity_id)
            if ent.rel_path not in included_files:
                included_files.append(ent.rel_path)

        # Add compact failing test / traceback slice if budget remains
        if (
            include_test_snippet
            and localization.linked_tests
            and used_tokens < int(max_tokens * 0.85)
        ):
            test_rel = localization.linked_tests[0]
            tlines = self._get_lines(test_rel)
            if tlines:
                snippet_lines = tlines[:45]
                tblock = (
                    f"### LINKED TEST (READ-ONLY — DO NOT EDIT) `{test_rel}`\n"
                    + _format_numbered_lines(snippet_lines, 1, len(snippet_lines))
                )
                ttok = estimate_tokens(tblock)
                if used_tokens + ttok <= max_tokens:
                    sections.append(tblock)
                    used_tokens += ttok
                    if test_rel not in included_files:
                        included_files.append(test_rel)
                else:
                    truncated = True

        prompt_context = "\n\n".join(sections)
        return ContextSlice(
            prompt_context=prompt_context,
            estimated_tokens=estimate_tokens(prompt_context) if prompt_context else 0,
            included_entities=included_entities,
            included_files=included_files,
            truncated=truncated,
        )
