"""AST Fault Localization with symbol/call-graph propagation for Gemma 4 SWE agents."""

from __future__ import annotations

import ast
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "build",
        "dist",
        ".eggs",
    }
)

STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "when",
        "where",
        "what",
        "which",
        "while",
        "should",
        "would",
        "could",
        "fail",
        "fails",
        "failed",
        "error",
        "exception",
        "traceback",
        "line",
        "file",
        "none",
        "true",
        "false",
        "self",
        "cls",
        "args",
        "kwargs",
        "return",
        "returns",
        "raise",
        "raises",
        "test",
        "tests",
        "def",
        "class",
        "import",
        "bug",
        "issue",
        "fix",
    }
)

TRACEBACK_RE = re.compile(
    r'File\s+["\'](?P<path>[^"\']+\.py)["\']\s*,\s*line\s+(?P<line>\d+)(?:\s*,\s*in\s+(?P<func>[A-Za-z_][A-Za-z0-9_<>]*))?'
)
PYTEST_LINE_RE = re.compile(r"(?P<path>[A-Za-z0-9_./\\-]+\.py):(?P<line>\d+):")
FILE_MENTION_RE = re.compile(r"(?P<path>[A-Za-z0-9_./\\-]+\.py)")


def split_identifier_tokens(text: str) -> list[str]:
    """Split snake_case and CamelCase identifiers into lowercase search tokens."""
    raw_words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    tokens: list[str] = []
    for word in raw_words:
        for snake_part in word.split("_"):
            if not snake_part:
                continue
            camel_parts = re.findall(
                r"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z]+|[0-9]+", snake_part
            )
            if camel_parts:
                for cp in camel_parts:
                    low = cp.lower()
                    if len(low) >= 2 and low not in STOPWORDS:
                        tokens.append(low)
            else:
                low = snake_part.lower()
                if len(low) >= 2 and low not in STOPWORDS:
                    tokens.append(low)
    return tokens


def is_test_path(rel_path: str) -> bool:
    """Return True when a repository-relative path represents a test file."""
    norm = rel_path.replace("\\", "/").lower()
    parts = norm.split("/")
    filename = parts[-1]
    if any(p in ("tests", "test", "testing") for p in parts[:-1]):
        return True
    return filename.startswith("test_") or filename.endswith("_test.py")


@dataclass(frozen=True)
class TracebackHit:
    """Single parsed stack frame from a traceback or pytest output."""

    rel_path: str
    line: int
    func_name: str
    depth_weight: float


@dataclass
class CodeEntity:
    """AST-indexed function, method, or class inside a Python repository."""

    entity_id: str
    rel_path: str
    qualname: str
    name: str
    kind: str  # "function", "async_function", "method", "class"
    start_line: int
    end_line: int
    parent_class: str | None = None
    signature: str = ""
    docstring: str = ""
    calls: set[str] = field(default_factory=set)
    identifiers: set[str] = field(default_factory=set)
    imports: set[str] = field(default_factory=set)
    is_test: bool = False

    def contains_line(self, line: int) -> bool:
        return self.start_line <= line <= self.end_line

    @property
    def span_lines(self) -> int:
        return max(1, self.end_line - self.start_line + 1)


@dataclass
class SuspiciousEntity:
    """Ranked candidate fault location with decomposed evidence."""

    entity: CodeEntity
    score: float
    direct_score: float
    graph_score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class LocalizationResult:
    """Complete AST fault localization output."""

    ranked_entities: list[SuspiciousEntity]
    ranked_files: list[tuple[str, float]]
    linked_tests: list[str]
    traceback_hits: list[TracebackHit]
    call_graph_edges: dict[str, list[str]]
    total_entities_indexed: int
    total_files_indexed: int


class _EntityVisitor(ast.NodeVisitor):
    """Collect AST symbol definitions, signatures, calls, and lexical tokens."""

    def __init__(
        self, rel_path: str, module_name: str, source_lines: list[str]
    ) -> None:
        self.rel_path = rel_path
        self.module_name = module_name
        self.source_lines = source_lines
        self.is_test = is_test_path(rel_path)
        self.module_imports: set[str] = set()
        self.entities: list[CodeEntity] = []
        self._class_stack: list[str] = []
        self._entity_stack: list[CodeEntity] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.module_imports.add(alias.name.split(".")[0])
            if self._entity_stack:
                self._entity_stack[-1].imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if mod:
            self.module_imports.add(mod.split(".")[0])
        for alias in node.names:
            self.module_imports.add(alias.name)
            if self._entity_stack:
                self._entity_stack[-1].imports.add(f"{mod}.{alias.name}".strip("."))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        end_line = getattr(node, "end_lineno", node.lineno) or node.lineno
        qual_parts = [self.module_name, *self._class_stack, node.name]
        qualname = ".".join(p for p in qual_parts if p)
        bases = [ast.unparse(b) for b in node.bases] if hasattr(ast, "unparse") else []
        sig = (
            f"class {node.name}({', '.join(bases)}):"
            if bases
            else f"class {node.name}:"
        )
        doc = (
            (ast.get_docstring(node) or "").splitlines()[0]
            if ast.get_docstring(node)
            else ""
        )
        entity = CodeEntity(
            entity_id=f"{self.rel_path}::{qualname}",
            rel_path=self.rel_path,
            qualname=qualname,
            name=node.name,
            kind="class",
            start_line=node.lineno,
            end_line=end_line,
            parent_class=self._class_stack[-1] if self._class_stack else None,
            signature=sig,
            docstring=doc,
            imports=set(self.module_imports),
            is_test=self.is_test,
        )
        entity.identifiers.update(split_identifier_tokens(node.name))
        if doc:
            entity.identifiers.update(split_identifier_tokens(doc))
        self.entities.append(entity)

        self._class_stack.append(node.name)
        self._entity_stack.append(entity)
        self.generic_visit(node)
        self._entity_stack.pop()
        self._class_stack.pop()

    def _visit_func(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool
    ) -> None:
        end_line = getattr(node, "end_lineno", node.lineno) or node.lineno
        parent_cls = self._class_stack[-1] if self._class_stack else None
        qual_parts = [self.module_name, *self._class_stack, node.name]
        qualname = ".".join(p for p in qual_parts if p)
        prefix = "async def" if is_async else "def"
        arg_names = [a.arg for a in node.args.posonlyargs + node.args.args]
        if node.args.vararg:
            arg_names.append(f"*{node.args.vararg.arg}")
        arg_names.extend(a.arg for a in node.args.kwonlyargs)
        if node.args.kwarg:
            arg_names.append(f"**{node.args.kwarg.arg}")
        ret_ann = ""
        if node.returns is not None and hasattr(ast, "unparse"):
            try:
                ret_ann = f" -> {ast.unparse(node.returns)}"
            except Exception:
                ret_ann = ""
        sig = f"{prefix} {node.name}({', '.join(arg_names)}){ret_ann}:"
        raw_doc = ast.get_docstring(node) or ""
        first_doc = raw_doc.splitlines()[0] if raw_doc else ""
        kind = (
            "method" if parent_cls else ("async_function" if is_async else "function")
        )

        entity = CodeEntity(
            entity_id=f"{self.rel_path}::{qualname}",
            rel_path=self.rel_path,
            qualname=qualname,
            name=node.name,
            kind=kind,
            start_line=node.lineno,
            end_line=end_line,
            parent_class=parent_cls,
            signature=sig,
            docstring=first_doc,
            imports=set(self.module_imports),
            is_test=self.is_test,
        )
        body_slice = "\n".join(self.source_lines[node.lineno - 1 : end_line])
        entity.identifiers.update(split_identifier_tokens(node.name))
        entity.identifiers.update(split_identifier_tokens(body_slice))
        if parent_cls:
            entity.identifiers.update(split_identifier_tokens(parent_cls))

        self.entities.append(entity)
        self._entity_stack.append(entity)
        self.generic_visit(node)
        self._entity_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node, is_async=True)

    def visit_Call(self, node: ast.Call) -> None:
        if self._entity_stack:
            target_name = None
            if isinstance(node.func, ast.Name):
                target_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                target_name = node.func.attr
            if target_name:
                self._entity_stack[-1].calls.add(target_name)
        self.generic_visit(node)


class ASTFaultLocalizer:
    """Hybrid AST symbol-graph + traceback + BM25 fault localizer for Python repos."""

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.entities: dict[str, CodeEntity] = {}
        self.file_entities: dict[str, list[CodeEntity]] = defaultdict(list)
        self.name_to_ids: dict[str, list[str]] = defaultdict(list)
        self.call_graph: dict[str, list[str]] = defaultdict(list)
        self.reverse_graph: dict[str, list[str]] = defaultdict(list)
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.file_count = 0
        self._indexed = False

    def index_repository(self) -> None:
        """Walk repository Python files and construct AST symbol & call graphs."""
        self.entities.clear()
        self.file_entities.clear()
        self.name_to_ids.clear()
        self.call_graph.clear()
        self.reverse_graph.clear()
        self.doc_freq.clear()
        self.file_count = 0

        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = sorted(
                d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")
            )
            for fname in sorted(files):
                if not fname.endswith(".py"):
                    continue
                full_path = Path(root) / fname
                rel_path = full_path.relative_to(self.repo_root).as_posix()
                try:
                    text = full_path.read_text(encoding="utf-8", errors="replace")
                    tree = ast.parse(text, filename=rel_path)
                except SyntaxError:
                    continue
                self.file_count += 1
                mod_name = rel_path[:-3].replace("/", ".")
                if mod_name.endswith(".__init__"):
                    mod_name = mod_name[: -len(".__init__")]
                visitor = _EntityVisitor(rel_path, mod_name, text.splitlines())
                visitor.visit(tree)
                for entity in visitor.entities:
                    self.entities[entity.entity_id] = entity
                    self.file_entities[rel_path].append(entity)
                    self.name_to_ids[entity.name].append(entity.entity_id)
                    for tok in entity.identifiers:
                        self.doc_freq[tok] += 1

        # Resolve symbol calls into directed edges (caller -> callee)
        for caller_id, caller in self.entities.items():
            seen_targets: set[str] = set()
            for call_name in caller.calls:
                for target_id in self.name_to_ids.get(call_name, ()):
                    if target_id == caller_id or target_id in seen_targets:
                        continue
                    target = self.entities[target_id]
                    if target.is_test and not caller.is_test:
                        continue
                    seen_targets.add(target_id)
                    self.call_graph[caller_id].append(target_id)
                    self.reverse_graph[target_id].append(caller_id)

        self._indexed = True

    def _normalize_repo_path(self, raw_path: str) -> str | None:
        cleaned = raw_path.replace("\\", "/").lstrip("./")
        if cleaned in self.file_entities:
            return cleaned
        # Handle absolute or nested paths by longest suffix match
        candidates = [
            rel
            for rel in self.file_entities
            if cleaned.endswith("/" + rel)
            or rel.endswith("/" + cleaned)
            or rel == cleaned
        ]
        if candidates:
            return max(candidates, key=len)
        base = Path(cleaned).name
        by_name = [rel for rel in self.file_entities if Path(rel).name == base]
        return by_name[0] if len(by_name) == 1 else None

    def parse_traceback_hits(self, issue_or_trace: str) -> list[TracebackHit]:
        """Extract stack frames and pytest location hits with innermost boost."""
        raw_hits: list[tuple[str, int, str]] = []
        for match in TRACEBACK_RE.finditer(issue_or_trace):
            rel = self._normalize_repo_path(match.group("path"))
            if rel:
                raw_hits.append(
                    (rel, int(match.group("line")), match.group("func") or "")
                )
        for match in PYTEST_LINE_RE.finditer(issue_or_trace):
            rel = self._normalize_repo_path(match.group("path"))
            if rel:
                raw_hits.append((rel, int(match.group("line")), ""))

        if not raw_hits:
            return []

        # Deduplicate while preserving order; deepest stack frames get highest weight
        deduped: list[tuple[str, int, str]] = []
        seen: set[tuple[str, int]] = set()
        for item in raw_hits:
            key = (item[0], item[1])
            if key not in seen:
                seen.add(key)
                deduped.append(item)

        hits: list[TracebackHit] = []
        total = len(deduped)
        for idx, (rel, line, func_name) in enumerate(deduped):
            # Innermost frame (largest idx) receives up to 1.6x boost
            depth_weight = 1.0 + (0.6 * (idx / max(1, total - 1))) if total > 1 else 1.4
            hits.append(
                TracebackHit(
                    rel_path=rel,
                    line=line,
                    func_name=func_name,
                    depth_weight=round(depth_weight, 4),
                )
            )
        return hits

    def _idf(self, token: str) -> float:
        n = max(1, len(self.entities))
        df = self.doc_freq.get(token, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def localize(
        self,
        issue_text: str,
        traceback_text: str = "",
        top_k: int = 8,
        include_tests: bool = False,
    ) -> LocalizationResult:
        """Rank suspicious AST entities and files using hybrid static + traceback evidence."""
        if not self._indexed:
            self.index_repository()

        combined_query = f"{issue_text}\n{traceback_text}".strip()
        tb_hits = self.parse_traceback_hits(combined_query)
        mentioned_files: set[str] = set()
        for match in FILE_MENTION_RE.finditer(combined_query):
            norm = self._normalize_repo_path(match.group("path"))
            if norm:
                mentioned_files.add(norm)

        exact_identifiers = set(
            re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", combined_query)
        )
        query_tokens = set(split_identifier_tokens(combined_query))

        direct_scores: dict[str, float] = defaultdict(float)
        reasons_map: dict[str, list[str]] = defaultdict(list)

        # 1. Exact stacktrace / failing test line alignment
        for hit in tb_hits:
            for ent in self.file_entities.get(hit.rel_path, ()):
                if ent.contains_line(hit.line):
                    specificity = 1.0 if ent.kind != "class" else 0.45
                    w = 3.5 * hit.depth_weight * specificity
                    direct_scores[ent.entity_id] += w
                    reasons_map[ent.entity_id].append(
                        f"traceback_line:{hit.rel_path}:{hit.line}(+{w:.2f})"
                    )
                elif hit.func_name and ent.name == hit.func_name:
                    w = 2.4 * hit.depth_weight
                    direct_scores[ent.entity_id] += w
                    reasons_map[ent.entity_id].append(
                        f"traceback_func:{hit.func_name}(+{w:.2f})"
                    )

        # 2. Mentioned files + exact symbol hits + IDF-weighted subtoken overlap
        for eid, ent in self.entities.items():
            if ent.rel_path in mentioned_files:
                w = 1.6 if not ent.is_test else 0.4
                direct_scores[eid] += w
                reasons_map[eid].append(f"file_mention:{ent.rel_path}(+{w:.2f})")

            if ent.name in exact_identifiers and ent.name not in STOPWORDS:
                w = 2.8 if ent.kind != "class" else 1.8
                direct_scores[eid] += w
                reasons_map[eid].append(f"exact_symbol:{ent.name}(+{w:.2f})")

            overlap = query_tokens & ent.identifiers
            if overlap:
                idf_sum = sum(self._idf(tok) for tok in overlap)
                # Normalize slightly by entity size to favor crisp fault functions
                len_norm = 1.0 + 0.15 * math.log1p(ent.span_lines)
                bm25_like = (idf_sum / len_norm) * 0.75
                if bm25_like > 0.05:
                    direct_scores[eid] += bm25_like
                    top_toks = ",".join(sorted(overlap)[:4])
                    reasons_map[eid].append(f"token_idf[{top_toks}](+{bm25_like:.2f})")

        # 3. Call-graph 2-hop propagation (both test->callee and symptom->callee/caller)
        graph_scores: dict[str, float] = defaultdict(float)
        # First hop from all directly scored entities (including test entities calling production!)
        for src_id, src_score in list(direct_scores.items()):
            if src_score <= 0.1:
                continue
            for callee_id in self.call_graph.get(src_id, ()):
                prop = src_score * 0.48
                graph_scores[callee_id] += prop
                reasons_map[callee_id].append(
                    f"called_by:{self.entities[src_id].name}(+{prop:.2f})"
                )
                for hop2_id in self.call_graph.get(callee_id, ()):
                    prop2 = src_score * 0.20
                    graph_scores[hop2_id] += prop2
                    reasons_map[hop2_id].append(
                        f"2hop_callee_via:{self.entities[callee_id].name}(+{prop2:.2f})"
                    )
            for caller_id in self.reverse_graph.get(src_id, ()):
                if not self.entities[caller_id].is_test:
                    prop_up = src_score * 0.22
                    graph_scores[caller_id] += prop_up
                    reasons_map[caller_id].append(
                        f"calls:{self.entities[src_id].name}(+{prop_up:.2f})"
                    )

        # Combine and rank deterministically
        ranked: list[SuspiciousEntity] = []
        file_scores: dict[str, float] = defaultdict(float)
        linked_tests: list[tuple[float, str]] = []

        for eid, ent in self.entities.items():
            d_score = direct_scores.get(eid, 0.0)
            g_score = graph_scores.get(eid, 0.0)
            total = d_score + g_score
            if total <= 0.0:
                continue
            if ent.is_test:
                linked_tests.append((total, ent.rel_path))
                if not include_tests:
                    continue
            # Slight preference for leaf methods/functions over whole class wrapper
            if ent.kind == "class":
                total *= 0.72
            ranked.append(
                SuspiciousEntity(
                    entity=ent,
                    score=round(total, 6),
                    direct_score=round(d_score, 6),
                    graph_score=round(g_score, 6),
                    reasons=reasons_map.get(eid, [])[:6],
                )
            )
            file_scores[ent.rel_path] = max(file_scores[ent.rel_path], round(total, 6))

        ranked.sort(
            key=lambda item: (-item.score, item.entity.rel_path, item.entity.start_line)
        )
        ranked_files = sorted(file_scores.items(), key=lambda kv: (-kv[1], kv[0]))

        seen_test_files: set[str] = set()
        ordered_tests: list[str] = []
        for _, tpath in sorted(linked_tests, key=lambda x: (-x[0], x[1])):
            if tpath not in seen_test_files:
                seen_test_files.add(tpath)
                ordered_tests.append(tpath)

        return LocalizationResult(
            ranked_entities=ranked[:top_k],
            ranked_files=ranked_files[:top_k],
            linked_tests=ordered_tests,
            traceback_hits=tb_hits,
            call_graph_edges={k: list(v) for k, v in self.call_graph.items() if v},
            total_entities_indexed=len(self.entities),
            total_files_indexed=self.file_count,
        )
