"""Build and strictly validate ADK submission.zip for Kaggle Gemma 4 Developer Agent."""

from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path

import yaml


class _IncludeLoader(yaml.SafeLoader):
    """YAML SafeLoader supporting Google ADK !include tags relative to bundle root."""

    def __init__(self, stream, base_dir: Path) -> None:
        self._base_dir = base_dir
        super().__init__(stream)


def _construct_include(loader: _IncludeLoader, node: yaml.Node):
    rel_path = loader.construct_scalar(node)
    target = (loader._base_dir / rel_path).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"!include target missing: {rel_path} ({target})")
    text = target.read_text(encoding="utf-8")
    if target.suffix in (".yaml", ".yml"):
        return load_adk_yaml(target, loader._base_dir)
    return text


_IncludeLoader.add_constructor("!include", _construct_include)


def load_adk_yaml(yaml_path: Path, base_dir: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as fh:
        loader = _IncludeLoader(fh, base_dir=base_dir)
        try:
            data = loader.get_single_data()
        finally:
            loader.dispose()
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {yaml_path}")
    return data


def validate_and_pack_submission(bundle_dir: Path, output_zip: Path) -> dict:
    bundle_dir = bundle_dir.resolve()
    output_zip = output_zip.resolve()
    agent_yaml = bundle_dir / "agent.yaml"
    if not agent_yaml.is_file():
        raise FileNotFoundError(f"Missing required root agent.yaml in {bundle_dir}")

    agent_cfg = load_adk_yaml(agent_yaml, base_dir=bundle_dir)
    required_keys = {"name", "model", "instruction", "tools", "generate_content_config"}
    missing = required_keys - set(agent_cfg.keys())
    if missing:
        raise ValueError(f"agent.yaml missing required ADK fields: {sorted(missing)}")

    allowed_models = {
        "gemma-4-31b-it-qat-w4a16-ct",
        "gemma-4-12b-it-qat-w4a16-ct",
        "gemma-4-e4b-it-qat-w4a16-ct",
        "gemma-4-e2b-it-qat-w4a16-ct",
    }
    if agent_cfg["model"] not in allowed_models:
        raise ValueError(f"Unexpected Gemma 4 model variant: {agent_cfg['model']}")

    gen_cfg = agent_cfg.get("generate_content_config") or {}
    if "thinking_budget" in gen_cfg or (
        isinstance(gen_cfg.get("thinking_config"), dict)
        and "thinking_budget" in gen_cfg["thinking_config"]
    ):
        raise ValueError("Deprecated 'thinking_budget' detected in sampling config!")

    eval_yaml = bundle_dir / "eval_config.yaml"
    eval_cfg = (
        load_adk_yaml(eval_yaml, base_dir=bundle_dir) if eval_yaml.is_file() else {}
    )

    packed_files: list[str] = []
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(bundle_dir).as_posix()
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            zf.write(path, arcname=rel)
            packed_files.append(rel)

    with zipfile.ZipFile(output_zip, "r") as zf:
        names = set(zf.namelist())
        if "agent.yaml" not in names:
            raise AssertionError("submission.zip missing root agent.yaml")

    return {
        "status": "VALID_SUBMISSION_ZIP",
        "zip_path": str(output_zip),
        "zip_bytes": output_zip.stat().st_size,
        "agent_name": agent_cfg["name"],
        "model": agent_cfg["model"],
        "tools_count": len(agent_cfg["tools"]),
        "eval_config": eval_cfg,
        "packed_files": packed_files,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    summary = validate_and_pack_submission(
        bundle_dir=root / "submission_bundle",
        output_zip=root / "submission.zip",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
