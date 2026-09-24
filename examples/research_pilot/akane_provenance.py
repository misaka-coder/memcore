"""Pin public source and synthetic inputs for the actual-Akane retest.

Only allowlisted code and bundled prompt resources enter this manifest. Runtime
stores, logs, environment files and model caches are never copied or hashed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

from .runner import digest, write_json


def source_files(memcore_root: Path, akane_root: Path, pack_root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for name, root, patterns in (
        (
            "memcore",
            memcore_root,
            (
                "memcore/**/*.py",
                "examples/research_pilot/*.py",
                "tests/test_research_pilot_akane_*.py",
                "pyproject.toml",
            ),
        ),
        (
            "akane",
            akane_root,
            (
                "config.py",
                "akane_paths.py",
                "companion_v01/**/*.py",
                "services/**/*.py",
                "companion_v01/persona_profiles.toml",
                "desktop_pet_creator_kit/characters/akane_v1/persona.md",
                "desktop_pet_creator_kit/characters/akane_v1/character.toml",
                "desktop_pet_creator_kit/characters/akane_v1/character.json",
                "pyproject.toml",
            ),
        ),
        ("pack", pack_root, ("scenarios.json", "answer_key.json", "run_manifest.json")),
    ):
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file() and path.resolve().is_relative_to(root.resolve()):
                    files[f"{name}/{path.relative_to(root).as_posix()}"] = path
    for package in ("capcore", "capcore_provider_openai"):
        spec = importlib.util.find_spec(package)
        if spec is not None:
            for location in spec.submodule_search_locations or ():
                package_root = Path(location).resolve()
                for path in package_root.rglob("*.py"):
                    if path.resolve().is_relative_to(package_root):
                        files[f"dependencies/{package}/{path.relative_to(package_root).as_posix()}"] = path
    for required in ("akane/config.py", "akane/companion_v01/engine.py", "pack/scenarios.json"):
        if required not in files:
            raise RuntimeError("required_retest_source_missing")
    return dict(sorted(files.items()))


def fingerprint(files: dict[str, Path]) -> dict[str, Any]:
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    return {"file_hashes": hashes, "source_fingerprint": digest(hashes)}


def freeze_sources(memcore_root: Path, akane_root: Path, pack_root: Path, output: Path) -> dict[str, Any]:
    """Preserve source evidence without pretending dependencies are vendored."""
    files = source_files(memcore_root, akane_root, pack_root)
    result = {
        "format": "akane_actual_host_source_evidence_v2",
        "scope": "allowlisted_source_and_bundled_prompt_resources",
        "execution": "source_verified_original_packages_in_existing_akane_environment",
        "dependency_binaries_frozen": False,
        **fingerprint(files),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, path in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json(output / "source_evidence.json", result)
    return result


def verify_sources(expected: dict[str, Any], memcore_root: Path, akane_root: Path, pack_root: Path) -> None:
    current = fingerprint(source_files(memcore_root, akane_root, pack_root))
    if current["file_hashes"] != expected["file_hashes"]:
        raise RuntimeError("retest_source_changed")


def load_pack(pack_root: Path) -> dict[str, Any]:
    """Load model-facing scenarios and run plan; evaluator gold stays unopened."""
    scenarios = json.loads((pack_root / "scenarios.json").read_text(encoding="utf-8"))
    manifest = json.loads((pack_root / "run_manifest.json").read_text(encoding="utf-8"))
    if scenarios.get("synthetic") is not True or len(manifest.get("runs", [])) != 4:
        raise RuntimeError("invalid_actual_host_retest_pack")
    if set(manifest["condition_definitions"]) != {"full", "card"}:
        raise RuntimeError("invalid_actual_host_retest_conditions")
    if sum(step["stage"] == "probe" for case in scenarios["scenarios"] for step in case["steps"]) != 6:
        raise RuntimeError("invalid_actual_host_retest_probe_count")
    return {"scenarios": scenarios, "manifest": manifest}
