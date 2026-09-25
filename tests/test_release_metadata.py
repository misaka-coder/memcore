"""Release metadata invariants."""

from __future__ import annotations

import unittest
from pathlib import Path

import memcore


ROOT = Path(__file__).resolve().parents[1]


def project_field(name: str) -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = text.split("[project]", 1)[1].split("[", 1)[0]
    prefix = f"{name} = "
    for line in project.splitlines():
        if line.strip().startswith(prefix):
            value = line.split("=", 1)[1].strip()
            if value.startswith('"') and value.endswith('"'):
                return value[1:-1]
    raise AssertionError(f"missing [project].{name}")


class ReleaseMetadataTests(unittest.TestCase):
    def test_distribution_name_is_public_release_name(self) -> None:
        self.assertEqual(project_field("name"), "memcore-runtime")

    def test_package_version_matches_distribution_version(self) -> None:
        self.assertEqual(project_field("version"), memcore.__version__)


if __name__ == "__main__":
    unittest.main()
