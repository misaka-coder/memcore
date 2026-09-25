"""Release metadata invariants."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import memcore


ROOT = Path(__file__).resolve().parents[1]


def project_field(name: str) -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = text.split("[project]", 1)[1].split("[", 1)[0]
    match = re.search(rf'^\\s*{re.escape(name)}\\s*=\\s*"([^"]+)"', project, re.MULTILINE)
    if match is None:
        raise AssertionError(f"missing [project].{name}")
    return match.group(1)


class ReleaseMetadataTests(unittest.TestCase):
    def test_distribution_name_is_public_release_name(self) -> None:
        self.assertEqual(project_field("name"), "memcore-runtime")

    def test_package_version_matches_distribution_version(self) -> None:
        self.assertEqual(project_field("version"), memcore.__version__)


if __name__ == "__main__":
    unittest.main()
