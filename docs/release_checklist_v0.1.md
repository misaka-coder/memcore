# v0.1 Launch Checklist

This file tracks the public-release work that should be true before announcing MemCore broadly.

## Already prepared in the repository

- Apache-2.0 license.
- CI on Python 3.10 and 3.13.
- Unit tests, Ruff lint/format checks, package build, and the settlement smoke demo run in CI.
- Zero-API-key `examples/settlement_demo.py`.
- Source-install instructions in both READMEs.
- Public package metadata normalized to `0.1.0`.

## PyPI distribution name

The Python import package remains:

```python
import memcore
```

The distribution name on PyPI does not have to match the import name. The obvious names
`memcore`, `memcore-ai`, and `memcore-sdk` are already in use by other projects.

Current launch candidate: **`memcore-runtime`**.

Do not treat the candidate as reserved until the first upload succeeds. Re-check PyPI immediately
before publishing because package names can be claimed at any time.

When the name is finalized, update:

1. `[project].name` in `pyproject.toml`.
2. The install command in `README.md` and `README_EN.md`.
3. Release notes and any launch posts.

## Release gate

Before creating `v0.1.0`:

```bash
python -m unittest discover -s tests -v
ruff check . --select E9,F63,F7,F82
python -m build
python examples/settlement_demo.py
```

The repository currently has pre-existing non-critical Ruff/format debt. CI reports the full Ruff
result without making that historical cleanup a v0.1 release blocker; syntax/undefined-name class
errors remain blocking through the critical-lint step.

Then inspect both wheel and sdist contents and verify that no research-only or host-private files are
included.

## Suggested first release

- Tag: `v0.1.0`
- Title: `MemCore v0.1.0 — Context runtime for long-horizon agents`
- Primary demo: terminal settlement of a large tool result followed by exact `open_memory` reload.
- Positioning: context residency management, not another vector-memory layer.

Publishing to PyPI and creating the GitHub Release are intentionally separate actions from this
checklist so they can be performed only after the final distribution name is confirmed.
