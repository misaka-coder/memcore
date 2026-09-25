# MemCore v0.1.0 launch checklist

This checklist keeps the first public release reproducible and keeps package
publishing credentials out of the repository.

## Distribution name

- GitHub project: `memcore`
- PyPI distribution: `memcore-runtime`
- Python import package: `memcore`
- Release version: `0.1.0`
- Git tag: `v0.1.0`

The PyPI distribution name and Python import name are intentionally different.
Users install `memcore-runtime` and continue to write `import memcore`.

> A PyPI pending publisher does **not** reserve a name. Re-check that
> `memcore-runtime` is still available immediately before the first publish.

## One-time PyPI Trusted Publishing setup

Before publishing the first GitHub Release, create a **pending GitHub publisher**
in the PyPI account that should own the project.

Use exactly:

| Field | Value |
| --- | --- |
| PyPI project name | `memcore-runtime` |
| GitHub owner | `misaka-coder` |
| Repository | `memcore` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

The repository workflow uses OIDC through
`pypa/gh-action-pypi-publish@release/v1`; no long-lived PyPI API token is
required.

For extra protection, create a GitHub Environment named `pypi` and require
manual approval before deployment.

## Pre-release gate

Run locally:

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
python -m build
python examples/settlement_demo.py
```

Verify the built wheel is named like:

```text
memcore_runtime-0.1.0-py3-none-any.whl
```

and verify the installed import remains:

```python
import memcore
print(memcore.__version__)
```

## Publish

1. Merge the launch-readiness PR and wait for CI to pass on `main`.
2. Re-check the `memcore-runtime` name on PyPI.
3. Configure the pending publisher above.
4. Create GitHub tag `v0.1.0`.
5. Publish a GitHub Release from `v0.1.0`.
6. `.github/workflows/publish.yml` builds the distributions and publishes them
   to PyPI through Trusted Publishing.
7. Verify `pip install memcore-runtime` in a clean environment.
8. Change README wording from "after the first PyPI release" to the normal
   installation command if desired.

## 60-second demo storyboard

The executable source is `examples/settlement_demo.py`.

- **0–10s — Problem:** show a large shell/web result entering the active turn.
- **10–25s — Full evidence:** run the demo and highlight that the open turn still
  contains the complete raw payload.
- **25–40s — Settlement:** show the same history after `complete_turn()`; the
  large body has become a small `[compact_reloadable]` card.
- **40–53s — Readback:** call `open_memory` and show the exact original root
  cause returning.
- **53–60s — Thesis:** "Keep full evidence when it matters. Page it out when it
  doesn't. Reload it when needed." End on the GitHub repository URL.

Record the terminal at a readable font size; do not scroll through the full raw
payload. The value proposition is the transition **full → card → exact raw**.
