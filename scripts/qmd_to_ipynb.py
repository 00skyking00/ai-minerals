"""Convert a Quarto .qmd to a Jupyter .ipynb without needing the `quarto` CLI.

Handles the subset of Quarto syntax we use: YAML front matter, ```{python}
code fences with `#| label:` and `#| code-summary:` cell options, and
plain markdown between cells.

    uv run python scripts/qmd_to_ipynb.py notebooks/intro.qmd

Writes a sibling .ipynb with the same basename. Pairs well with `jupytext`
for tracking the .qmd as canonical source.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def qmd_to_notebook(qmd_text: str) -> dict:
    # Strip optional YAML front matter at the top of the file.
    if qmd_text.startswith("---\n"):
        end = qmd_text.find("\n---\n", 4)
        if end != -1:
            qmd_text = qmd_text[end + 5 :]

    cell_re = re.compile(r"```\{python\}\n(.*?)\n```", re.DOTALL)
    cells = []
    last = 0
    for match in cell_re.finditer(qmd_text):
        md_chunk = qmd_text[last : match.start()].strip("\n")
        if md_chunk.strip():
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": md_chunk.splitlines(keepends=True),
            })
        body = match.group(1)
        # Strip #| lines into cell metadata.
        lines = body.splitlines(keepends=True)
        meta: dict = {}
        while lines and lines[0].startswith("#|"):
            opt_line = lines.pop(0).rstrip()
            try:
                key, val = opt_line[2:].strip().split(":", 1)
                meta[key.strip()] = val.strip().strip('"')
            except ValueError:
                pass
        cells.append({
            "cell_type": "code",
            "metadata": meta,
            "execution_count": None,
            "outputs": [],
            "source": lines,
        })
        last = match.end()
    tail = qmd_text[last:].strip("\n")
    if tail.strip():
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": tail.splitlines(keepends=True),
        })

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    src = Path(argv[1])
    dst = src.with_suffix(".ipynb")
    nb = qmd_to_notebook(src.read_text())
    dst.write_text(json.dumps(nb, indent=1))
    print(f"{src} -> {dst} ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
