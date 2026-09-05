"""Validate a Jupyter notebook and execute its Python code cells in order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    if notebook.get("nbformat") != 4:
        raise ValueError("expected Jupyter notebook format 4")
    if not isinstance(notebook.get("cells"), list):
        raise ValueError("notebook cells must be a list")

    namespace: dict[str, object] = {"__name__": "__notebook__"}
    code_count = 0
    for cell_index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        code_count += 1
        source = cell.get("source")
        if not isinstance(source, str):
            raise TypeError(f"cell {cell_index} source must be a string")
        compiled = compile(source, f"{notebook_path.name}:cell-{cell_index}", "exec")
        exec(compiled, namespace)

    print(
        f"Notebook validation passed: {code_count} code cells, "
        f"{len(notebook['cells'])} total cells."
    )


if __name__ == "__main__":
    main()
