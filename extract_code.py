from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


def extract_code_from_notebook(notebook_path: Path, output_path: Path) -> int:
    """
    Export code cells from a Jupyter notebook into a plain Python script.

    Args:
        notebook_path: Path to the source .ipynb file.
        output_path: Target path for the generated .py file.

    Returns:
        Number of code cells written.
    """
    with notebook_path.open("r", encoding="utf-8") as handle:
        nb = nbformat.read(handle, as_version=4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_cells = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                handle.write(cell.get("source", "") + "\n\n")
                written_cells += 1

    return written_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract code cells from a notebook.")
    parser.add_argument("notebook", type=Path, help="Path to the input .ipynb notebook.")
    parser.add_argument(
        "output_script",
        type=Path,
        help="Destination path for the extracted Python script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written_cells = extract_code_from_notebook(args.notebook, args.output_script)
    print(f"=> Exported {written_cells} code cells to {args.output_script}")


if __name__ == "__main__":
    main()
