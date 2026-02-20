import argparse
import json
from pathlib import Path


MARKER = "# AUTO-GENERATED: hardcoded env vars for standalone rerun"


def build_cell(model_key: str, model_name: str, model_signal_dir: str, returns_mode: str) -> dict:
    source = [
        f"{MARKER}\n",
        "import os\n",
        f"os.environ['MODEL_KEY'] = '{model_key}'\n",
        f"os.environ['MODEL_NAME'] = '{model_name}'\n",
        f"os.environ['MODEL_SIGNAL_DIR'] = '{model_signal_dir}'\n",
        f"os.environ['RETURNS_MODE'] = '{returns_mode}'\n",
    ]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject hardcoded environment variables into a report notebook."
    )
    parser.add_argument("--notebook", required=True, help="Notebook path to edit in place.")
    parser.add_argument("--model-key", required=True, help="MODEL_KEY value.")
    parser.add_argument("--model-name", required=True, help="MODEL_NAME value.")
    parser.add_argument(
        "--model-signal-dir", required=True, help="MODEL_SIGNAL_DIR value."
    )
    parser.add_argument(
        "--returns-mode", required=True, choices=["hedged", "unhedged"],
        help="RETURNS_MODE value."
    )
    args = parser.parse_args()

    notebook_path = Path(args.notebook)
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    new_cell = build_cell(args.model_key, args.model_name, args.model_signal_dir, args.returns_mode)
    cells = nb.get("cells", [])

    if cells:
        first = cells[0]
        first_source = "".join(first.get("source", []))
        if first.get("cell_type") == "code" and MARKER in first_source:
            cells[0] = new_cell
        else:
            cells.insert(0, new_cell)
    else:
        cells = [new_cell]

    nb["cells"] = cells
    with notebook_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
