import argparse
import json
from pathlib import Path


MARKER = "# AUTO-GENERATED: hardcoded env vars for standalone rerun"


def build_cell(
    model_key: str,
    model_name: str,
    model_signal_dir: str,
    returns_mode: str,
    model_train_dir: str | None = None,
    date_filter_mode: str | None = None,
    reference_signal_dir: str | None = None,
    require_reference_overlap: str | None = None,
) -> dict:
    source = [
        f"{MARKER}\n",
        "import os\n",
        f"os.environ['MODEL_KEY'] = '{model_key}'\n",
        f"os.environ['MODEL_NAME'] = '{model_name}'\n",
        f"os.environ['MODEL_SIGNAL_DIR'] = '{model_signal_dir}'\n",
        f"os.environ['RETURNS_MODE'] = '{returns_mode}'\n",
    ]
    if model_train_dir:
        source.append(f"os.environ['MODEL_TRAIN_DIR'] = '{model_train_dir}'\n")
    if date_filter_mode:
        source.append(f"os.environ['DATE_FILTER_MODE'] = '{date_filter_mode}'\n")
    if reference_signal_dir:
        source.append(f"os.environ['REFERENCE_SIGNAL_DIR'] = '{reference_signal_dir}'\n")
    if require_reference_overlap:
        source.append(f"os.environ['REQUIRE_REFERENCE_OVERLAP'] = '{require_reference_overlap}'\n")
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
    parser.add_argument(
        "--model-train-dir",
        default="",
        help="Optional MODEL_TRAIN_DIR value for eval date filtering.",
    )
    parser.add_argument(
        "--date-filter-mode",
        default="",
        help="Optional DATE_FILTER_MODE value for eval date filtering.",
    )
    parser.add_argument(
        "--reference-signal-dir",
        default="",
        help="Optional REFERENCE_SIGNAL_DIR for overlap-based date filtering.",
    )
    parser.add_argument(
        "--require-reference-overlap",
        default="",
        help="Optional REQUIRE_REFERENCE_OVERLAP flag value (e.g. 1).",
    )
    args = parser.parse_args()

    notebook_path = Path(args.notebook)
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    new_cell = build_cell(
        args.model_key,
        args.model_name,
        args.model_signal_dir,
        args.returns_mode,
        model_train_dir=(args.model_train_dir or None),
        date_filter_mode=(args.date_filter_mode or None),
        reference_signal_dir=(args.reference_signal_dir or None),
        require_reference_overlap=(args.require_reference_overlap or None),
    )
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
