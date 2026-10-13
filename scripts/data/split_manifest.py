"""Generate reproducible explicit train/validation/test manifest assignments."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from multimodal.data_adapters.splits import generate_manifest_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sample-column", default="sample_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--source-column", default="source_id")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--allow-sample-level", action="store_true",
                        help="Allow stratified sample-level split if source IDs are unavailable")
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    summary = generate_manifest_splits(input_path, output_path, seed=args.seed,
        sample_column=args.sample_column, label_column=args.label_column,
        source_column=args.source_column, split_column=args.split_column,
        require_source_disjoint=not args.allow_sample_level)
    print(f"Wrote split manifest {summary['output_manifest']} ({summary['sample_count']} rows)")
    print(f"Summary: {Path(summary['output_manifest']).with_suffix(Path(summary['output_manifest']).suffix + '.split_summary.json')}")


if __name__ == "__main__":
    main()
