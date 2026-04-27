import argparse
import csv
import subprocess
import sys
from pathlib import Path


def find_datasets(dataset_root):
    return sorted(Path(dataset_root).rglob("processed_dataset.pt"))


def read_metrics_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Batch-train RGNN and aggregate metrics")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(script_dir / "data" / "dataset"),
        help="Root directory containing dataset folders",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to aggregated metrics CSV (default: <dataset_root>/metrics.csv)",
    )
    parser.add_argument(
        "--train_script",
        type=str,
        default=str(script_dir / "train.py"),
        help="Path to train.py",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to train.py (prefix with --; leading -- is ignored)",
    )

    args = parser.parse_args()

    if args.train_args and args.train_args[0] == "--":
        args.train_args = args.train_args[1:]

    if args.output_csv is None:
        args.output_csv = str(Path(args.dataset_root) / "metrics.csv")

    dataset_paths = find_datasets(args.dataset_root)
    if not dataset_paths:
        print(f"No processed_dataset.pt found under {args.dataset_root}")
        return 1

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    aggregated_rows = []
    for dataset_path in dataset_paths:
        dataset_dir = dataset_path.parent
        dataset_name = dataset_dir.name
        test_metrics_path = dataset_dir / "test_metrics.csv"
        best_model_path = dataset_dir / "best_model.pt"

        cmd = [
            sys.executable,
            args.train_script,
            "--dataset_path",
            str(dataset_path),
            "--test_metrics_csv",
            str(test_metrics_path),
            "--checkpoint_path",
            str(best_model_path),
            "--no_final_model",
        ] + args.train_args

        print(f"\n[train] {dataset_name}")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        if not test_metrics_path.exists():
            raise FileNotFoundError(f"Metrics not found: {test_metrics_path}")

        rows = read_metrics_csv(test_metrics_path)
        for row in rows:
            aggregated_rows.append(
                {
                    "dataset": dataset_name,
                    "dataset_path": str(dataset_path),
                    "relation_type": row.get("relation_type", ""),
                    "auc": row.get("auc", ""),
                    "ap": row.get("ap", ""),
                    "acc": row.get("acc", ""),
                    "precision": row.get("precision", ""),
                    "recall": row.get("recall", ""),
                    "f1": row.get("f1", ""),
                    "opt_threshold": row.get("opt_threshold", ""),
                    "mrr": row.get("mrr", ""),
                    "hits_at_1": row.get("hits_at_1", ""),
                    "hits_at_5": row.get("hits_at_5", ""),
                    "hits_at_10": row.get("hits_at_10", ""),
                }
            )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "dataset_path",
                "relation_type",
                "auc",
                "ap",
                "acc",
                "precision",
                "recall",
                "f1",
                "opt_threshold",
                "mrr",
                "hits_at_1",
                "hits_at_5",
                "hits_at_10",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregated_rows)

    print(f"\nAggregated metrics saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
