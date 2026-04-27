from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


def load_terms(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def calculate_metrics(ground_truth_path: Path, result_path: Path) -> dict[str, object]:
    gt_terms = load_terms(ground_truth_path)
    result_terms = load_terms(result_path)
    true_positives = len(gt_terms.intersection(result_terms))

    precision = true_positives / len(result_terms) if result_terms else 0.0
    recall = true_positives / len(gt_terms) if gt_terms else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "len_gt": len(gt_terms),
        "len_result": len(result_terms),
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def find_ngram_file(folder: Path, split: str, domain: str, ngram: str) -> Path:
    exact = folder / f"{split}_term_{domain}_{ngram}.txt"
    if exact.exists():
        return exact
    wanted = ngram.lower()
    for path in sorted(folder.glob("*.txt"), key=lambda p: p.name.lower()):
        if wanted in path.name.lower():
            return path
    raise FileNotFoundError(f"No {ngram} test file found in {folder}")


def evaluate_domains(
    *,
    data_dir: Path,
    reconstructed_terms_dir: Path,
    domains: list[str],
    ngram: str,
    output_csv: Path,
) -> list[dict[str, object]]:
    rows = []
    for domain in domains:
        gt_path = find_ngram_file(data_dir / domain / "test", "test", domain, ngram)
        result_path = reconstructed_terms_dir / f"{domain}_reconstruct_terms.txt"
        if not result_path.exists():
            raise FileNotFoundError(f"Reconstructed terms not found: {result_path}")

        metrics = calculate_metrics(gt_path, result_path)
        rows.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "domain": domain,
                "ngram": ngram,
                "ground_truth_file": str(gt_path),
                "result_file": str(result_path),
                **metrics,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "domain",
                "ngram",
                "ground_truth_file",
                "result_file",
                "len_gt",
                "len_result",
                "true_positives",
                "precision",
                "recall",
                "f1_score",
            ],
        )
        writer.writeheader()
        for row in rows:
            formatted = row.copy()
            for key in ["precision", "recall", "f1_score"]:
                formatted[key] = f"{float(formatted[key]):.6f}"
            writer.writerow(formatted)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate reconstructed terms against test terms.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--reconstructed-terms-dir", required=True)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--ngram", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    evaluate_domains(
        data_dir=Path(args.data_dir),
        reconstructed_terms_dir=Path(args.reconstructed_terms_dir),
        domains=args.domains,
        ngram=args.ngram,
        output_csv=Path(args.out),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
