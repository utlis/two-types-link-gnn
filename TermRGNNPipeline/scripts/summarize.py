from __future__ import annotations

import csv
from pathlib import Path

from stages import read_link_metrics, read_overall_link_metrics


def _read_term_metrics(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["domain"]: row for row in csv.DictReader(f)}


def write_summary(
    *,
    run_name: str,
    ngram: str,
    domains: list[str],
    link_metrics_dir: Path,
    term_metrics_csv: Path,
    output_csv: Path,
) -> None:
    term_metrics = _read_term_metrics(term_metrics_csv)
    rows = []
    for domain in domains:
        link_metrics_path = link_metrics_dir / domain / "test_metrics.csv"
        link = read_overall_link_metrics(link_metrics_path)
        type2_link = read_link_metrics(link_metrics_path, "2")
        term = term_metrics.get(domain, {})
        rows.append(
            {
                "run_name": run_name,
                "domain": domain,
                "ngram": ngram,
                "link_auc": link.get("auc", ""),
                "link_ap": link.get("ap", ""),
                "link_acc": link.get("acc", ""),
                "link_precision": link.get("precision", ""),
                "link_recall": link.get("recall", ""),
                "link_f1": link.get("f1", ""),
                "link_opt_threshold": link.get("opt_threshold", ""),
                "link_mrr": link.get("mrr", ""),
                "link_hits_at_1": link.get("hits_at_1", ""),
                "link_hits_at_5": link.get("hits_at_5", ""),
                "link_hits_at_10": link.get("hits_at_10", ""),
                "type2_mrr": type2_link.get("mrr", ""),
                "type2_hits_at_1": type2_link.get("hits_at_1", ""),
                "type2_hits_at_5": type2_link.get("hits_at_5", ""),
                "type2_hits_at_10": type2_link.get("hits_at_10", ""),
                "term_len_gt": term.get("len_gt", ""),
                "term_len_result": term.get("len_result", ""),
                "term_true_positives": term.get("true_positives", ""),
                "term_precision": term.get("precision", ""),
                "term_recall": term.get("recall", ""),
                "term_f1": term.get("f1_score", ""),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "domain",
                "ngram",
                "link_auc",
                "link_ap",
                "link_acc",
                "link_precision",
                "link_recall",
                "link_f1",
                "link_opt_threshold",
                "link_mrr",
                "link_hits_at_1",
                "link_hits_at_5",
                "link_hits_at_10",
                "type2_mrr",
                "type2_hits_at_1",
                "type2_hits_at_5",
                "type2_hits_at_10",
                "term_len_gt",
                "term_len_result",
                "term_true_positives",
                "term_precision",
                "term_recall",
                "term_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
