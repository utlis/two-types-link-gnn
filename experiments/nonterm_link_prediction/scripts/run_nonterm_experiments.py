from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


REPO_ROOT = resolve_repo_root()
sys.path.append(str(REPO_ROOT / "TermRGNNPipeline" / "scripts"))

from summarize import write_summary  # noqa: E402
from term_eval import evaluate_domains  # noqa: E402


DEFAULT_CONFIG: dict[str, Any] = {
    "run_name": None,
    "data_dir": "data",
    "use_embedding": False,
    "embedding_path": "data/cc.en.300.bin",
    "domains": ["Agriculture"],
    "ngram": "2gram",
    "seed": 42,
    "experiments": [
        {"name": "exp1_st_only", "use_nonterm_train": False},
        {"name": "exp2_st_sn", "use_nonterm_train": True},
    ],
    "negative_sampling": {
        "seed": 42,
        "ratio": 1,
        "explicit_policy": "match_positive",
        "hard_mode": "mixed",
        "corrupt_mode": "both",
    },
    "training": {
        "epochs": 100,
        "eval_every": 10,
        "seed": 42,
        "device": "auto",
        "encoder": "rgcn",
        "decoder": "complex",
        "compgcn_composition": "mult",
        "num_bases": None,
        "hidden_dim": 64,
        "embedding_dim": 32,
        "dropout": 0.2,
        "lr": 0.01,
        "loss_type": "bce_balanced",
        "weight_decay": 0.0001,
        "edge_dropout": 0.0,
        "label_smoothing": 0.0,
        "lr_scheduler": False,
        "no_final_model": True,
        "neg_resample_every": 0,
        "early_stopping_patience": 0,
    },
    "postprocess": {
        "cutoff": 5,
        "shortest_only": True,
        "reconstruction_mode": "predicted_term_links",
        "include_predicted_negatives": False,
        "composition_from_test": False,
    },
    "outputs": {
        "root": "experiments/nonterm_link_prediction/runs",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def normalize_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    config = deep_merge(DEFAULT_CONFIG, loaded)
    if not config["run_name"]:
        config["run_name"] = datetime.now().strftime("nonterm_%Y%m%d_%H%M%S")
    config["data_dir"] = str(normalize_path(config["data_dir"]))
    config["embedding_path"] = str(normalize_path(config["embedding_path"]))
    config["outputs"]["root"] = str(normalize_path(config["outputs"]["root"]))
    return config


def run_command(cmd: list[str], cwd: Path, log_path: Path, dry_run: bool) -> None:
    printable = " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd)
    print(f"  $ {printable}")
    if dry_run:
        return
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {printable}\n\n[stdout]\n{result.stdout}\n\n[stderr]\n{result.stderr}\n",
        encoding="utf-8",
    )
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).splitlines()[-20:])
        raise RuntimeError(f"Command failed. See {log_path}\n{tail}")


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ensure_dirs(exp_dir: Path) -> dict[str, Path]:
    paths = {
        "datasets": exp_dir / "rgnn" / "datasets",
        "models": exp_dir / "rgnn" / "models",
        "link_metrics": exp_dir / "rgnn" / "link_metrics",
        "postprocess": exp_dir / "rgnn" / "postprocess",
        "reconstructed_terms": exp_dir / "rgnn" / "reconstructed_terms",
        "term_eval": exp_dir / "term_eval",
        "summary": exp_dir / "summary",
        "logs": exp_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def create_dataset(
    *,
    config: dict[str, Any],
    experiment: dict[str, Any],
    domain: str,
    exp_paths: dict[str, Path],
    dry_run: bool,
) -> Path:
    dataset_dir = exp_paths["datasets"] / domain
    dataset_path = dataset_dir / "processed_dataset.pt"
    script = REPO_ROOT / "experiments" / "nonterm_link_prediction" / "scripts" / "create_nonterm_dataset.py"
    neg = config["negative_sampling"]
    cmd = [
        sys.executable,
        str(script),
        "--data_dir",
        config["data_dir"],
        "--domain",
        domain,
        "--ngram",
        str(config["ngram"]),
        "--experiment",
        str(experiment["name"]),
        "--output_path",
        str(dataset_path),
        "--metadata_path",
        str(dataset_dir / "metadata.json"),
        "--preview_dir",
        str(dataset_dir / "preview"),
        "--seed",
        str(config["seed"]),
        "--neg_seed",
        str(neg["seed"]),
        "--neg_ratio",
        str(neg["ratio"]),
        "--explicit_negative_policy",
        str(neg["explicit_policy"]),
        "--hard_negative_mode",
        str(neg["hard_mode"]),
        "--corrupt_mode",
        str(neg["corrupt_mode"]),
    ]
    if config.get("use_embedding", False):
        cmd.extend(["--use_embedding", "--embedding_path", str(config["embedding_path"])])
    if experiment.get("use_nonterm_train", False):
        cmd.append("--use_nonterm_train")
    run_command(cmd, REPO_ROOT, exp_paths["logs"] / f"create_dataset_{domain}.log", dry_run)
    return dataset_path


def train_domain(
    *,
    config: dict[str, Any],
    domain: str,
    dataset_path: Path,
    exp_paths: dict[str, Path],
    dry_run: bool,
) -> dict[str, Path]:
    train = config["training"]
    model_dir = exp_paths["models"] / domain
    metrics_dir = exp_paths["link_metrics"] / domain
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    best_model = model_dir / "best_model.pt"
    final_model = model_dir / "final_model.pt"
    test_metrics = metrics_dir / "test_metrics.csv"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "RGNN" / "train.py"),
        "--dataset_path",
        str(dataset_path),
        "--test_metrics_csv",
        str(test_metrics),
        "--checkpoint_path",
        str(best_model),
        "--final_model_path",
        str(final_model),
        "--epochs",
        str(train["epochs"]),
        "--eval_every",
        str(train["eval_every"]),
        "--seed",
        str(train["seed"]),
        "--device",
        resolve_device(str(train["device"])),
        "--encoder",
        str(train["encoder"]),
        "--decoder",
        str(train["decoder"]),
        "--compgcn_composition",
        str(train["compgcn_composition"]),
        "--hidden_dim",
        str(train["hidden_dim"]),
        "--embedding_dim",
        str(train["embedding_dim"]),
        "--lr",
        str(train["lr"]),
        "--loss_type",
        str(train["loss_type"]),
        "--weight_decay",
        str(train["weight_decay"]),
        "--dropout",
        str(train["dropout"]),
        "--edge_dropout",
        str(train["edge_dropout"]),
        "--label_smoothing",
        str(train["label_smoothing"]),
        "--neg_resample_every",
        str(train.get("neg_resample_every", 0)),
        "--early_stopping_patience",
        str(train.get("early_stopping_patience", 0)),
    ]
    if train.get("num_bases") is not None:
        cmd.extend(["--num_bases", str(train["num_bases"])])
    if train.get("lr_scheduler", False):
        cmd.append("--lr_scheduler")
    if train.get("no_final_model", True):
        cmd.append("--no_final_model")
    run_command(cmd, REPO_ROOT, exp_paths["logs"] / f"train_{domain}.log", dry_run)
    return {
        "best_model": best_model,
        "test_metrics": test_metrics,
        "trained_dataset": dataset_path.parent / "trained_dataset.pt",
    }


def load_threshold(trained_dataset: Path) -> float | None:
    try:
        import torch

        data = torch.load(trained_dataset, map_location="cpu", weights_only=False)
        value = data.get("optimal_threshold")
        return None if value is None else float(value)
    except Exception:
        return None


def postprocess_domain(
    *,
    config: dict[str, Any],
    domain: str,
    dataset_path: Path,
    train_outputs: dict[str, Path],
    exp_paths: dict[str, Path],
    dry_run: bool,
) -> Path:
    post = config["postprocess"]
    result_dir = exp_paths["postprocess"] / domain / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    threshold = load_threshold(train_outputs["trained_dataset"]) if train_outputs["trained_dataset"].exists() else None
    pos_cmd = [
        sys.executable,
        str(REPO_ROOT / "RGNN" / "pos_distribution.py"),
        "--dataset_path",
        str(dataset_path),
        "--model_path",
        str(train_outputs["best_model"]),
        "--output_csv",
        str(result_dir / "test_prob_hist.csv"),
        "--plot_pos_path",
        str(result_dir / "test_prob_hist_pos.png"),
        "--plot_neg_path",
        str(result_dir / "test_prob_hist_neg.png"),
        "--plot_overlay_path",
        str(result_dir / "test_prob_hist_overlay.png"),
        "--pos_recovered_csv",
        str(result_dir / "test_pos_recovered_edges.csv"),
        "--neg_recovered_csv",
        str(result_dir / "test_neg_recovered_edges.csv"),
        "--encoder",
        str(config["training"]["encoder"]),
        "--decoder",
        str(config["training"]["decoder"]),
        "--compgcn_composition",
        str(config["training"]["compgcn_composition"]),
        "--hidden_dim",
        str(config["training"]["hidden_dim"]),
        "--embedding_dim",
        str(config["training"]["embedding_dim"]),
        "--dropout",
        str(config["training"]["dropout"]),
        "--device",
        resolve_device(str(config["training"]["device"])),
    ]
    if config["training"].get("num_bases") is not None:
        pos_cmd.extend(["--num_bases", str(config["training"]["num_bases"])])
    if threshold is not None:
        pos_cmd.extend(["--threshold", str(threshold)])
    run_command(pos_cmd, REPO_ROOT, exp_paths["logs"] / f"pos_distribution_{domain}.log", dry_run)

    output_terms = exp_paths["reconstructed_terms"] / f"{domain}_reconstruct_terms.txt"
    rec_cmd = [
        sys.executable,
        str(REPO_ROOT / "RGNN" / "reconstruct_terms.py"),
        "--dataset_path",
        str(train_outputs["trained_dataset"]),
        "--output_path",
        str(output_terms),
        "--cutoff",
        str(post["cutoff"]),
    ]
    if post.get("shortest_only", True):
        rec_cmd.append("--shortest_only")
    if post["reconstruction_mode"] == "predicted_term_links":
        rec_cmd.append("--predicted_only")
    elif post["reconstruction_mode"] == "prediction_only_eval":
        rec_cmd.append("--prediction_only_eval")
    elif post["reconstruction_mode"] != "full_graph":
        raise ValueError(f"Unknown reconstruction_mode: {post['reconstruction_mode']}")
    if post.get("include_predicted_negatives", False):
        rec_cmd.append("--include_predicted_negatives")
    if post.get("composition_from_test", False):
        rec_cmd.append("--composition_from_test")
    run_command(rec_cmd, REPO_ROOT, exp_paths["logs"] / f"reconstruct_terms_{domain}.log", dry_run)
    return output_terms


def write_comparison(run_dir: Path, experiments: list[dict[str, Any]], domains: list[str]) -> None:
    rows: list[dict[str, str]] = []
    for experiment in experiments:
        exp_name = experiment["name"]
        summary_path = run_dir / exp_name / "summary" / "summary.csv"
        if not summary_path.exists():
            continue
        with summary_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["experiment"] = exp_name
                rows.append(row)

    comparison_dir = run_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    out = comparison_dir / "comparison_summary.csv"
    if not rows:
        out.write_text("", encoding="utf-8")
        return

    fieldnames = ["experiment", *[name for name in rows[0].keys() if name != "experiment"]]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    delta_out = comparison_dir / "metric_deltas.csv"
    by_key = {(row["domain"], row["experiment"]): row for row in rows}
    metrics = ["link_auc", "link_ap", "link_acc", "link_precision", "link_recall", "link_f1", "term_precision", "term_recall", "term_f1"]
    with delta_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", "metric", "exp1_st_only", "exp2_st_sn", "delta_exp2_minus_exp1"])
        writer.writeheader()
        for domain in domains:
            exp1 = by_key.get((domain, "exp1_st_only"), {})
            exp2 = by_key.get((domain, "exp2_st_sn"), {})
            for metric in metrics:
                try:
                    v1 = float(exp1.get(metric, ""))
                    v2 = float(exp2.get(metric, ""))
                    delta = v2 - v1
                    writer.writerow(
                        {
                            "domain": domain,
                            "metric": metric,
                            "exp1_st_only": f"{v1:.6f}",
                            "exp2_st_sn": f"{v2:.6f}",
                            "delta_exp2_minus_exp1": f"{delta:.6f}",
                        }
                    )
                except ValueError:
                    continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run St-only vs St+Sn RGNN experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    run_dir = Path(config["outputs"]["root"]) / config["run_name"]

    if run_dir.exists() and not args.resume and not args.dry_run:
        raise SystemExit(f"Run directory already exists: {run_dir}\nUse --resume or choose a new run_name.")
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, run_dir / "config.yaml")
        (run_dir / "config.resolved.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Run name: {config['run_name']}")
    print(f"Domains : {', '.join(config['domains'])}")
    print(f"N-gram  : {config['ngram']}")
    print(f"Output  : {run_dir}")

    for experiment in config["experiments"]:
        exp_name = experiment["name"]
        exp_dir = run_dir / exp_name
        print(f"\n=== {exp_name} ===")
        exp_paths = ensure_dirs(exp_dir) if not args.dry_run else {
            key: exp_dir / rel
            for key, rel in {
                "datasets": Path("rgnn/datasets"),
                "models": Path("rgnn/models"),
                "link_metrics": Path("rgnn/link_metrics"),
                "postprocess": Path("rgnn/postprocess"),
                "reconstructed_terms": Path("rgnn/reconstructed_terms"),
                "term_eval": Path("term_eval"),
                "summary": Path("summary"),
                "logs": Path("logs"),
            }.items()
        }

        for domain in config["domains"]:
            print(f"\n[1/3] Dataset: {domain}")
            dataset_path = create_dataset(config=config, experiment=experiment, domain=domain, exp_paths=exp_paths, dry_run=args.dry_run)
            print(f"[2/3] Train: {domain}")
            outputs = train_domain(config=config, domain=domain, dataset_path=dataset_path, exp_paths=exp_paths, dry_run=args.dry_run)
            print(f"[3/3] Postprocess: {domain}")
            postprocess_domain(config=config, domain=domain, dataset_path=dataset_path, train_outputs=outputs, exp_paths=exp_paths, dry_run=args.dry_run)

        if not args.dry_run:
            term_metrics_csv = exp_paths["term_eval"] / "term_metrics.csv"
            evaluate_domains(
                data_dir=Path(config["data_dir"]),
                reconstructed_terms_dir=exp_paths["reconstructed_terms"],
                domains=list(config["domains"]),
                ngram=str(config["ngram"]),
                output_csv=term_metrics_csv,
            )
            write_summary(
                run_name=f"{config['run_name']}_{exp_name}",
                ngram=str(config["ngram"]),
                domains=list(config["domains"]),
                link_metrics_dir=exp_paths["link_metrics"],
                term_metrics_csv=term_metrics_csv,
                output_csv=exp_paths["summary"] / "summary.csv",
            )

    if not args.dry_run:
        write_comparison(run_dir, list(config["experiments"]), list(config["domains"]))
        print(f"\nDone. Comparison: {run_dir / 'comparison' / 'comparison_summary.csv'}")
    else:
        print("\nDry run complete. No files were generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
