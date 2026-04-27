from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DOMAINS_DEFAULT = ["Agriculture", "Computer", "Dentistry", "Physics"]


@dataclass(frozen=True)
class PipelinePaths:
    repo_root: Path
    run_dir: Path

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def datasets_dir(self) -> Path:
        return self.run_dir / "rgnn" / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.run_dir / "rgnn" / "models"

    @property
    def link_metrics_dir(self) -> Path:
        return self.run_dir / "rgnn" / "link_metrics"

    @property
    def postprocess_dir(self) -> Path:
        return self.run_dir / "rgnn" / "postprocess"

    @property
    def reconstructed_terms_dir(self) -> Path:
        return self.run_dir / "rgnn" / "reconstructed_terms"

    @property
    def term_eval_dir(self) -> Path:
        return self.run_dir / "term_eval"

    @property
    def summary_dir(self) -> Path:
        return self.run_dir / "summary"


def ensure_output_dirs(paths: PipelinePaths) -> None:
    for directory in [
        paths.logs_dir,
        paths.datasets_dir,
        paths.models_dir,
        paths.link_metrics_dir,
        paths.postprocess_dir,
        paths.reconstructed_terms_dir,
        paths.term_eval_dir,
        paths.summary_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def find_domain_folders(data_dir: Path) -> list[str]:
    domains = []
    for item in data_dir.iterdir():
        if item.is_dir() and all((item / split).is_dir() for split in ["train", "dev", "test"]):
            domains.append(item.name)
    return sorted(domains)


def resolve_domains(data_dir: Path, requested: list[str] | None) -> list[str]:
    available = find_domain_folders(data_dir)
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise FileNotFoundError(f"Domain folders not found under {data_dir}: {missing}")
        return requested
    return available or DOMAINS_DEFAULT


def find_ngram_file(folder: Path, split: str, domain: str, ngram: str) -> Path:
    exact = folder / f"{split}_term_{domain}_{ngram}.txt"
    if exact.exists():
        return exact

    wanted = ngram.lower()
    candidates = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt"],
        key=lambda p: p.name.lower(),
    )
    for path in candidates:
        if wanted in path.name.lower():
            return path
    raise FileNotFoundError(f"No {ngram} file found in {folder}")


def run_command(cmd: list[str], cwd: Path, log_path: Path, dry_run: bool = False) -> None:
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


def write_run_config(config: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def create_dataset_for_domain(
    *,
    paths: PipelinePaths,
    data_dir: Path,
    domain: str,
    ngram: str,
    config: dict[str, Any],
    dry_run: bool,
) -> Path:
    domain_input = data_dir / domain
    train_path = find_ngram_file(domain_input / "train", "train", domain, ngram)
    dev_path = find_ngram_file(domain_input / "dev", "dev", domain, ngram)
    test_path = find_ngram_file(domain_input / "test", "test", domain, ngram)

    domain_dataset_dir = paths.datasets_dir / domain
    output_path = domain_dataset_dir / "processed_dataset.pt"
    if not dry_run:
        domain_dataset_dir.mkdir(parents=True, exist_ok=True)

    create_script = paths.repo_root / "RGNN" / "create_dataset.py"
    cmd = [
        sys.executable,
        str(create_script),
        "--train_path",
        str(train_path),
        "--dev_path",
        str(dev_path),
        "--test_path",
        str(test_path),
        "--output_path",
        str(output_path),
        "--seed",
        str(config["seed"]),
        "--neg_seed",
        str(config["negative_sampling"]["seed"]),
        "--neg_ratio",
        str(config["negative_sampling"]["ratio"]),
        "--hard_negative_mode",
        str(config["negative_sampling"]["hard_mode"]),
        "--corrupt_mode",
        str(config["negative_sampling"]["corrupt_mode"]),
    ]
    if config["use_embedding"]:
        cmd.extend(["--use_embedding", "--embedding_path", str(config["embedding_path"])])

    run_command(cmd, paths.repo_root, paths.logs_dir / f"create_dataset_{domain}.log", dry_run)
    return output_path


def train_domain(
    *,
    paths: PipelinePaths,
    domain: str,
    dataset_path: Path,
    config: dict[str, Any],
    dry_run: bool,
) -> dict[str, Path]:
    model_dir = paths.models_dir / domain
    metrics_dir = paths.link_metrics_dir / domain
    if not dry_run:
        model_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = config["training"]
    train_script = paths.repo_root / "RGNN" / "train.py"
    best_model = model_dir / "best_model.pt"
    final_model = model_dir / "final_model.pt"
    test_metrics = metrics_dir / "test_metrics.csv"

    device = train_cfg["device"]
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    cmd = [
        sys.executable,
        str(train_script),
        "--dataset_path",
        str(dataset_path),
        "--test_metrics_csv",
        str(test_metrics),
        "--checkpoint_path",
        str(best_model),
        "--final_model_path",
        str(final_model),
        "--epochs",
        str(train_cfg["epochs"]),
        "--eval_every",
        str(train_cfg["eval_every"]),
        "--seed",
        str(train_cfg["seed"]),
        "--device",
        str(device),
        "--encoder",
        str(train_cfg["encoder"]),
        "--decoder",
        str(train_cfg["decoder"]),
        "--compgcn_composition",
        str(train_cfg["compgcn_composition"]),
        "--hidden_dim",
        str(train_cfg["hidden_dim"]),
        "--embedding_dim",
        str(train_cfg["embedding_dim"]),
        "--lr",
        str(train_cfg["lr"]),
    ]
    if not train_cfg["save_final_model"]:
        cmd.append("--no_final_model")
    if train_cfg["lr_scheduler"]:
        cmd.append("--lr_scheduler")

    run_command(cmd, paths.repo_root, paths.logs_dir / f"train_{domain}.log", dry_run)
    return {
        "best_model": best_model,
        "final_model": final_model,
        "test_metrics": test_metrics,
        "trained_dataset": dataset_path.parent / "trained_dataset.pt",
    }


def load_optimal_threshold(trained_dataset: Path) -> float | None:
    try:
        import torch

        dataset = torch.load(trained_dataset, map_location="cpu", weights_only=False)
        value = dataset.get("optimal_threshold")
        return None if value is None else float(value)
    except Exception:
        return None


def postprocess_domain(
    *,
    paths: PipelinePaths,
    domain: str,
    dataset_path: Path,
    trained_dataset: Path,
    best_model: Path,
    config: dict[str, Any],
    dry_run: bool,
) -> Path:
    post_cfg = config["postprocess"]
    result_dir = paths.postprocess_dir / domain / "result"
    if not dry_run:
        result_dir.mkdir(parents=True, exist_ok=True)

    pos_script = paths.repo_root / "RGNN" / "pos_distribution.py"
    threshold = load_optimal_threshold(trained_dataset) if trained_dataset.exists() else None
    pos_cmd = [
        sys.executable,
        str(pos_script),
        "--dataset_path",
        str(dataset_path),
        "--model_path",
        str(best_model),
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
    ]
    if threshold is not None:
        pos_cmd.extend(["--threshold", str(threshold)])
    pos_cmd.extend([
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
    ])
    run_command(pos_cmd, paths.repo_root, paths.logs_dir / f"pos_distribution_{domain}.log", dry_run)

    reconstruct_script = paths.repo_root / "RGNN" / "reconstruct_terms.py"
    reconstruct_dataset = trained_dataset if post_cfg["use_trained_dataset"] else dataset_path
    output_terms = paths.reconstructed_terms_dir / f"{domain}_reconstruct_terms.txt"
    rec_cmd = [
        sys.executable,
        str(reconstruct_script),
        "--dataset_path",
        str(reconstruct_dataset),
        "--output_path",
        str(output_terms),
        "--cutoff",
        str(post_cfg["cutoff"]),
    ]
    if post_cfg["shortest_only"]:
        rec_cmd.append("--shortest_only")

    mode = post_cfg["reconstruction_mode"]
    if mode == "predicted_term_links":
        rec_cmd.append("--predicted_only")
    elif mode == "prediction_only_eval":
        rec_cmd.append("--prediction_only_eval")
    elif mode != "full_graph":
        raise ValueError(f"Unknown reconstruction_mode: {mode}")

    if post_cfg["include_predicted_negatives"]:
        rec_cmd.append("--include_predicted_negatives")
    if post_cfg["composition_from_test"]:
        rec_cmd.append("--composition_from_test")

    run_command(rec_cmd, paths.repo_root, paths.logs_dir / f"reconstruct_terms_{domain}.log", dry_run)
    return output_terms


def read_link_metrics(path: Path, relation_type: str) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("relation_type") == relation_type:
                return row
    return {}


def read_overall_link_metrics(path: Path) -> dict[str, str]:
    return read_link_metrics(path, "overall")
