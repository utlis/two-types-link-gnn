from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from stages import (
    PipelinePaths,
    create_dataset_for_domain,
    ensure_output_dirs,
    postprocess_domain,
    resolve_domains,
    train_domain,
    write_run_config,
)
from summarize import write_summary
from term_eval import evaluate_domains


DEFAULT_CONFIG: dict[str, Any] = {
    "run_name": None,
    "data_dir": "data",
    "ngram": "all",
    "domains": None,
    "use_embedding": False,
    "embedding_path": "data/cc.en.300.bin",
    "seed": 42,
    "negative_sampling": {
        "seed": 42,
        "ratio": 1,
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
        "hidden_dim": 64,
        "embedding_dim": 32,
        "lr": 0.01,
        "lr_scheduler": False,
        "save_final_model": False,
    },
    "postprocess": {
        "cutoff": 5,
        "use_trained_dataset": True,
        "shortest_only": True,
        "reconstruction_mode": "predicted_term_links",
        "include_predicted_negatives": False,
        "composition_from_test": False,
    },
    "outputs": {
        "root": "TermRGNNPipeline/runs",
    },
}


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none", "~"}:
        return None
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> dict[str, Any]:
    # Small YAML reader for the repository config shape.
    root: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if indent == 0:
            key, _, value = stripped.partition(":")
            if value.strip():
                root[key] = parse_scalar(value)
                current_key = None
            else:
                root[key] = {}
                current_key = key
            continue

        if current_key is None:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        if stripped.startswith("- "):
            if not isinstance(root[current_key], list):
                root[current_key] = []
            root[current_key].append(parse_scalar(stripped[2:]))
            continue

        key, _, value = stripped.partition(":")
        if not isinstance(root[current_key], dict):
            raise ValueError(f"Unsupported YAML nesting under {current_key}")
        root[current_key][key] = parse_scalar(value)
    return root


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return load_simple_yaml(path)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key in ["run_name", "data_dir", "ngram", "embedding_path"]:
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    if args.domains:
        overrides["domains"] = args.domains
    if args.use_embedding:
        overrides["use_embedding"] = True
    if args.output_root is not None:
        overrides["outputs"] = {"root": args.output_root}

    training = {}
    for key in ["epochs", "eval_every", "device", "seed"]:
        value = getattr(args, key)
        if value is not None:
            training[key] = value
    if training:
        overrides["training"] = training

    postprocess = {}
    if args.cutoff is not None:
        postprocess["cutoff"] = args.cutoff
    if args.reconstruction_mode is not None:
        postprocess["reconstruction_mode"] = args.reconstruction_mode
    if postprocess:
        overrides["postprocess"] = postprocess
    return deep_merge(config, overrides)


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_paths(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    for key in ["data_dir", "embedding_path"]:
        path = Path(config[key])
        config[key] = str(path if path.is_absolute() else repo_root / path)
    output_root = Path(config["outputs"]["root"])
    config["outputs"]["root"] = str(output_root if output_root.is_absolute() else repo_root / output_root)
    return config


def normalize_negative_sampling(config: dict[str, Any]) -> dict[str, Any]:
    neg_cfg = config["negative_sampling"]
    if "hardmode" in neg_cfg and "hard_mode" not in neg_cfg:
        neg_cfg["hard_mode"] = neg_cfg.pop("hardmode")

    hard_mode = neg_cfg.get("hard_mode", "mixed")
    if isinstance(hard_mode, bool):
        neg_cfg["hard_mode"] = "mixed" if hard_mode else "none"
    elif isinstance(hard_mode, str) and hard_mode.lower() in {"true", "false"}:
        neg_cfg["hard_mode"] = "mixed" if hard_mode.lower() == "true" else "none"
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RGNN and term-level evaluation end to end.")
    parser.add_argument("--config", type=str, help="YAML or JSON config path.")
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--ngram", type=str)
    parser.add_argument("--domains", nargs="*")
    parser.add_argument("--use-embedding", action="store_true")
    parser.add_argument("--embedding-path", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cutoff", type=int)
    parser.add_argument(
        "--reconstruction-mode",
        choices=["predicted_term_links", "prediction_only_eval", "full_graph"],
    )
    parser.add_argument("--output-root", type=str)
    parser.add_argument("--resume", action="store_true", help="Allow writing into an existing run directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    repo_root = resolve_repo_root()
    file_config = load_config(Path(args.config)) if args.config else {}
    config = deep_merge(DEFAULT_CONFIG, file_config)
    config = apply_cli_overrides(config, args)
    config = normalize_negative_sampling(config)
    config = normalize_paths(config, repo_root)

    if not config["run_name"]:
        config["run_name"] = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    data_dir = Path(config["data_dir"])
    output_root = Path(config["outputs"]["root"])
    run_dir = output_root / config["run_name"]
    if run_dir.exists() and not args.resume and not args.dry_run:
        raise SystemExit(f"Run directory already exists: {run_dir}\nUse --resume or choose another --run-name.")

    paths = PipelinePaths(repo_root=repo_root, run_dir=run_dir)
    if not args.dry_run:
        ensure_output_dirs(paths)
        write_run_config(config, run_dir / "config.json")

    domains = resolve_domains(data_dir, config["domains"])
    print(f"Run name: {config['run_name']}")
    print(f"Domains : {', '.join(domains)}")
    print(f"N-gram  : {config['ngram']}")
    print(f"Output  : {run_dir}")

    artifacts: dict[str, dict[str, Path]] = {}
    for domain in domains:
        print(f"\n[1/4] Create dataset: {domain}")
        dataset_path = create_dataset_for_domain(
            paths=paths,
            data_dir=data_dir,
            domain=domain,
            ngram=config["ngram"],
            config=config,
            dry_run=args.dry_run,
        )

        print(f"[2/4] Train RGNN: {domain}")
        train_outputs = train_domain(
            paths=paths,
            domain=domain,
            dataset_path=dataset_path,
            config=config,
            dry_run=args.dry_run,
        )

        print(f"[3/4] Postprocess: {domain}")
        terms_path = postprocess_domain(
            paths=paths,
            domain=domain,
            dataset_path=dataset_path,
            trained_dataset=train_outputs["trained_dataset"],
            best_model=train_outputs["best_model"],
            config=config,
            dry_run=args.dry_run,
        )
        artifacts[domain] = {**train_outputs, "dataset": dataset_path, "terms": terms_path}

    if args.dry_run:
        print("\nDry run complete. No files were generated.")
        return 0

    print("\n[4/4] Evaluate reconstructed terms")
    term_metrics_csv = paths.term_eval_dir / "term_metrics.csv"
    evaluate_domains(
        data_dir=data_dir,
        reconstructed_terms_dir=paths.reconstructed_terms_dir,
        domains=domains,
        ngram=config["ngram"],
        output_csv=term_metrics_csv,
    )

    summary_csv = paths.summary_dir / "summary.csv"
    write_summary(
        run_name=config["run_name"],
        ngram=config["ngram"],
        domains=domains,
        link_metrics_dir=paths.link_metrics_dir,
        term_metrics_csv=term_metrics_csv,
        output_csv=summary_csv,
    )
    print(f"\nDone. Summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
