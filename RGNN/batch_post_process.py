#!/usr/bin/env python
"""
Batch post-processing for trained models.

This script runs pos_distribution.py and reconstruct_terms.py for each domain
in RGNN/data/dataset.

Usage:
    python RGNN/batch_post_process.py
    python RGNN/batch_post_process.py --domains Agriculture Computer
    python RGNN/batch_post_process.py --skip_pos_distribution
    python RGNN/batch_post_process.py --skip_reconstruct
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_domain_folders(dataset_dir):
    """Find all domain folders containing processed_dataset.pt."""
    domains = []
    for item in Path(dataset_dir).iterdir():
        if item.is_dir() and (item / "processed_dataset.pt").exists():
            domains.append(item.name)
    return sorted(domains)


def needs_trained_dataset(args):
    """Return True when reconstruction depends on predicted test edges."""
    return (
        args.use_trained_dataset
        or args.predicted_only
        or args.prediction_only_eval
        or args.include_predicted_negatives
    )


def get_threshold_from_trained_dataset(trained_dataset_path):
    """Extract optimal_threshold from trained_dataset.pt if available."""
    import torch
    try:
        dataset = torch.load(trained_dataset_path, map_location="cpu", weights_only=False)
        if "optimal_threshold" in dataset:
            return float(dataset["optimal_threshold"])
    except Exception as e:
        print(f"  Warning: Could not load trained_dataset.pt: {e}")
    return None


def run_pos_distribution(
    dataset_path,
    model_path,
    output_dir,
    threshold=None,
    pos_distribution_script=None,
    extra_args=None
):
    """Run pos_distribution.py for a domain."""
    cmd = [
        sys.executable,
        pos_distribution_script,
        "--dataset_path", str(dataset_path),
        "--model_path", str(model_path),
        "--output_csv", str(output_dir / "test_prob_hist.csv"),
        "--plot_pos_path", str(output_dir / "test_prob_hist_pos.png"),
        "--plot_neg_path", str(output_dir / "test_prob_hist_neg.png"),
        "--plot_overlay_path", str(output_dir / "test_prob_hist_overlay.png"),
        "--pos_recovered_csv", str(output_dir / "test_pos_recovered_edges.csv"),
        "--neg_recovered_csv", str(output_dir / "test_neg_recovered_edges.csv"),
    ]
    
    if threshold is not None:
        cmd.extend(["--threshold", str(threshold)])
    
    if extra_args:
        cmd.extend(extra_args)
    
    print("  Running pos_distribution.py...")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    
    if result.returncode != 0:
        print("  Error running pos_distribution.py:")
        print(result.stderr)
        return False
    
    # Print summary
    for line in result.stdout.split("\n"):
        if line.strip():
            print(f"    {line}")
    
    return True


def run_reconstruct_terms(
    dataset_path,
    output_path,
    reconstruct_script=None,
    cutoff=5,
    shortest_only=True,
    predicted_only=False,
    include_predicted_negatives=False,
    composition_from_test=False,
    prediction_only_eval=False
):
    """Run reconstruct_terms.py for a domain."""
    cmd = [
        sys.executable,
        reconstruct_script,
        "--dataset_path", str(dataset_path),
        "--output_path", str(output_path),
        "--cutoff", str(cutoff),
    ]
    
    if shortest_only:
        cmd.append("--shortest_only")
    
    if predicted_only:
        cmd.append("--predicted_only")
    
    if include_predicted_negatives:
        cmd.append("--include_predicted_negatives")
    
    if composition_from_test:
        cmd.append("--composition_from_test")
    
    if prediction_only_eval:
        cmd.append("--prediction_only_eval")
    
    print("  Running reconstruct_terms.py...")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    
    if result.returncode != 0:
        print("  Error running reconstruct_terms.py:")
        print(result.stderr)
        return False
    
    # Print summary
    for line in result.stdout.split("\n"):
        if "Reconstructed" in line or "Terms saved" in line:
            print(f"    {line}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch post-processing for trained models"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="RGNN/data/dataset",
        help="Base directory containing domain folders"
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="*",
        default=None,
        help="Specific domains to process (default: all found domains)"
    )
    parser.add_argument(
        "--skip_pos_distribution",
        action="store_true",
        help="Skip running pos_distribution.py"
    )
    parser.add_argument(
        "--skip_reconstruct",
        action="store_true",
        help="Skip running reconstruct_terms.py"
    )
    parser.add_argument(
        "--use_trained_dataset",
        action="store_true",
        help="Use trained_dataset.pt instead of processed_dataset.pt for reconstruct_terms"
    )
    
    # reconstruct_terms.py options
    parser.add_argument(
        "--cutoff",
        type=int,
        default=5,
        help="Max path length for term reconstruction (default: 5)"
    )
    parser.add_argument(
        "--shortest_only",
        action="store_true",
        default=True,
        help="Find only shortest paths (faster, default: True)"
    )
    parser.add_argument(
        "--all_paths",
        action="store_true",
        help="Find all simple paths instead of shortest only (slower)"
    )
    parser.add_argument(
        "--predicted_only",
        action="store_true",
        help="Reconstruct only links predicted in test set (requires trained_dataset.pt)"
    )
    parser.add_argument(
        "--include_predicted_negatives",
        action="store_true",
        help="Include negative test samples with score > threshold"
    )
    parser.add_argument(
        "--composition_from_test",
        action="store_true",
        help="Build composition graph using only test split Type 1 edges"
    )
    parser.add_argument(
        "--prediction_only_eval",
        action="store_true",
        help="Use ONLY predicted edges for both composition (Type 1) and term links (Type 2). "
             "This enables pure prediction-based evaluation. (requires trained_dataset.pt)"
    )
    
    parser.add_argument(
        "--pos_distribution_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments for pos_distribution.py (must be last)"
    )
    
    args = parser.parse_args()
    
    # Determine script directory
    script_dir = Path(__file__).resolve().parent
    
    # Script paths
    pos_distribution_script = str(script_dir / "pos_distribution.py")
    reconstruct_script = str(script_dir / "reconstruct_terms.py")
    
    # Resolve dataset directory
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        workspace_root = script_dir.parent
        dataset_dir = workspace_root / args.dataset_dir
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return 1
    
    # Find domains
    available_domains = find_domain_folders(dataset_dir)
    print(f"Available domains: {available_domains}")
    
    if args.domains:
        domains = [d for d in args.domains if d in available_domains]
        if len(domains) != len(args.domains):
            missing = set(args.domains) - set(domains)
            print(f"Warning: Domains not found: {missing}")
    else:
        domains = available_domains
    
    if not domains:
        print("No domains to process.")
        return 1
    
    print(f"Processing domains: {domains}")
    
    # Create reconstruct_terms output directory
    reconstruct_output_dir = dataset_dir / "reconstruct_terms"
    reconstruct_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each domain
    results = {}
    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Processing: {domain}")
        print(f"{'='*60}")
        
        domain_dir = dataset_dir / domain
        
        # Check required files
        processed_dataset = domain_dir / "processed_dataset.pt"
        trained_dataset = domain_dir / "trained_dataset.pt"
        best_model = domain_dir / "best_model.pt"
        
        if not processed_dataset.exists():
            print("  Skipping: processed_dataset.pt not found")
            results[domain] = {"status": "skipped", "reason": "no processed_dataset.pt"}
            continue
        
        if not best_model.exists():
            print("  Skipping: best_model.pt not found")
            results[domain] = {"status": "skipped", "reason": "no best_model.pt"}
            continue
        
        # Get threshold from trained_dataset if available
        threshold = None
        if trained_dataset.exists():
            threshold = get_threshold_from_trained_dataset(trained_dataset)
            if threshold is not None:
                print(f"  Using optimal threshold from trained_dataset: {threshold:.4f}")
        
        domain_results = {"status": "success", "pos_distribution": False, "reconstruct": False}
        
        # Run pos_distribution.py
        if not args.skip_pos_distribution:
            result_dir = domain_dir / "result"
            result_dir.mkdir(parents=True, exist_ok=True)
            
            success = run_pos_distribution(
                dataset_path=processed_dataset,
                model_path=best_model,
                output_dir=result_dir,
                threshold=threshold,
                pos_distribution_script=pos_distribution_script,
                extra_args=args.pos_distribution_args if args.pos_distribution_args else None
            )
            domain_results["pos_distribution"] = success
        
        # Run reconstruct_terms.py
        if not args.skip_reconstruct:
            # Predicted-edge modes require trained_dataset.pt.
            if needs_trained_dataset(args) and trained_dataset.exists():
                reconstruct_dataset = trained_dataset
            elif needs_trained_dataset(args) and not trained_dataset.exists():
                print("  Skipping reconstruct: selected options require trained_dataset.pt")
                domain_results["reconstruct"] = False
                results[domain] = domain_results
                continue
            else:
                reconstruct_dataset = processed_dataset
            
            output_file = reconstruct_output_dir / f"{domain}_reconstruct_terms.txt"
            
            # Determine shortest_only (--all_paths overrides default)
            use_shortest_only = not args.all_paths
            
            # Use predicted Type 2 term links when prediction-specific options are selected.
            use_predicted_only = args.predicted_only
            if (
                (args.use_trained_dataset or args.include_predicted_negatives)
                and trained_dataset.exists()
                and not args.prediction_only_eval
            ):
                use_predicted_only = True
            
            success = run_reconstruct_terms(
                dataset_path=reconstruct_dataset,
                output_path=output_file,
                reconstruct_script=reconstruct_script,
                cutoff=args.cutoff,
                shortest_only=use_shortest_only,
                predicted_only=use_predicted_only,
                include_predicted_negatives=args.include_predicted_negatives,
                composition_from_test=args.composition_from_test,
                prediction_only_eval=args.prediction_only_eval
            )
            domain_results["reconstruct"] = success
        
        results[domain] = domain_results
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for domain, result in results.items():
        if result["status"] == "skipped":
            print(f"  {domain}: SKIP ({result['reason']})")
        else:
            pos_status = "OK" if result.get("pos_distribution", False) else "NO"
            rec_status = "OK" if result.get("reconstruct", False) else "NO"
            print(f"  {domain}: pos_distribution={pos_status}, reconstruct={rec_status}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
