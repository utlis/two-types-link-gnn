#!/usr/bin/env python
"""
Batch create datasets for multiple domains.

This script processes each domain folder (Agriculture, Computer, etc.) under the data directory,
using the specified n-gram files for train/dev/test splits.

Usage:
    python RGNN/batch_create_dataset.py --ngram 2gram
    python RGNN/batch_create_dataset.py --ngram all --domains Agriculture Computer
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_domain_folders(data_dir):
    """Find all domain folders (directories containing train/dev/test subfolders)."""
    domains = []
    for item in Path(data_dir).iterdir():
        if item.is_dir() and all((item / split).is_dir() for split in ["train", "dev", "test"]):
            domains.append(item.name)
    return sorted(domains)


def find_ngram_file(folder_path, split, domain, ngram):
    """
    Find the n-gram file for a given split and domain.
    
    Expected naming convention:
    - {split}_term_{domain}_{ngram}.txt
    - e.g., train_term_Agriculture_2gram.txt
    """
    folder = Path(folder_path)
    
    # Try exact pattern first
    pattern = f"{split}_term_{domain}_{ngram}.txt"
    file_path = folder / pattern
    if file_path.exists():
        return str(file_path)
    
    # Try a case-insensitive fallback for naming variants.
    for f in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if f.is_file() and ngram.lower() in f.name.lower() and f.suffix == ".txt":
            return str(f)
    
    return None


def run_create_dataset(train_path, dev_path, test_path, output_dir, domain_name, create_dataset_script, extra_args=None):
    """Run create_dataset.py and capture output."""
    output_path = Path(output_dir) / "processed_dataset.pt"
    
    cmd = [
        sys.executable,
        create_dataset_script,
        "--train_path", train_path,
        "--dev_path", dev_path,
        "--test_path", test_path,
        "--output_path", str(output_path),
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*60}")
    print(f"Creating dataset for: {domain_name}")
    print(f"  Train: {train_path}")
    print(f"  Dev:   {dev_path}")
    print(f"  Test:  {test_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")
    
    # Run and capture output
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    
    # Print stdout
    if result.stdout:
        print(result.stdout)
    
    # Print stderr if any
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    # Check for errors
    if result.returncode != 0:
        print(f"Error: create_dataset.py failed for {domain_name}")
        return None, result.stdout + result.stderr
    
    return output_path, result.stdout


def extract_metrics_from_output(output_text):
    """Extract relevant metrics from create_dataset.py output."""
    lines = []
    capture = False
    
    for line in output_text.split("\n"):
        # Capture edge split summaries
        if line.startswith("[train]") or line.startswith("[val]") or line.startswith("[test]"):
            lines.append(line)
            capture = True
        elif line.strip().startswith("Type ") and capture:
            lines.append(line)
        elif line.startswith("[full_graph]") or line.startswith("[train_graph]"):
            lines.append(line)
        elif not line.strip():
            if capture and lines and lines[-1].strip():
                lines.append("")
            capture = False
    
    return "\n".join(lines)


def save_metrics_file(output_dir, domain_name, metrics_text):
    """Save metrics to a text file."""
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = metrics_dir / f"{domain_name}.txt"
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(metrics_text)
    
    print(f"Metrics saved to: {metrics_file}")
    return metrics_file


def main():
    parser = argparse.ArgumentParser(
        description="Batch create datasets for multiple domains"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to the data directory containing domain folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="RGNN/data/dataset",
        help="Base output directory for datasets"
    )
    parser.add_argument(
        "--ngram",
        type=str,
        default="all",
        help="N-gram type to use (e.g., 1gram, 2gram, 3gram, all)"
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="*",
        default=None,
        help="Specific domains to process (default: all found domains)"
    )
    parser.add_argument(
        "--create_dataset_script",
        type=str,
        default=None,
        help="Path to create_dataset.py (default: auto-detect)"
    )
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments to pass to create_dataset.py (must be last argument)"
    )
    
    args = parser.parse_args()
    
    # Determine script directory
    script_dir = Path(__file__).resolve().parent
    
    # Auto-detect create_dataset.py
    if args.create_dataset_script is None:
        args.create_dataset_script = str(script_dir / "create_dataset.py")
    
    if not Path(args.create_dataset_script).exists():
        print(f"Error: create_dataset.py not found at {args.create_dataset_script}")
        return 1
    
    # Resolve data directory
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        # Try relative to workspace root
        workspace_root = script_dir.parent
        data_dir = workspace_root / args.data_dir
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1
    
    # Find domains
    available_domains = find_domain_folders(data_dir)
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
    print(f"Using n-gram: {args.ngram}")
    
    # Process each domain
    results = {}
    for domain in domains:
        domain_data_dir = data_dir / domain
        
        # Find n-gram files
        train_file = find_ngram_file(domain_data_dir / "train", "train", domain, args.ngram)
        dev_file = find_ngram_file(domain_data_dir / "dev", "dev", domain, args.ngram)
        test_file = find_ngram_file(domain_data_dir / "test", "test", domain, args.ngram)
        
        if not all([train_file, dev_file, test_file]):
            print(f"\nWarning: Missing files for {domain} with {args.ngram}:")
            print(f"  Train: {train_file or 'NOT FOUND'}")
            print(f"  Dev:   {dev_file or 'NOT FOUND'}")
            print(f"  Test:  {test_file or 'NOT FOUND'}")
            results[domain] = {"status": "skipped", "reason": "missing files"}
            continue
        
        # Create output directory
        output_dir = Path(args.output_dir) / domain
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run create_dataset.py
        output_path, output_text = run_create_dataset(
            train_file,
            dev_file,
            test_file,
            output_dir,
            domain,
            args.create_dataset_script,
            args.extra_args if args.extra_args else None
        )
        
        if output_path and output_text:
            # Extract and save metrics
            metrics_text = extract_metrics_from_output(output_text)
            if metrics_text.strip():
                save_metrics_file(output_dir, domain, metrics_text)
            
            results[domain] = {"status": "success", "output": str(output_path)}
        else:
            results[domain] = {"status": "failed", "output": output_text}
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for domain, result in results.items():
        status = result["status"]
        if status == "success":
            print(f"  {domain}: OK Success -> {result['output']}")
        elif status == "skipped":
            print(f"  {domain}: SKIP ({result['reason']})")
        else:
            print(f"  {domain}: FAIL")
    
    # Return non-zero if any failed
    failed_count = sum(1 for r in results.values() if r["status"] == "failed")
    return failed_count


if __name__ == "__main__":
    sys.exit(main())

