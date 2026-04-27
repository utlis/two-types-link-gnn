import argparse
from pathlib import Path

from evaluate_terms import calculate_metrics, save_to_csv


def find_ground_truth_file(domain_dir: Path, gt_suffix: str = None):
    """Find one test file in either flat or train/dev/test layout."""
    search_dirs = [domain_dir]
    if (domain_dir / "test").is_dir():
        search_dirs.insert(0, domain_dir / "test")

    for directory in search_dirs:
        gt_files = sorted(
            [p for p in directory.iterdir() if p.is_file() and "test" in p.name.lower()],
            key=lambda p: p.name.lower(),
        )
        if gt_suffix:
            gt_files = [p for p in gt_files if p.name.endswith(gt_suffix)]
        if len(gt_files) == 1:
            return gt_files[0], None
        if len(gt_files) > 1:
            return None, f"Expected exactly 1 gt file (name contains 'test') in: {directory}"

    return None, None


def candidate_result_names(domain_name: str, suffix: str = None):
    """Return result filenames in the order they should be tried."""
    if suffix:
        return [f"{domain_name}{suffix}"]
    return [
        f"{domain_name}_reconstruct_terms_2gram.txt",
        f"{domain_name}_reconstruct_terms.txt",
    ]


def resolve_pairs(gt_dir: Path, res_dir: Path, suffix: str = None, gt_suffix: str = None):
    pairs = []
    errors = []

    subdirs = sorted([d for d in gt_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower())
    if not subdirs:
        errors.append(f"No subdirectories found in: {gt_dir}")
        return pairs, errors

    for subdir in subdirs:
        gt_path, error = find_ground_truth_file(subdir, gt_suffix)
        if error:
            errors.append(error)
            continue
        if gt_path is None:
            # Skip folders that do not look like dataset/domain folders.
            continue

        candidates = candidate_result_names(subdir.name, suffix)
        res_path = None
        for fname in candidates:
            p = res_dir / fname
            if p.exists():
                res_path = p
                break

        if res_path is None:
            searched_str = ", ".join(candidates)
            errors.append(
                f"Result file not found for {subdir.name} in {res_dir}. Checked: {searched_str}"
            )
            continue

        pairs.append((gt_path, res_path))

    return pairs, errors


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluate_terms.py across ground truth and result directories."
    )
    parser.add_argument(
        "--ground-truth",
        default="data",
        help="Path to the directory containing ground truth subfolders.",
    )
    parser.add_argument(
        "--result",
        default="reconstruct_terms_2grams",
        help="Path to the directory containing result files.",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Specific suffix of the result files. If not set, tries default patterns (_reconstruct_terms_2gram.txt, _reconstruct_terms.txt).",
    )
    parser.add_argument(
        "--ground-truth-suffix",
        default=None,
        help="Specific suffix of the ground-truth test file, e.g. _2gram.txt or _All.txt.",
    )
    parser.add_argument(
        "--out",
        default="metrics_all.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print gt/res pairs without running evaluation.",
    )

    args = parser.parse_args()

    gt_dir = Path(args.ground_truth)
    res_dir = Path(args.result)

    if not gt_dir.is_dir():
        raise SystemExit(f"Ground truth directory not found: {gt_dir}")
    if not res_dir.is_dir():
        raise SystemExit(f"Result directory not found: {res_dir}")

    pairs, errors = resolve_pairs(gt_dir, res_dir, args.suffix, args.ground_truth_suffix)
    for error in errors:
        print(f"Error: {error}")

    if errors and not pairs:
        raise SystemExit(1)

    if args.dry_run:
        for gt_path, res_path in pairs:
            print(f"GT : {gt_path}")
            print(f"RES: {res_path}")
            print("")
        return

    for gt_path, res_path in pairs:
        print(f"Ground Truth file: {gt_path}")
        print(f"Result file      : {res_path}")
        print("")
        metrics = calculate_metrics(str(gt_path), str(res_path))
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1_score']:.4f}")
        print("")
        save_to_csv(args.out, str(gt_path), str(res_path), metrics)


if __name__ == "__main__":
    main()
