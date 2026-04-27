import argparse
from pathlib import Path


def iter_input_files(inputs, base_dir, pattern, recursive):
    if inputs:
        for path_str in inputs:
            yield Path(path_str)
        return

    base_path = Path(base_dir)
    if not base_path.is_dir():
        raise SystemExit(f"Input directory not found: {base_path}")

    if recursive:
        candidates = base_path.rglob(pattern)
    else:
        candidates = base_path.glob(pattern)

    for path in sorted(candidates, key=lambda p: str(p).lower()):
        if path.is_file():
            yield path


def build_output_path(input_path, out_dir, suffix):
    output_name = f"{input_path.stem}{suffix}{input_path.suffix}"
    if out_dir:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        return out_dir_path / output_name
    return input_path.with_name(output_name)


def extract_two_word_terms(input_path, output_path):
    kept = 0
    total = 0
    with input_path.open("r", encoding="utf-8") as f, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as w:
        for line in f:
            term = line.strip()
            if not term:
                continue
            total += 1
            if len(term.split()) == 2:
                kept += 1
                w.write(term + "\n")
    return total, kept


def main():
    parser = argparse.ArgumentParser(
        description="Extract only 2-word terms from multiple text files."
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        help="Explicit input file paths. If omitted, --dir is used.",
    )
    parser.add_argument(
        "--dir",
        default="data",
        help="Directory to search for input files (used when --inputs is omitted).",
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern for input files (used with --dir).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for files recursively under --dir.",
    )
    parser.add_argument(
        "--suffix",
        default="_2gram",
        help="Suffix to append to output filename before the extension.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory (default: same directory as input).",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Process files that already end with the suffix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show input/output pairs without writing files.",
    )

    args = parser.parse_args()

    input_files = list(
        iter_input_files(args.inputs, args.dir, args.pattern, args.recursive)
    )
    if not input_files:
        raise SystemExit("No input files found.")

    total_files = 0
    total_terms = 0
    total_kept = 0

    for input_path in input_files:
        if not input_path.is_file():
            print(f"Skip (not a file): {input_path}")
            continue

        if not args.include_existing and input_path.stem.endswith(args.suffix):
            print(f"Skip (already has suffix): {input_path}")
            continue

        output_path = build_output_path(input_path, args.out_dir, args.suffix)

        if args.dry_run:
            print(f"IN : {input_path}")
            print(f"OUT: {output_path}")
            print("")
            continue

        total, kept = extract_two_word_terms(input_path, output_path)
        total_files += 1
        total_terms += total
        total_kept += kept
        print(f"{input_path} -> {output_path} (kept {kept}/{total})")

    if not args.dry_run:
        print("")
        print(f"Files processed : {total_files}")
        print(f"Total terms     : {total_terms}")
        print(f"2-word terms    : {total_kept}")


if __name__ == "__main__":
    main()
