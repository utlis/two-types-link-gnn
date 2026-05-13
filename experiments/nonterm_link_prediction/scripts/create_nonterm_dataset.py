from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


REPO_ROOT = resolve_repo_root()
RGNN_DIR = REPO_ROOT / "RGNN"
sys.path.append(str(RGNN_DIR))

from data.data_loader import build_vocab, load_fasttext_model  # noqa: E402
from data.graph_builder import build_rgnn_graph, iter_edges_for_term  # noqa: E402


def read_terms(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def find_term_file(data_dir: Path, domain: str, split: str, ngram: str) -> Path:
    folder = data_dir / domain / split
    exact = folder / f"{split}_term_{domain}_{ngram}.txt"
    if exact.exists():
        return exact
    wanted = ngram.lower()
    for path in sorted(folder.glob(f"{split}_term_{domain}_*.txt")):
        if wanted in path.name.lower():
            return path
    raise FileNotFoundError(f"No term {ngram} file found in {folder}")


def find_nonterm_files(data_dir: Path, domain: str, split: str, ngram: str) -> list[Path]:
    folder = data_dir / domain / split
    exact = folder / f"{split}_nonterm_{domain}_{ngram}.txt"
    if exact.exists():
        return [exact]

    if ngram.lower() == "all":
        merged = []
        for part in ["1gram", "2gram", "3gram"]:
            path = folder / f"{split}_nonterm_{domain}_{part}.txt"
            if path.exists():
                merged.append(path)
        if merged:
            return merged

    wanted = ngram.lower()
    candidates = [p for p in sorted(folder.glob(f"{split}_nonterm_{domain}_*.txt")) if wanted in p.name.lower()]
    if candidates:
        return candidates
    raise FileNotFoundError(f"No nonterm {ngram} file found in {folder}")


def read_nonterms(data_dir: Path, domain: str, split: str, ngram: str) -> tuple[list[str], list[str]]:
    paths = find_nonterm_files(data_dir, domain, split, ngram)
    terms: list[str] = []
    for path in paths:
        terms.extend(read_terms(path))
    return terms, [str(path) for path in paths]


def build_edge_set(expressions: list[str], vocab: dict[str, int]) -> set[tuple[int, int, int]]:
    edges: set[tuple[int, int, int]] = set()
    for expression in expressions:
        edges.update(iter_edges_for_term(expression, vocab))
    return edges


def edge_set_to_tensors(edge_set: set[tuple[int, int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    ordered = sorted(edge_set)
    if not ordered:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    edge_index = torch.tensor([[src, dst] for src, dst, _ in ordered], dtype=torch.long).t().contiguous()
    edge_type = torch.tensor([etype for _, _, etype in ordered], dtype=torch.long)
    return edge_index, edge_type


def edge_tensors_to_triplets(edge_index: torch.Tensor, edge_type: torch.Tensor) -> set[tuple[int, int, int]]:
    return {
        (int(edge_index[0, i]), int(edge_type[i]), int(edge_index[1, i]))
        for i in range(edge_index.size(1))
    }


def edge_set_to_triplets(edge_set: set[tuple[int, int, int]]) -> set[tuple[int, int, int]]:
    return {(src, rel, dst) for src, dst, rel in edge_set}


def build_embedding_cache_key(vocab: dict[str, int], embedding_path: Path) -> str:
    resolved = embedding_path.resolve()
    stat = resolved.stat()
    tokens = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]
    payload = {
        "embedding_path": str(resolved),
        "embedding_size": stat.st_size,
        "embedding_mtime_ns": stat.st_mtime_ns,
        "tokens": tokens,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_or_build_embedding_matrix(vocab: dict[str, int], embedding_path: Path, cache_dir: Path) -> torch.Tensor:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = build_embedding_cache_key(vocab, embedding_path)
    cache_path = cache_dir / f"{cache_key}.pt"
    tokens = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("tokens") == tokens and "x" in cached:
            print(f"Loading cached embedding matrix: {cache_path}")
            return cached["x"]

    print(f"Loading FastText model from {embedding_path}...")
    emb_model = load_fasttext_model(str(embedding_path))
    x_list = []
    for token in tokens:
        try:
            vector = emb_model.wv[token]
        except KeyError:
            vector = np.zeros(emb_model.vector_size, dtype=np.float32)
        x_list.append(torch.tensor(vector, dtype=torch.float))
    x = torch.stack(x_list, dim=0)
    torch.save(
        {
            "x": x,
            "tokens": tokens,
            "embedding_path": str(embedding_path.resolve()),
            "vector_size": int(emb_model.vector_size),
        },
        cache_path,
    )
    print(f"Saved embedding matrix cache: {cache_path}")
    return x


def sample_negative_triplets_excluding(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
    *,
    seed: int,
    extra_positive_edges: list[tuple[torch.Tensor, torch.Tensor]],
    forbidden_negative_edges: set[tuple[int, int, int]],
    neg_ratio: int,
    hard_negative_mode: str,
    corrupt_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    if neg_ratio < 1:
        raise ValueError("neg_ratio must be >= 1")
    if hard_negative_mode not in {"none", "degree", "mixed"}:
        raise ValueError("hard_negative_mode must be one of: none, degree, mixed")
    if corrupt_mode not in {"tail", "head", "both"}:
        raise ValueError("corrupt_mode must be one of: tail, head, both")

    rng = np.random.default_rng(seed)
    forbidden = set(forbidden_negative_edges)
    degree = np.ones(num_nodes, dtype=np.float64)

    def add_positive_triplets(e_index: torch.Tensor, e_type: torch.Tensor) -> None:
        for i in range(e_index.size(1)):
            src = int(e_index[0, i])
            dst = int(e_index[1, i])
            rel = int(e_type[i])
            forbidden.add((src, rel, dst))
            degree[src] += 1.0
            degree[dst] += 1.0

    add_positive_triplets(edge_index, edge_type)
    for e_index_extra, e_type_extra in extra_positive_edges:
        if e_index_extra.numel() > 0:
            add_positive_triplets(e_index_extra, e_type_extra)

    degree_prob = degree / degree.sum()

    def choose_corrupt_side() -> str:
        if corrupt_mode == "both":
            return "head" if rng.random() < 0.5 else "tail"
        return corrupt_mode

    def sample_node() -> int:
        use_degree = hard_negative_mode == "degree" or (
            hard_negative_mode == "mixed" and rng.random() < 0.5
        )
        if use_degree:
            return int(rng.choice(num_nodes, p=degree_prob))
        return int(rng.integers(0, num_nodes))

    def build_candidate(src: int, rel: int, dst: int, side: str, node: int) -> tuple[int, int, int]:
        if side == "head":
            return node, rel, dst
        return src, rel, node

    def is_valid_negative(src: int, rel: int, dst: int, side: str, node: int) -> bool:
        if side == "head" and node == src:
            return False
        if side == "tail" and node == dst:
            return False
        return build_candidate(src, rel, dst, side, node) not in forbidden

    neg_edge_list = []
    neg_type_list = []
    fallback_count = 0
    rejected_forbidden_count = 0
    head_corrupt_count = 0
    tail_corrupt_count = 0

    for i in range(edge_index.size(1)):
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])
        rel = int(edge_type[i])

        for _ in range(neg_ratio):
            side = choose_corrupt_side()
            candidate = None

            for _ in range(100):
                node = sample_node()
                if is_valid_negative(src, rel, dst, side, node):
                    candidate = build_candidate(src, rel, dst, side, node)
                    break
                rejected_forbidden_count += 1

            if candidate is None:
                fallback_count += 1
                for node in rng.permutation(num_nodes):
                    node = int(node)
                    if is_valid_negative(src, rel, dst, side, node):
                        candidate = build_candidate(src, rel, dst, side, node)
                        break

            if candidate is None:
                raise RuntimeError("Could not sample a negative edge outside positive and Sn candidate sets.")

            neg_src, neg_rel, neg_dst = candidate
            neg_edge_list.append([neg_src, neg_dst])
            neg_type_list.append(neg_rel)
            if side == "head":
                head_corrupt_count += 1
            else:
                tail_corrupt_count += 1

    if not neg_edge_list:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            {
                "fallback_count": 0,
                "total": 0,
                "head_corrupt_count": 0,
                "tail_corrupt_count": 0,
                "rejected_forbidden_count": 0,
                "forbidden_sn_candidate_count": len(forbidden_negative_edges),
            },
        )

    neg_edge_index = torch.tensor(neg_edge_list, dtype=torch.long).t().contiguous()
    neg_edge_type = torch.tensor(neg_type_list, dtype=torch.long)
    return neg_edge_index, neg_edge_type, {
        "fallback_count": fallback_count,
        "total": int(neg_edge_index.size(1)),
        "head_corrupt_count": head_corrupt_count,
        "tail_corrupt_count": tail_corrupt_count,
        "rejected_forbidden_count": rejected_forbidden_count,
        "forbidden_sn_candidate_count": len(forbidden_negative_edges),
    }


def remove_positive_and_prior_edges(
    train_edges: set[tuple[int, int, int]],
    dev_edges: set[tuple[int, int, int]],
    test_edges: set[tuple[int, int, int]],
) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]], set[tuple[int, int, int]], dict[str, int]]:
    stats = {
        "dev_overlap_train": len(dev_edges & train_edges),
        "test_overlap_train": len(test_edges & train_edges),
        "test_overlap_dev": len(test_edges & dev_edges),
    }
    dev_clean = dev_edges - train_edges
    test_clean = test_edges - train_edges - dev_clean
    stats.update(
        {
            "train_edges": len(train_edges),
            "dev_edges_before": len(dev_edges),
            "dev_edges_after": len(dev_clean),
            "test_edges_before": len(test_edges),
            "test_edges_after": len(test_clean),
        }
    )
    return train_edges, dev_clean, test_clean, stats


def sample_explicit_negatives(
    edge_set: set[tuple[int, int, int]],
    pos_count: int,
    ratio: int,
    policy: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | str]]:
    if policy not in {"match_positive", "all"}:
        raise ValueError("explicit negative policy must be 'match_positive' or 'all'")

    ordered = sorted(edge_set)
    requested = len(ordered) if policy == "all" else pos_count * ratio
    rng = random.Random(seed)
    rng.shuffle(ordered)
    selected = ordered if policy == "all" else ordered[: min(requested, len(ordered))]
    selected_set = set(selected)
    edge_index, edge_type = edge_set_to_tensors(selected_set)
    return edge_index, edge_type, {
        "policy": policy,
        "available": len(edge_set),
        "requested": requested,
        "selected": int(edge_index.size(1)),
    }


def write_split_preview(path: Path, vocab: dict[str, int], split_name: str, label: str, edge_index: torch.Tensor, edge_type: torch.Tensor) -> None:
    id_to_token = {idx: token for token, idx in vocab.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "src_id", "dst_id", "src_token", "dst_token", "edge_type"])
        for i in range(edge_index.size(1)):
            src = int(edge_index[0, i])
            dst = int(edge_index[1, i])
            writer.writerow([split_name, label, src, dst, id_to_token.get(src, ""), id_to_token.get(dst, ""), int(edge_type[i]) + 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create RGNN dataset for St/Sn non-term experiments.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--ngram", required=True)
    parser.add_argument("--experiment", required=True, choices=["exp1_st_only", "exp2_st_sn"])
    parser.add_argument("--use_nonterm_train", action="store_true")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--metadata_path")
    parser.add_argument("--preview_dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg_seed", type=int, default=42)
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument("--explicit_negative_policy", choices=["match_positive", "all"], default="match_positive")
    parser.add_argument("--hard_negative_mode", choices=["none", "degree", "mixed"], default="mixed")
    parser.add_argument("--corrupt_mode", choices=["tail", "head", "both"], default="both")
    parser.add_argument("--use_embedding", action="store_true")
    parser.add_argument("--embedding_path", default="data/cc.en.300.bin")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_path = Path(args.output_path)

    term_paths = {split: find_term_file(data_dir, args.domain, split, args.ngram) for split in ["train", "dev", "test"]}
    st_terms = {split: read_terms(path) for split, path in term_paths.items()}
    sn_terms: dict[str, list[str]] = {}
    sn_paths: dict[str, list[str]] = {}
    for split in ["train", "dev", "test"]:
        sn_terms[split], sn_paths[split] = read_nonterms(data_dir, args.domain, split, args.ngram)

    all_expressions = st_terms["train"] + st_terms["dev"] + st_terms["test"] + sn_terms["train"] + sn_terms["dev"] + sn_terms["test"]
    vocab = build_vocab(all_expressions)

    graph_data = build_rgnn_graph(st_terms["train"] + st_terms["dev"] + st_terms["test"], vocab, None)
    if args.use_embedding:
        graph_data.x = load_or_build_embedding_matrix(
            vocab,
            Path(args.embedding_path),
            REPO_ROOT / "experiments" / "nonterm_link_prediction" / ".cache" / "embeddings",
        )

    st_edge_sets = {split: build_edge_set(st_terms[split], vocab) for split in ["train", "dev", "test"]}
    st_train, st_dev, st_test, pos_dedup_stats = remove_positive_and_prior_edges(
        st_edge_sets["train"], st_edge_sets["dev"], st_edge_sets["test"]
    )

    all_positive_edges = st_train | st_dev | st_test
    sn_edge_sets_raw = {split: build_edge_set(sn_terms[split], vocab) for split in ["train", "dev", "test"]}
    sn_overlap_positive = {split: len(edges & all_positive_edges) for split, edges in sn_edge_sets_raw.items()}
    sn_edge_sets = {split: edges - all_positive_edges for split, edges in sn_edge_sets_raw.items()}
    all_sn_candidate_triplets = set().union(*(edge_set_to_triplets(edges) for edges in sn_edge_sets.values()))
    sn_train, sn_dev, sn_test, neg_dedup_stats = remove_positive_and_prior_edges(
        sn_edge_sets["train"], sn_edge_sets["dev"], sn_edge_sets["test"]
    )

    train_pos_index, train_pos_type = edge_set_to_tensors(st_train)
    val_pos_index, val_pos_type = edge_set_to_tensors(st_dev)
    test_pos_index, test_pos_type = edge_set_to_tensors(st_test)

    all_pos_for_random = [
        (train_pos_index, train_pos_type),
        (val_pos_index, val_pos_type),
        (test_pos_index, test_pos_type),
    ]

    if args.use_nonterm_train:
        train_neg_index, train_neg_type, train_neg_stats = sample_explicit_negatives(
            sn_train, train_pos_index.size(1), args.neg_ratio, args.explicit_negative_policy, args.neg_seed
        )
        val_neg_index, val_neg_type, val_neg_stats = sample_explicit_negatives(
            sn_dev, val_pos_index.size(1), args.neg_ratio, args.explicit_negative_policy, args.neg_seed + 1
        )
    else:
        train_neg_index, train_neg_type, train_neg_stats = sample_negative_triplets_excluding(
            train_pos_index,
            train_pos_type,
            graph_data.num_nodes,
            seed=args.neg_seed,
            extra_positive_edges=all_pos_for_random,
            forbidden_negative_edges=all_sn_candidate_triplets,
            neg_ratio=args.neg_ratio,
            hard_negative_mode=args.hard_negative_mode,
            corrupt_mode=args.corrupt_mode,
        )
        val_neg_index, val_neg_type, val_neg_stats = sample_negative_triplets_excluding(
            val_pos_index,
            val_pos_type,
            graph_data.num_nodes,
            seed=args.neg_seed + 1,
            extra_positive_edges=all_pos_for_random,
            forbidden_negative_edges=all_sn_candidate_triplets,
            neg_ratio=args.neg_ratio,
            hard_negative_mode=args.hard_negative_mode,
            corrupt_mode=args.corrupt_mode,
        )

    test_neg_index, test_neg_type, test_neg_stats = sample_explicit_negatives(
        sn_test, test_pos_index.size(1), args.neg_ratio, args.explicit_negative_policy, args.neg_seed + 2
    )

    dataset = {
        "vocab": vocab,
        "num_nodes": graph_data.num_nodes,
        "x": graph_data.x,
        "edge_index": graph_data.edge_index,
        "edge_type": graph_data.edge_type,
        "split_mode": "pre_split_st_sn",
        "experiment": args.experiment,
        "train": {
            "pos_edge_index": train_pos_index,
            "pos_edge_type": train_pos_type,
            "neg_edge_index": train_neg_index,
            "neg_edge_type": train_neg_type,
        },
        "val": {
            "pos_edge_index": val_pos_index,
            "pos_edge_type": val_pos_type,
            "neg_edge_index": val_neg_index,
            "neg_edge_type": val_neg_type,
        },
        "test": {
            "pos_edge_index": test_pos_index,
            "pos_edge_type": test_pos_type,
            "neg_edge_index": test_neg_index,
            "neg_edge_type": test_neg_type,
        },
        "train_neg_sampling_config": None
        if args.use_nonterm_train
        else {
            "neg_seed": args.neg_seed,
            "neg_ratio": args.neg_ratio,
            "hard_negative_mode": args.hard_negative_mode,
            "corrupt_mode": args.corrupt_mode,
        },
        "metadata": {
            "domain": args.domain,
            "ngram": args.ngram,
            "experiment": args.experiment,
            "use_nonterm_train": bool(args.use_nonterm_train),
            "term_paths": {split: str(path) for split, path in term_paths.items()},
            "nonterm_paths": sn_paths,
            "positive_dedup": pos_dedup_stats,
            "negative_dedup": neg_dedup_stats,
            "nonterm_edges_overlapping_positive": sn_overlap_positive,
            "negative_sampling": {
                "train": train_neg_stats,
                "val": val_neg_stats,
                "test": test_neg_stats,
            },
            "counts": {
                "vocab": len(vocab),
                "train_pos": int(train_pos_index.size(1)),
                "val_pos": int(val_pos_index.size(1)),
                "test_pos": int(test_pos_index.size(1)),
                "train_neg": int(train_neg_index.size(1)),
                "val_neg": int(val_neg_index.size(1)),
                "test_neg": int(test_neg_index.size(1)),
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)

    metadata_path = Path(args.metadata_path) if args.metadata_path else output_path.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(dataset["metadata"], indent=2, ensure_ascii=False), encoding="utf-8")

    if args.preview_dir:
        preview_dir = Path(args.preview_dir)
        for split, edge_index, edge_type in [
            ("train_pos", train_pos_index, train_pos_type),
            ("val_pos", val_pos_index, val_pos_type),
            ("test_pos", test_pos_index, test_pos_type),
            ("train_neg", train_neg_index, train_neg_type),
            ("val_neg", val_neg_index, val_neg_type),
            ("test_neg", test_neg_index, test_neg_type),
        ]:
            split_name, label = split.rsplit("_", 1)
            write_split_preview(preview_dir / f"{split}.csv", vocab, split_name, label, edge_index, edge_type)

    print(json.dumps(dataset["metadata"]["counts"], indent=2))
    print(f"Dataset saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
