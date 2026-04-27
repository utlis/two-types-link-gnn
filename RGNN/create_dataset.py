import argparse
import csv
import random
import pandas as pd
import torch
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import build_vocab, load_fasttext_model
from data.graph_builder import build_rgnn_graph, iter_edges_for_term
from data.dataset import split_edges_stratified, sample_negative_triplets


def display_edge_type(edge_type_id):
    """Map internal relation ids to the project-facing Type labels."""
    return int(edge_type_id) + 1


def compute_density(num_nodes, num_edges):
    if num_nodes <= 1:
        return 0.0
    max_edges = num_nodes * (num_nodes - 1)
    return float(num_edges) / max_edges

def compute_degree_distribution(edge_index, num_nodes):
    if edge_index.numel() == 0:
        degrees = [0] * num_nodes
    else:
        out_deg = torch.bincount(edge_index[0], minlength=num_nodes)
        in_deg = torch.bincount(edge_index[1], minlength=num_nodes)
        degrees = (out_deg + in_deg).tolist()
    degree_hist = Counter(int(d) for d in degrees)
    return degrees, dict(degree_hist)

def plot_degree_distribution(degree_hist, title, output_path):
    if not degree_hist:
        degree_hist = {0: 0}
    degrees = sorted(degree_hist.keys())
    counts = [degree_hist[d] for d in degrees]
    plt.figure(figsize=(8, 4))
    plt.bar(degrees, counts, color="#4c72b0")
    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_degree_loglog(degree_hist, title, output_path):
    filtered = {d: c for d, c in degree_hist.items() if d > 0}
    if not filtered:
        filtered = {1: max(degree_hist.get(0, 0), 1)}
    degrees = sorted(filtered.keys())
    counts = [filtered[d] for d in degrees]
    plt.figure(figsize=(8, 4))
    plt.scatter(degrees, counts, color="#dd8452", s=40)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Degree (log scale)")
    plt.ylabel("Number of Nodes (log scale)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def compute_global_clustering(edge_index, num_nodes):
    if num_nodes < 3:
        return 0.0
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    if edge_index.numel() > 0:
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)
    if G.number_of_edges() == 0:
        return 0.0
    return nx.transitivity(G.to_undirected())

def compute_er_clustering(num_nodes, num_edges, seed):
    if num_nodes < 3 or num_edges == 0:
        return 0.0
    max_edges = num_nodes * (num_nodes - 1)
    m = min(int(num_edges), max_edges)
    G_er = nx.gnm_random_graph(num_nodes, m, seed=seed, directed=True)
    if G_er.number_of_edges() == 0:
        return 0.0
    return nx.transitivity(G_er.to_undirected())

def write_top_degree_csv(degrees, vocab, output_path, top_k=10):
    idx_to_word = {idx: token for token, idx in vocab.items()}
    entries = [(idx_to_word.get(i, str(i)), degrees[i], i) for i in range(len(degrees))]
    entries.sort(key=lambda x: x[1], reverse=True)
    top_entries = entries[:top_k]
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'degree'])
        for word, degree, _ in top_entries:
            writer.writerow([word, int(degree)])

def edges_to_tensors(edges):
    edge_list = list(edges)
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    edge_index = torch.tensor([[src, dst] for src, dst, _ in edge_list], dtype=torch.long).t().contiguous()
    edge_type = torch.tensor([etype for _, _, etype in edge_list], dtype=torch.long)
    return edge_index, edge_type

def convert_edge_records(edge_records):
    return edges_to_tensors(edge_records)

def build_edges_from_terms(terms, vocab):
    """
    Build edge set from a list of terms using the same rule as build_rgnn_graph.
    Returns a set of (src, dst, etype) to easily deduplicate.
    """
    edge_set = set()
    for term in terms:
        edge_set.update(iter_edges_for_term(term, vocab))
    return edge_set

def edge_set_to_tensors(edge_set):
    return edges_to_tensors(edge_set)

def deduplicate_edge_sets(train_set, dev_set, test_set):
    """
    Remove overlaps between splits to avoid leakage.
    Dev removes anything seen in train.
    Test removes anything seen in train or the cleaned dev.
    Returns cleaned sets and stats dict.
    """
    stats = {}
    stats['dev_overlap_train'] = len(dev_set & train_set)
    stats['test_overlap_train'] = len(test_set & train_set)
    stats['test_overlap_dev'] = len(test_set & dev_set)

    dev_clean = dev_set - train_set
    test_clean = test_set - train_set - dev_clean

    stats['train_count'] = len(train_set)
    stats['dev_count_before'] = len(dev_set)
    stats['dev_count_after'] = len(dev_clean)
    stats['test_count_before'] = len(test_set)
    stats['test_count_after'] = len(test_clean)

    return train_set, dev_clean, test_clean, stats

def summarize_edge_split(label, edge_index, edge_type):
    count = edge_index.size(1) if edge_index.dim() == 2 else 0
    print(f"[{label}] Edge count: {count}")
    if edge_type.numel() == 0:
        print(f"[{label}] No edges available.")
        return
    unique, counts = torch.unique(edge_type, return_counts=True)
    for rel, cnt in zip(unique.tolist(), counts.tolist()):
        print(f"  Type {display_edge_type(rel)}: {cnt}")

def split_edges_by_terms(terms, vocab, seed=42, train_ratio=0.8, val_ratio=0.1):
    train_ratio = float(train_ratio)
    val_ratio = float(val_ratio)
    test_ratio = 1.0 - train_ratio - val_ratio
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    term_indices = list(range(len(terms)))
    rng = random.Random(seed)
    rng.shuffle(term_indices)
    n_train = int(len(term_indices) * train_ratio)
    n_val = int(len(term_indices) * val_ratio)
    term_to_split = {}
    for idx in term_indices[:n_train]:
        term_to_split[idx] = 'train'
    for idx in term_indices[n_train:n_train + n_val]:
        term_to_split[idx] = 'val'
    for idx in term_indices[n_train + n_val:]:
        term_to_split[idx] = 'test'

    term_edges = [list(iter_edges_for_term(term, vocab)) for term in terms]

    train_edges = []
    val_edges = []
    test_edges = []
    train_edge_set = set()
    val_edge_set = set()

    train_term_idxs = [i for i in range(len(term_edges)) if term_to_split.get(i) == 'train']
    val_term_idxs = [i for i in range(len(term_edges)) if term_to_split.get(i) == 'val']
    test_term_idxs = [i for i in range(len(term_edges)) if term_to_split.get(i) == 'test']

    for term_idx in train_term_idxs:
        for edge in term_edges[term_idx]:
            train_edges.append(edge)
            train_edge_set.add(edge)

    for term_idx in val_term_idxs:
        for edge in term_edges[term_idx]:
            if edge in train_edge_set:
                continue
            val_edges.append(edge)
            val_edge_set.add(edge)

    for term_idx in test_term_idxs:
        for edge in term_edges[term_idx]:
            if edge in train_edge_set or edge in val_edge_set:
                continue
            test_edges.append(edge)

    return convert_edge_records(train_edges), convert_edge_records(val_edges), convert_edge_records(test_edges), term_to_split

def analyze_graph(edge_index, num_nodes, label, metrics_dir, seed):
    edge_index = edge_index.cpu()
    num_edges = edge_index.size(1) if edge_index.dim() == 2 else 0
    density = compute_density(num_nodes, num_edges)
    degrees, degree_hist = compute_degree_distribution(edge_index, num_nodes)

    bar_plot_path = os.path.join(metrics_dir, f"{label}_degree_distribution.png")
    plot_degree_distribution(
        degree_hist,
        f"{label.replace('_', ' ').title()} Degree Distribution",
        bar_plot_path
    )

    loglog_plot_path = os.path.join(metrics_dir, f"{label}_degree_loglog.png")
    plot_degree_loglog(
        degree_hist,
        f"{label.replace('_', ' ').title()} Degree Distribution (Log-Log)",
        loglog_plot_path
    )

    clustering_coeff = compute_global_clustering(edge_index, num_nodes)
    er_clustering = compute_er_clustering(num_nodes, num_edges, seed)

    print(f"\n[{label}] Number of Nodes: {num_nodes}")
    print(f"[{label}] Number of Edges: {num_edges}")
    print(f"[{label}] Network Density: {density:.6f}")
    print(f"[{label}] Global Clustering Coefficient: {clustering_coeff:.6f}")
    print(f"[{label}] Erdos-Renyi Clustering (same N/E): {er_clustering:.6f}")
    print(f"[{label}] Degree distribution plot saved to {bar_plot_path}")
    print(f"[{label}] Log-log degree scatter saved to {loglog_plot_path}")

    metrics = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'global_clustering_coefficient': clustering_coeff,
        'er_global_clustering_coefficient': er_clustering,
        'degree_distribution': degree_hist,
        'degree_distribution_plot': bar_plot_path,
        'degree_distribution_loglog_plot': loglog_plot_path
    }

    return metrics, degrees

def read_terms_from_csv(path):
    """
    Read terms from a CSV/TXT file.
    - If a 'term' column exists, use it.
    - If the file is single-column without a header, treat it as a term list.
    """
    header_df = pd.read_csv(path)
    if 'term' in header_df.columns:
        return header_df['term'].dropna().astype(str).tolist()

    raw_df = pd.read_csv(path, header=None)
    if raw_df.shape[1] == 1:
        terms = raw_df.iloc[:, 0].dropna().astype(str)
        # Avoid treating a plain "term" header as data.
        if not terms.empty and terms.iloc[0].strip().lower() == 'term':
            terms = terms.iloc[1:]
        return terms.tolist()
    raise ValueError(f"'term' column not found and file is not single-column: {path}")

def main():
    parser = argparse.ArgumentParser(description='Create dataset for RGNN')
    parser.add_argument('--data_path', type=str, default='./data/lise_difficulty.csv', help='Path to source CSV')
    parser.add_argument('--embedding_path', type=str, default='./data/cc.en.300.bin', help='Path to FastText model')
    parser.add_argument('--use_embedding', action='store_true', help='Use FastText embeddings')
    parser.add_argument('--output_path', type=str, default='./RGNN/data/dataset/processed_dataset.pt', help='Output path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--split_mode', choices=['edge', 'term'], default='edge', help='Edge or term level splitting')
    parser.add_argument('--train_path', type=str, help='Pre-split train CSV/TXT path')
    parser.add_argument('--dev_path', type=str, help='Pre-split dev CSV/TXT path')
    parser.add_argument('--test_path', type=str, help='Pre-split test CSV/TXT path')
    parser.add_argument('--neg_seed', type=int, default=42, help='Base seed for negative sampling (train/dev/test use offsets)')
    parser.add_argument('--neg_ratio', '--neg_ration', type=int, default=1, help='Negatives per positive edge')
    parser.add_argument('--hard_negative_mode', type=str, default='mixed',
                        choices=['none', 'degree', 'mixed'],
                        help='Hard negative sampling mode: none (uniform), degree (prefer high-degree nodes), mixed (50/50)')
    parser.add_argument('--corrupt_mode', type=str, default='both',
                        choices=['tail', 'head', 'both'],
                        help='Corruption mode: tail (corrupt object), head (corrupt subject), both (random)')
    
    args = parser.parse_args()
    args.output_path = os.path.abspath(args.output_path)
    pre_split_mode = bool(args.train_path and args.dev_path and args.test_path)

    if pre_split_mode:
        print("Using pre-split CSV files (train/dev/test).")
        train_terms = read_terms_from_csv(args.train_path)
        dev_terms = read_terms_from_csv(args.dev_path)
        test_terms = read_terms_from_csv(args.test_path)
        terms = train_terms + dev_terms + test_terms
        print(f"Loaded terms | train: {len(train_terms)}, dev: {len(dev_terms)}, test: {len(test_terms)}, total: {len(terms)}")
    else:
        print(f"Loading data from {args.data_path}...")
        # Adjust path if running from RGNN directory
        if not os.path.exists(args.data_path):
            # Try relative to workspace root if script run from root
            alt_path = args.data_path.replace('../', '')
            if os.path.exists(alt_path):
                args.data_path = alt_path
                
        df = pd.read_csv(args.data_path)
        terms = df['term'].tolist()
        print(f"Loaded {len(terms)} terms.")
    
    emb_model = None
    if args.use_embedding:
        print(f"Loading embeddings from {args.embedding_path}...")
        emb_model = load_fasttext_model(args.embedding_path)
        
    print("Building graph...")
    vocab = build_vocab(terms)
    graph_data = build_rgnn_graph(terms, vocab, emb_model)
    print(f"Nodes: {graph_data.num_nodes}, Edges: {graph_data.edge_index.size(1)}")
    
    print(f"Splitting edges using {args.split_mode} mode..." if not pre_split_mode else "Building edges from pre-split files...")
    term_assignments = None
    if pre_split_mode:
        train_edge_set = build_edges_from_terms(train_terms, vocab)
        dev_edge_set = build_edges_from_terms(dev_terms, vocab)
        test_edge_set = build_edges_from_terms(test_terms, vocab)

        train_edge_set, dev_edge_set, test_edge_set, dup_stats = deduplicate_edge_sets(
            train_edge_set, dev_edge_set, test_edge_set
        )
        print("[dedup] dev overlap with train:", dup_stats['dev_overlap_train'])
        print("[dedup] test overlap with train:", dup_stats['test_overlap_train'])
        print("[dedup] test overlap with dev :", dup_stats['test_overlap_dev'])
        print("[dedup] counts -> train: {train_count}, dev: {dev_before}->{dev_after}, test: {test_before}->{test_after}".format(
            train_count=dup_stats['train_count'],
            dev_before=dup_stats['dev_count_before'],
            dev_after=dup_stats['dev_count_after'],
            test_before=dup_stats['test_count_before'],
            test_after=dup_stats['test_count_after']
        ))

        train_edge_index, train_edge_type = edge_set_to_tensors(train_edge_set)
        val_edge_index, val_edge_type = edge_set_to_tensors(dev_edge_set)
        test_edge_index, test_edge_type = edge_set_to_tensors(test_edge_set)
    else:
        if args.split_mode == 'term':
            train_data, val_data, test_data, term_assignments = split_edges_by_terms(
                terms,
                vocab,
                seed=args.seed
            )
        else:
            train_data, val_data, test_data = split_edges_stratified(
                graph_data.edge_index, 
                graph_data.edge_type,
                seed=args.seed
            )
        train_edge_index, train_edge_type = train_data
        val_edge_index, val_edge_type = val_data
        test_edge_index, test_edge_type = test_data

    summarize_edge_split('train', train_edge_index, train_edge_type)
    summarize_edge_split('val', val_edge_index, val_edge_type)
    summarize_edge_split('test', test_edge_index, test_edge_type)
    
    output_dir = os.path.dirname(args.output_path)
    if not output_dir:
        output_dir = '.'
    os.makedirs(output_dir, exist_ok=True)
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    full_metrics, full_degrees = analyze_graph(
        graph_data.edge_index,
        graph_data.num_nodes,
        'full_graph',
        metrics_dir,
        seed=args.seed
    )

    train_metrics, _ = analyze_graph(
        train_edge_index,
        graph_data.num_nodes,
        'train_graph',
        metrics_dir,
        seed=args.seed + 1
    )

    top_degree_csv_path = os.path.join(metrics_dir, 'full_graph_top_degrees.csv')
    write_top_degree_csv(full_degrees, vocab, top_degree_csv_path, top_k=10)
    full_metrics['top_degree_csv'] = top_degree_csv_path
    print(f"[full_graph] Top-degree CSV saved to {top_degree_csv_path}")

    if (not pre_split_mode) and args.split_mode == 'term' and term_assignments is not None:
        term_split_csv = os.path.join(metrics_dir, 'term_split.csv')
        with open(term_split_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['term', 'split'])
            for idx, term in enumerate(terms):
                writer.writerow([term, term_assignments.get(idx, 'train')])
        print(f"[term_split] CSV saved to {term_split_csv}")

    print("Sampling negatives...")
    print(f"  Hard negative mode: {args.hard_negative_mode}")
    print(f"  Corrupt mode: {args.corrupt_mode}")
    # Use all positive edges across splits to avoid sampling positives from other splits as negatives
    all_pos_edges = [
        (train_edge_index, train_edge_type),
        (val_edge_index, val_edge_type),
        (test_edge_index, test_edge_type)
    ]
    # Train negatives
    train_neg_index, train_neg_type, train_neg_stats = sample_negative_triplets(
        train_edge_index,
        train_edge_type,
        graph_data.num_nodes,
        seed=args.neg_seed,
        extra_positive_edges=all_pos_edges,
        return_stats=True,
        neg_ratio=args.neg_ratio,
        hard_negative_mode=args.hard_negative_mode,
        corrupt_mode=args.corrupt_mode
    )
    
    # Val negatives
    val_neg_index, val_neg_type, val_neg_stats = sample_negative_triplets(
        val_edge_index,
        val_edge_type,
        graph_data.num_nodes,
        seed=args.neg_seed + 1,
        extra_positive_edges=all_pos_edges,
        return_stats=True,
        neg_ratio=args.neg_ratio,
        hard_negative_mode=args.hard_negative_mode,
        corrupt_mode=args.corrupt_mode
    )
    
    # Test negatives
    test_neg_index, test_neg_type, test_neg_stats = sample_negative_triplets(
        test_edge_index,
        test_edge_type,
        graph_data.num_nodes,
        seed=args.neg_seed + 2,
        extra_positive_edges=all_pos_edges,
        return_stats=True,
        neg_ratio=args.neg_ratio,
        hard_negative_mode=args.hard_negative_mode,
        corrupt_mode=args.corrupt_mode
    )
    print(
        "Negative sampling fallback counts "
        f"(train/val/test): {train_neg_stats['fallback_count']}/{val_neg_stats['fallback_count']}/{test_neg_stats['fallback_count']}"
    )
    print(
        "Negative sampling head/tail counts "
        f"(train): {train_neg_stats['head_corrupt_count']}/{train_neg_stats['tail_corrupt_count']}, "
        f"(val): {val_neg_stats['head_corrupt_count']}/{val_neg_stats['tail_corrupt_count']}, "
        f"(test): {test_neg_stats['head_corrupt_count']}/{test_neg_stats['tail_corrupt_count']}"
    )
    
    dataset = {
        'vocab': vocab,
        'num_nodes': graph_data.num_nodes,
        'x': graph_data.x,
        'split_mode': 'pre_split' if pre_split_mode else args.split_mode,
        'train': {
            'pos_edge_index': train_edge_index,
            'pos_edge_type': train_edge_type,
            'neg_edge_index': train_neg_index,
            'neg_edge_type': train_neg_type
        },
        'val': {
            'pos_edge_index': val_edge_index,
            'pos_edge_type': val_edge_type,
            'neg_edge_index': val_neg_index,
            'neg_edge_type': val_neg_type
        },
        'test': {
            'pos_edge_index': test_edge_index,
            'pos_edge_type': test_edge_type,
            'neg_edge_index': test_neg_index,
            'neg_edge_type': test_neg_type
        }
    }
    
    dataset['metrics'] = {
        'full_graph': full_metrics,
        'train_graph': train_metrics
    }
    dataset['train_neg_sampling_config'] = {
        'neg_seed': args.neg_seed,
        'neg_ratio': args.neg_ratio,
        'hard_negative_mode': args.hard_negative_mode,
        'corrupt_mode': args.corrupt_mode,
    }

    print(f"Saving to {args.output_path}...")
    torch.save(dataset, args.output_path)
    print("Done.")

if __name__ == '__main__':
    main()
