import argparse
import csv
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.rgcn_distmult import RGCNLinkPredictor


def display_edge_type(edge_type_id):
    """Map internal relation ids to the project-facing Type labels."""
    return int(edge_type_id) + 1


def load_model(model_path, dataset, args, device):
    x = dataset["x"]
    num_features = x.size(1)
    all_edge_types = torch.cat(
        [
            dataset["train"]["pos_edge_type"],
            dataset["val"]["pos_edge_type"],
            dataset["test"]["pos_edge_type"],
        ],
        dim=0,
    )
    if all_edge_types.numel() == 0:
        raise ValueError("No edge types found in dataset.")
    num_relations = int(all_edge_types.max().item()) + 1
    model = RGCNLinkPredictor(
        num_features=num_features,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_relations=num_relations,
        num_layers=args.num_layers,
        dropout=args.dropout,
        encoder_type=args.encoder,
        decoder_type=args.decoder,
        rgat_heads=args.rgat_heads,
        num_bases=args.num_bases,
        compgcn_composition=args.compgcn_composition,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def score_edges(model, z_re, z_im, edge_index, edge_type):
    """Score edges using pre-computed projected embeddings."""
    scores = model.decoder.score(z_re, z_im, edge_index, edge_type)
    return torch.sigmoid(scores)


def summarize_probs(label, values):
    if values.size == 0:
        print(f"{label}: no samples")
        return
    stats = np.percentile(values, [0, 5, 50, 95, 100])
    print(
        f"{label}: n={values.size} mean={values.mean():.4f} std={values.std():.4f} "
        f"min={stats[0]:.4f} p5={stats[1]:.4f} median={stats[2]:.4f} p95={stats[3]:.4f} max={stats[4]:.4f}"
    )


def build_id_to_word(vocab):
    return {int(idx): str(token) for token, idx in vocab.items()}


def write_recovered_edges(path, edge_index, edge_type, probs, mask, id_to_word):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["src_id", "dst_id", "src_word", "dst_word", "edge_type", "probability"])
        if edge_index.numel() == 0:
            return 0
        indices = np.where(mask)[0]
        for i in indices:
            src_id = int(edge_index[0, i])
            dst_id = int(edge_index[1, i])
            writer.writerow(
                [
                    src_id,
                    dst_id,
                    id_to_word.get(src_id, str(src_id)),
                    id_to_word.get(dst_id, str(dst_id)),
                    display_edge_type(edge_type[i]),
                    f"{float(probs[i]):.6f}",
                ]
            )
    return int(np.sum(mask))


def write_hist_csv(path, bin_edges, pos_hist, neg_hist):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_left", "bin_right", "pos_count", "neg_count", "total_count", "pos_ratio"])
        for i in range(len(pos_hist)):
            pos_count = int(pos_hist[i])
            neg_count = int(neg_hist[i])
            total = pos_count + neg_count
            pos_ratio = (pos_count / total) if total > 0 else 0.0
            writer.writerow(
                [
                    f"{bin_edges[i]:.6f}",
                    f"{bin_edges[i + 1]:.6f}",
                    pos_count,
                    neg_count,
                    total,
                    f"{pos_ratio:.6f}",
                ]
            )


def plot_histogram(path, probs, bins, title, color, label):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    if probs.size > 0:
        ax.hist(probs, bins=bins, range=(0.0, 1.0), alpha=0.8, color=color, label=label)
    ax.set_xlabel("Predicted link probability")
    ax.set_ylabel("Count")
    ax.set_title(title)
    if probs.size > 0:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_overlay_histogram(path, pos_probs, neg_probs, bins, title):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    if pos_probs.size > 0:
        ax.hist(pos_probs, bins=bins, range=(0.0, 1.0), alpha=0.6, color="#dd8452", label="pos")
    if neg_probs.size > 0:
        ax.hist(neg_probs, bins=bins, range=(0.0, 1.0), alpha=0.6, color="#4c72b0", label="neg")
    ax.set_xlabel("Predicted link probability")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Output probability distribution on the test split")
    parser.add_argument("--dataset_path", type=str, default="RGNN/data/dataset/processed_dataset.pt")
    parser.add_argument("--model_path", type=str, default="RGNN/data/model/best_model.pt")
    parser.add_argument("--output_csv", type=str, default="results/pos_distribution/test_prob_hist.csv")
    parser.add_argument("--plot_pos_path", type=str, default="results/pos_distribution/test_prob_hist_pos.png")
    parser.add_argument("--plot_neg_path", type=str, default="results/pos_distribution/test_prob_hist_neg.png")
    parser.add_argument("--plot_overlay_path", type=str, default="results/pos_distribution/test_prob_hist_overlay.png")
    
    parser.add_argument("--pos_recovered_csv", type=str, default="results/pos_distribution/test_pos_recovered_edges.csv")
    parser.add_argument("--neg_recovered_csv", type=str, default="results/pos_distribution/test_neg_recovered_edges.csv")
    parser.add_argument("--threshold", type=float, default=None, 
                        help="Threshold for edge prediction. If not set, uses optimal_threshold from dataset or 0.5")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--encoder", choices=["rgcn", "rgat", "compgcn"], default="rgcn")
    parser.add_argument("--decoder", choices=["distmult", "complex"], default="complex")
    parser.add_argument("--rgat_heads", type=int, default=4)
    parser.add_argument("--num_bases", type=int, default=None, help="Number of bases for RGCN")
    parser.add_argument("--compgcn_composition", choices=["mult", "sub"], default="mult")

    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        print("CUDA not available, switching to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    dataset = torch.load(args.dataset_path, map_location=device, weights_only=False)
    if "test" not in dataset:
        raise ValueError("Dataset does not contain a test split.")
    test_set = dataset["test"]

    x = dataset["x"].to(device)
    model = load_model(args.model_path, dataset, args, device)

    msg_edge_index = dataset["train"]["pos_edge_index"].to(device)
    msg_edge_type = dataset["train"]["pos_edge_type"].to(device)

    pos_edge_index = test_set["pos_edge_index"].to(device)
    pos_edge_type = test_set["pos_edge_type"].to(device)
    neg_edge_index = test_set["neg_edge_index"].to(device)
    neg_edge_type = test_set["neg_edge_type"].to(device)

    with torch.no_grad():
        # Encode once using training graph
        z = model.encoder(x, msg_edge_index, msg_edge_type)
        z_re, z_im = model.decoder.project(z)
        
        # Score positive and negative edges
        pos_probs = score_edges(model, z_re, z_im, pos_edge_index, pos_edge_type).cpu().numpy()
        if neg_edge_index.numel() == 0:
            neg_probs = np.array([], dtype=np.float32)
        else:
            neg_probs = score_edges(model, z_re, z_im, neg_edge_index, neg_edge_type).cpu().numpy()

    print("Test probability distribution summary:")
    summarize_probs("pos", pos_probs)
    summarize_probs("neg", neg_probs)

    bin_edges = np.linspace(0.0, 1.0, args.bins + 1)
    pos_hist, _ = np.histogram(pos_probs, bins=bin_edges)
    neg_hist, _ = np.histogram(neg_probs, bins=bin_edges)

    write_hist_csv(args.output_csv, bin_edges, pos_hist, neg_hist)
    print(f"Histogram CSV saved to {args.output_csv}")

    # Determine threshold: use provided, or from dataset, or default 0.5
    if args.threshold is not None:
        threshold = args.threshold
        print(f"Using provided threshold: {threshold:.4f}")
    elif "optimal_threshold" in dataset:
        threshold = float(dataset["optimal_threshold"])
        print(f"Using optimal threshold from dataset: {threshold:.4f}")
    else:
        threshold = 0.5
        print(f"Using default threshold: {threshold:.4f}")

    id_to_word = build_id_to_word(dataset.get("vocab", {}))
    pos_mask = pos_probs >= threshold
    neg_mask = neg_probs >= threshold
    if args.pos_recovered_csv:
        recovered_pos = write_recovered_edges(
            args.pos_recovered_csv,
            pos_edge_index.cpu(),
            pos_edge_type.cpu(),
            pos_probs,
            pos_mask,
            id_to_word,
        )
        print(f"Recovered pos edges saved to {args.pos_recovered_csv} (count: {recovered_pos})")
    if args.neg_recovered_csv:
        recovered_neg = write_recovered_edges(
            args.neg_recovered_csv,
            neg_edge_index.cpu(),
            neg_edge_type.cpu(),
            neg_probs,
            neg_mask,
            id_to_word,
        )
        print(f"Faulty recovered neg edges saved to {args.neg_recovered_csv} (count: {recovered_neg})")

    if args.plot_pos_path:
        plot_histogram(
            args.plot_pos_path,
            pos_probs,
            args.bins,
            "Test probability distribution (pos)",
            "#dd8452",
            "pos",
        )
        print(f"Pos histogram plot saved to {args.plot_pos_path}")
    if args.plot_neg_path:
        plot_histogram(
            args.plot_neg_path,
            neg_probs,
            args.bins,
            "Test probability distribution (neg)",
            "#4c72b0",
            "neg",
        )
        print(f"Neg histogram plot saved to {args.plot_neg_path}")
    if args.plot_overlay_path:
        plot_overlay_histogram(
            args.plot_overlay_path,
            pos_probs,
            neg_probs,
            args.bins,
            "Test probability distribution (pos vs neg)",
        )
        print(f"Overlay histogram plot saved to {args.plot_overlay_path}")


if __name__ == "__main__":
    main()
