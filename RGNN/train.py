import argparse
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.rgcn_distmult import RGCNLinkPredictor
from data.dataset import sample_negative_triplets


def display_edge_type(edge_type_id):
    """Map internal relation ids to the project-facing Type labels."""
    return int(edge_type_id) + 1


def train(model, x, train_set, optimizer, device, loss_type='bce_balanced', label_smoothing=0.0, edge_dropout=0.0):
    """
    Train one epoch.
    
    loss_type options:
    - 'bce': Standard BCE loss
    - 'bce_balanced': BCE with pos_weight to balance classes (recommended)
    - 'bce_rgcn': Original RGCN paper loss with coefficient
    - 'margin': Margin ranking loss (alternative approach)
    
    label_smoothing: If > 0, apply label smoothing (e.g., 0.1 -> labels become 0.1/0.9)
    edge_dropout: If > 0, randomly drop edges from message passing graph to prevent overfitting/leakage
    """
    model.train()
    optimizer.zero_grad()
    
    pos_edge_index = train_set['pos_edge_index'].to(device)
    pos_edge_type = train_set['pos_edge_type'].to(device)
    neg_edge_index = train_set['neg_edge_index'].to(device)
    neg_edge_type = train_set['neg_edge_type'].to(device)
    
    # Message passing edges (potentially with dropout)
    msg_edge_index = pos_edge_index
    msg_edge_type = pos_edge_type
    
    if edge_dropout > 0.0:
        # Drop message-passing edges while keeping edge types aligned.
        num_edges = pos_edge_index.size(1)
        mask = torch.rand(num_edges, device=device) > edge_dropout
        if mask.any():
            msg_edge_index = pos_edge_index[:, mask]
            msg_edge_type = pos_edge_type[mask]
    
    # Encode using (potentially dropped-out) training positive edges for message passing
    z = model.encoder(x, msg_edge_index, msg_edge_type)
    
    # Project embeddings (for ComplEx, creates re/im; for DistMult, identity)
    z_re, z_im = model.decoder.project(z)
    
    # Decode for both positive and negative edges
    scores_pos = model.decoder.score(z_re, z_im, pos_edge_index, pos_edge_type)
    scores_neg = model.decoder.score(z_re, z_im, neg_edge_index, neg_edge_type)
    
    pos_count = scores_pos.size(0)
    neg_count = scores_neg.size(0)
    if pos_count == 0:
        raise ValueError("No positive samples available to compute loss.")

    if loss_type == 'margin':
        # Margin Ranking Loss: encourages pos scores > neg scores by margin
        # Repeat pos/neg to create pairs
        min_count = min(pos_count, neg_count)
        scores_pos_paired = scores_pos[:min_count]
        scores_neg_paired = scores_neg[:min_count]
        target = torch.ones(min_count, device=device)
        loss = nn.MarginRankingLoss(margin=1.0)(scores_pos_paired, scores_neg_paired, target)
    else:
        predictions = torch.cat([scores_pos, scores_neg])
        # Apply label smoothing: 0 -> label_smoothing, 1 -> 1 - label_smoothing
        pos_label = 1.0 - label_smoothing
        neg_label = label_smoothing
        labels = torch.cat([
            torch.full_like(scores_pos, pos_label),
            torch.full_like(scores_neg, neg_label)
        ])
        
        if loss_type == 'bce_balanced':
            # Balanced BCE: weight positive class by neg/pos ratio
            pos_weight = torch.tensor([float(neg_count) / float(pos_count)], device=device)
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(predictions, labels)
        elif loss_type == 'bce_rgcn':
            # Original RGCN paper loss with coefficient (can cause vanishing gradients)
            E = float(pos_count)
            w = float(neg_count) / E if neg_count > 0 else 0.0
            coeff = 1.0 / ((1.0 + w) * E)
            loss = nn.BCEWithLogitsLoss()(predictions, labels) * coeff
        else:  # 'bce'
            # Standard BCE loss
            loss = nn.BCEWithLogitsLoss()(predictions, labels)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def compute_binary_metrics(scores_pos, scores_neg, threshold=0.5):
    if scores_pos.numel() == 0 or scores_neg.numel() == 0:
        return None
    y_true = torch.cat([torch.ones(scores_pos.size(0)), torch.zeros(scores_neg.size(0))]).cpu().numpy()
    y_scores = torch.cat([torch.sigmoid(scores_pos), torch.sigmoid(scores_neg)]).cpu().numpy()
    y_pred = (y_scores > threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
    except ValueError:
        return None
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        'auc': float(auc),
        'ap': float(ap),
        'acc': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'opt_threshold': float(threshold)
    }


def build_positive_filter_maps(*splits):
    """Build lookup maps for filtered ranking evaluation."""
    tails_by_head_rel = {}
    heads_by_rel_tail = {}
    for split_set in splits:
        edge_index = split_set['pos_edge_index']
        edge_type = split_set['pos_edge_type']
        for i in range(edge_index.size(1)):
            head = int(edge_index[0, i].item())
            tail = int(edge_index[1, i].item())
            rel = int(edge_type[i].item())
            tails_by_head_rel.setdefault((head, rel), set()).add(tail)
            heads_by_rel_tail.setdefault((rel, tail), set()).add(head)
    return tails_by_head_rel, heads_by_rel_tail


def _filtered_rank(scores, true_index, filtered_indices):
    """Return a realistic filtered rank for one corruption side."""
    scores = scores.clone()
    for idx in filtered_indices:
        if idx != true_index:
            scores[idx] = -torch.inf

    true_score = scores[true_index]
    greater_count = torch.sum(scores > true_score).item()
    equal_count = torch.sum(scores == true_score).item()
    return 1.0 + float(greater_count) + 0.5 * float(max(equal_count - 1, 0))


def compute_ranking_metrics(
    model,
    z_re,
    z_im,
    pos_edge_index,
    pos_edge_type,
    num_nodes,
    device,
    positive_filter_maps,
):
    """
    Compute filtered MRR and Hits@K by corrupting both head and tail nodes.
    """
    if pos_edge_index.numel() == 0:
        return None

    tails_by_head_rel, heads_by_rel_tail = positive_filter_maps
    nodes = torch.arange(num_nodes, device=device)
    ranks = []

    for i in range(pos_edge_index.size(1)):
        head = int(pos_edge_index[0, i].item())
        tail = int(pos_edge_index[1, i].item())
        rel = int(pos_edge_type[i].item())

        rel_types = torch.full((num_nodes,), rel, dtype=torch.long, device=device)

        # Tail corruption: (head, rel, ?)
        tail_edges = torch.stack([torch.full_like(nodes, head), nodes], dim=0)
        tail_scores = model.decoder.score(z_re, z_im, tail_edges, rel_types)
        tail_filters = tails_by_head_rel.get((head, rel), set())
        ranks.append(_filtered_rank(tail_scores, tail, tail_filters))

        # Head corruption: (?, rel, tail)
        head_edges = torch.stack([nodes, torch.full_like(nodes, tail)], dim=0)
        head_scores = model.decoder.score(z_re, z_im, head_edges, rel_types)
        head_filters = heads_by_rel_tail.get((rel, tail), set())
        ranks.append(_filtered_rank(head_scores, head, head_filters))

    ranks = np.asarray(ranks, dtype=np.float64)
    return {
        'mrr': float(np.mean(1.0 / ranks)),
        'hits_at_1': float(np.mean(ranks <= 1)),
        'hits_at_5': float(np.mean(ranks <= 5)),
        'hits_at_10': float(np.mean(ranks <= 10)),
    }


def merge_metric_dicts(binary_metrics, ranking_metrics):
    if binary_metrics is None and ranking_metrics is None:
        return None
    merged = {} if binary_metrics is None else dict(binary_metrics)
    if ranking_metrics is not None:
        merged.update(ranking_metrics)
    return merged


def find_optimal_threshold(scores_pos, scores_neg, metric='f1', num_thresholds=100):
    """
    Find the optimal threshold that maximizes the specified metric on validation set.
    
    Args:
        scores_pos: Positive sample scores (logits)
        scores_neg: Negative sample scores (logits)
        metric: Metric to optimize ('f1', 'acc', 'youden')
        num_thresholds: Number of threshold candidates to try
    
    Returns:
        optimal_threshold, best_metric_value
    """
    if scores_pos.numel() == 0 or scores_neg.numel() == 0:
        return 0.5, 0.0
    
    y_true = torch.cat([torch.ones(scores_pos.size(0)), torch.zeros(scores_neg.size(0))]).cpu().numpy()
    y_scores = torch.cat([torch.sigmoid(scores_pos), torch.sigmoid(scores_neg)]).cpu().numpy()
    
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_threshold = 0.5
    best_value = -1.0
    
    for thresh in thresholds:
        y_pred = (y_scores > thresh).astype(int)
        
        if metric == 'f1':
            value = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'acc':
            value = accuracy_score(y_true, y_pred)
        elif metric == 'youden':
            # Youden's J statistic: TPR - FPR = sensitivity + specificity - 1
            tp = np.sum((y_pred == 1) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            value = tpr - fpr
        else:
            value = f1_score(y_true, y_pred, zero_division=0)
        
        if value > best_value:
            best_value = value
            best_threshold = thresh
    
    return best_threshold, best_value

def count_edges_by_type(edge_type):
    if edge_type.numel() == 0:
        return {}
    unique, counts = torch.unique(edge_type.cpu(), return_counts=True)
    return {int(u.item()): int(c.item()) for u, c in zip(unique, counts)}

def log_edge_counts(split_name, split_set):
    counts = count_edges_by_type(split_set['pos_edge_type'])
    print(f"{split_name.capitalize()} positive edge counts by type:")
    if not counts:
        print("  (no edges)")
    else:
        for rel in sorted(counts.keys()):
            print(f"  Type {display_edge_type(rel)}: {counts[rel]}")
    print()

def print_per_type_metrics(split_label, per_type_metrics):
    for rel in sorted(per_type_metrics.keys()):
        rel_metrics = per_type_metrics[rel]
        if rel_metrics is None:
            print(f"    {split_label} Type {display_edge_type(rel)}: insufficient data for metrics")
        else:
            ranking_msg = ""
            if 'mrr' in rel_metrics:
                ranking_msg = (
                    f" | MRR: {rel_metrics['mrr']:.4f} | "
                    f"H@1: {rel_metrics['hits_at_1']:.4f} | "
                    f"H@5: {rel_metrics['hits_at_5']:.4f} | "
                    f"H@10: {rel_metrics['hits_at_10']:.4f}"
                )
            print(
                f"    {split_label} Type {display_edge_type(rel)} | AUC: {rel_metrics['auc']:.4f} | AP: {rel_metrics['ap']:.4f} | "
                f"Acc: {rel_metrics['acc']:.4f} | P: {rel_metrics['precision']:.4f} | "
                f"R: {rel_metrics['recall']:.4f} | F1: {rel_metrics['f1']:.4f}{ranking_msg}"
            )

def save_test_metrics_csv(test_overall, test_per_type, output_path):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'relation_type',
            'auc',
            'ap',
            'acc',
            'precision',
            'recall',
            'f1',
            'opt_threshold',
            'mrr',
            'hits_at_1',
            'hits_at_5',
            'hits_at_10',
        ])

        def metrics_row(rel_label, metrics):
            if metrics is None:
                return [rel_label, '', '', '', '', '', '', '', '', '', '', '']
            return [
                rel_label,
                f"{metrics['auc']:.4f}" if 'auc' in metrics else '',
                f"{metrics['ap']:.4f}" if 'ap' in metrics else '',
                f"{metrics['acc']:.4f}" if 'acc' in metrics else '',
                f"{metrics['precision']:.4f}" if 'precision' in metrics else '',
                f"{metrics['recall']:.4f}" if 'recall' in metrics else '',
                f"{metrics['f1']:.4f}" if 'f1' in metrics else '',
                f"{metrics['opt_threshold']:.6f}" if 'opt_threshold' in metrics else '',
                f"{metrics['mrr']:.6f}" if 'mrr' in metrics else '',
                f"{metrics['hits_at_1']:.6f}" if 'hits_at_1' in metrics else '',
                f"{metrics['hits_at_5']:.6f}" if 'hits_at_5' in metrics else '',
                f"{metrics['hits_at_10']:.6f}" if 'hits_at_10' in metrics else '',
            ]

        writer.writerow(metrics_row('overall', test_overall))
        for rel in sorted(test_per_type.keys()):
            writer.writerow(metrics_row(str(display_edge_type(rel)), test_per_type[rel]))

@torch.no_grad()
def evaluate(model, x, msg_edge_index, msg_edge_type, eval_set, device):
    model.eval()
    
    pos_edge_index = eval_set['pos_edge_index'].to(device)
    pos_edge_type = eval_set['pos_edge_type'].to(device)
    neg_edge_index = eval_set['neg_edge_index'].to(device)
    neg_edge_type = eval_set['neg_edge_type'].to(device)
    
    # Encode once using training graph for message passing
    z = model.encoder(x, msg_edge_index, msg_edge_type)
    
    # Project embeddings (for ComplEx, creates re/im; for DistMult, identity)
    z_re, z_im = model.decoder.project(z)
    
    # Decode for both positive and negative edges
    scores_pos = model.decoder.score(z_re, z_im, pos_edge_index, pos_edge_type)
    scores_neg = model.decoder.score(z_re, z_im, neg_edge_index, neg_edge_type)
    
    overall_metrics = compute_binary_metrics(scores_pos, scores_neg)
    per_type_metrics = {}
    combined_types = torch.unique(torch.cat([pos_edge_type, neg_edge_type], dim=0)).cpu().tolist()
    for rel in combined_types:
        rel = int(rel)
        pos_mask = (pos_edge_type == rel)
        neg_mask = (neg_edge_type == rel)
        rel_metrics = compute_binary_metrics(scores_pos[pos_mask], scores_neg[neg_mask])
        per_type_metrics[rel] = rel_metrics
    
    return overall_metrics, per_type_metrics

def main():
    parser = argparse.ArgumentParser(description='Train RGNN Link Predictor')
    parser.add_argument('--dataset_path', type=str, default='RGNN/data/dataset/processed_dataset.pt', help='Path to processed dataset')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for training')
    parser.add_argument('--encoder', choices=['rgcn', 'rgat', 'compgcn'], default='rgcn', help='Encoder architecture')
    parser.add_argument('--num_bases', type=int, default=None, help='Number of bases for RGCN basis decomposition (default: None)')
    parser.add_argument('--decoder', choices=['distmult', 'complex'], default='complex', help='Decoder type')
    parser.add_argument('--compgcn_composition', choices=['mult', 'sub'], default='mult',
                        help='Composition operation for CompGCN encoder')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint_path', type=str, default='RGNN/data/model/best_model.pt', help='Path to save best dev model')
    parser.add_argument('--final_model_path', type=str, default='RGNN/data/model/final_model.pt', help='Path to save final epoch model')
    parser.add_argument('--no_final_model', action='store_true', help='Disable saving final epoch model')
    parser.add_argument('--eval_every', type=int, default=10, help='Evaluate dev every N epochs')
    parser.add_argument('--test_metrics_csv', type=str, default='results/test_metrics.csv', help='Path to save test metrics CSV (default: alongside dataset)')
    
    # New arguments for better training
    parser.add_argument('--loss_type', type=str, default='bce_balanced',
                        choices=['bce', 'bce_balanced', 'bce_rgcn', 'margin'],
                        help='Loss function type: bce (standard), bce_balanced (recommended), bce_rgcn (original), margin')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 regularization weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Early stopping patience in number of evaluations (0 to disable)')
    parser.add_argument('--lr_scheduler', action='store_true', help='Use ReduceLROnPlateau scheduler')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--edge_dropout', type=float, default=0.0, help='Edge dropout rate for training (default: 0.0)')
    parser.add_argument('--neg_resample_every', type=int, default=5, 
                        help='Resample train negatives every N epochs (0 to disable)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing factor (e.g., 0.1 -> labels become 0.1/0.9)')
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Using device: {args.device}")
    print(f"Loss type: {args.loss_type}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Edge dropout: {args.edge_dropout}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    if args.label_smoothing > 0:
        print(f"Label smoothing: {args.label_smoothing} (pos={1-args.label_smoothing}, neg={args.label_smoothing})")
    
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
        
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = torch.load(args.dataset_path)
    
    x = dataset['x'].to(args.device)
    num_features = x.size(1)
    
    # Dynamically determine number of relations from dataset
    all_edge_types = torch.cat([
        dataset['train']['pos_edge_type'],
        dataset['val']['pos_edge_type'],
        dataset['test']['pos_edge_type']
    ])
    num_relations = int(all_edge_types.max().item()) + 1
    print(f"Number of relations detected: {num_relations}")
    
    model = RGCNLinkPredictor(
        num_features=num_features,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_relations=num_relations,
        encoder_type=args.encoder,
        decoder_type=args.decoder,
        dropout=args.dropout,
        num_bases=args.num_bases,
        compgcn_composition=args.compgcn_composition,
    ).to(args.device)
    print(f"Encoder: {args.encoder}, Decoder: {args.decoder}")
    
    # Add weight decay for L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Optional learning rate scheduler
    scheduler = None
    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
    
    train_set = dataset['train']
    val_set = dataset['val']
    test_set = dataset['test']
    positive_filter_maps = build_positive_filter_maps(train_set, val_set, test_set)
    
    # Get negative sampling config for dynamic train negative resampling
    train_neg_config = dataset.get('train_neg_sampling_config', None)
    if train_neg_config is not None and args.neg_resample_every > 0:
        print(f"Train negative resampling enabled: every {args.neg_resample_every} epochs")
        print(f"  neg_ratio: {train_neg_config['neg_ratio']}, "
              f"hard_negative_mode: {train_neg_config['hard_negative_mode']}, "
              f"corrupt_mode: {train_neg_config['corrupt_mode']}")
        # Build all_pos_edges for avoiding positive edges as negatives
        all_pos_edges = [
            (train_set['pos_edge_index'], train_set['pos_edge_type']),
            (val_set['pos_edge_index'], val_set['pos_edge_type']),
            (test_set['pos_edge_index'], test_set['pos_edge_type'])
        ]
    else:
        if args.neg_resample_every > 0:
            print("Warning: train_neg_sampling_config not found in dataset. "
                  "Negative resampling disabled.")
        args.neg_resample_every = 0
        all_pos_edges = None
    
    log_edge_counts('train', train_set)
    log_edge_counts('test', test_set)

    best_val_auc = float('-inf')
    best_epoch = None
    evals_without_improvement = 0
    best_model_path = args.checkpoint_path

    print("Starting training...")
    msg_edge_index = train_set['pos_edge_index'].to(args.device)
    msg_edge_type = train_set['pos_edge_type'].to(args.device)
    
    for epoch in range(1, args.epochs + 1):
        # Resample train negatives every N epochs
        if args.neg_resample_every > 0 and (epoch == 1 or epoch % args.neg_resample_every == 0):
            resample_seed = train_neg_config['neg_seed'] + (epoch // args.neg_resample_every)
            new_neg_index, new_neg_type = sample_negative_triplets(
                train_set['pos_edge_index'],
                train_set['pos_edge_type'],
                dataset['num_nodes'],
                seed=resample_seed,
                extra_positive_edges=all_pos_edges,
                return_stats=False,
                neg_ratio=train_neg_config['neg_ratio'],
                hard_negative_mode=train_neg_config['hard_negative_mode'],
                corrupt_mode=train_neg_config['corrupt_mode']
            )
            train_set['neg_edge_index'] = new_neg_index
            train_set['neg_edge_type'] = new_neg_type
            print(f"Epoch {epoch:03d} | Resampled train negatives (seed={resample_seed})")
        
        loss = train(model, x, train_set, optimizer, args.device, 
                     loss_type=args.loss_type, label_smoothing=args.label_smoothing,
                     edge_dropout=args.edge_dropout)
        
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_eval:
            val_overall, val_per_type = evaluate(model, x, msg_edge_index, msg_edge_type, val_set, args.device)
            if val_overall is not None:
                overall_msg = (
                    f"Val AUC: {val_overall['auc']:.4f} | Val AP: {val_overall['ap']:.4f} | "
                    f"Val Acc: {val_overall['acc']:.4f} | Val P: {val_overall['precision']:.4f} | "
                    f"Val R: {val_overall['recall']:.4f} | Val F1: {val_overall['f1']:.4f}"
                )
                
                # Update learning rate scheduler if enabled
                if scheduler is not None:
                    scheduler.step(val_overall['auc'])
            else:
                overall_msg = "Val metrics unavailable"
            print(f"Epoch {epoch:03d} | Loss: {loss:.6f} | {overall_msg}")
            print_per_type_metrics("Val", val_per_type)
            
            if val_overall is not None and val_overall['auc'] > best_val_auc:
                best_val_auc = val_overall['auc']
                best_epoch = epoch
                evals_without_improvement = 0
                os.makedirs(os.path.dirname(best_model_path) or '.', exist_ok=True)
                torch.save({'model_state': model.state_dict()}, best_model_path)
                print(f"    Saved new best model (epoch {epoch}) to {best_model_path}")
            elif val_overall is not None:
                evals_without_improvement += 1
                
            # Early stopping check
            if args.early_stopping_patience > 0 and evals_without_improvement >= args.early_stopping_patience:
                print(f"\nEarly stopping triggered after {evals_without_improvement} evaluations without improvement.")
                break
            
    print("\nTraining finished.")
    if args.no_final_model:
        print("Final epoch model save disabled.")
    else:
        # Save final epoch weights separately
        final_model_path = args.final_model_path
        os.makedirs(os.path.dirname(final_model_path) or '.', exist_ok=True)
        torch.save({'model_state': model.state_dict()}, final_model_path)
        print(f"Final epoch model saved to {final_model_path}")

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=args.device)
        state_dict = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        loaded_epoch = best_epoch if best_epoch is not None else "N/A"
        print(f"Loaded best dev model from {best_model_path} (epoch {loaded_epoch})")
    else:
        print("Best model checkpoint not found; proceeding with final epoch weights.")

    # Find optimal threshold on validation set
    print("\nFinding optimal threshold on validation set...")
    model.eval()
    with torch.no_grad():
        val_pos_edge_index = val_set['pos_edge_index'].to(args.device)
        val_pos_edge_type = val_set['pos_edge_type'].to(args.device)
        val_neg_edge_index = val_set['neg_edge_index'].to(args.device)
        val_neg_edge_type = val_set['neg_edge_type'].to(args.device)
        
        z = model.encoder(x, msg_edge_index, msg_edge_type)
        z_re, z_im = model.decoder.project(z)
        
        val_scores_pos = model.decoder.score(z_re, z_im, val_pos_edge_index, val_pos_edge_type)
        val_scores_neg = model.decoder.score(z_re, z_im, val_neg_edge_index, val_neg_edge_type)
        
        optimal_threshold, best_f1 = find_optimal_threshold(val_scores_pos, val_scores_neg, metric='f1')
    
    print(f"Optimal threshold (F1-based): {optimal_threshold:.4f} (Val F1: {best_f1:.4f})")

    print("\nEvaluating on Test set with optimal threshold...")
    # Re-evaluate with optimal threshold
    with torch.no_grad():
        test_pos_edge_index = test_set['pos_edge_index'].to(args.device)
        test_pos_edge_type = test_set['pos_edge_type'].to(args.device)
        test_neg_edge_index = test_set['neg_edge_index'].to(args.device)
        test_neg_edge_type = test_set['neg_edge_type'].to(args.device)
        
        test_scores_pos = model.decoder.score(z_re, z_im, test_pos_edge_index, test_pos_edge_type)
        test_scores_neg = model.decoder.score(z_re, z_im, test_neg_edge_index, test_neg_edge_type)
        
        test_binary_overall = compute_binary_metrics(test_scores_pos, test_scores_neg, threshold=optimal_threshold)
        test_ranking_overall = compute_ranking_metrics(
            model,
            z_re,
            z_im,
            test_pos_edge_index,
            test_pos_edge_type,
            dataset['num_nodes'],
            args.device,
            positive_filter_maps,
        )
        test_overall = merge_metric_dicts(test_binary_overall, test_ranking_overall)
        
        # Per-type metrics
        test_per_type = {}
        combined_types = torch.unique(torch.cat([test_pos_edge_type, test_neg_edge_type], dim=0)).cpu().tolist()
        for rel in combined_types:
            rel = int(rel)
            pos_mask = (test_pos_edge_type == rel)
            neg_mask = (test_neg_edge_type == rel)
            rel_binary_metrics = compute_binary_metrics(
                test_scores_pos[pos_mask],
                test_scores_neg[neg_mask],
                threshold=optimal_threshold,
            )
            rel_ranking_metrics = compute_ranking_metrics(
                model,
                z_re,
                z_im,
                test_pos_edge_index[:, pos_mask],
                test_pos_edge_type[pos_mask],
                dataset['num_nodes'],
                args.device,
                positive_filter_maps,
            )
            rel_metrics = merge_metric_dicts(rel_binary_metrics, rel_ranking_metrics)
            test_per_type[rel] = rel_metrics
    if test_overall is not None:
        test_msg = (
            f"Test Results | AUC: {test_overall['auc']:.4f} | AP: {test_overall['ap']:.4f} | "
            f"Acc: {test_overall['acc']:.4f} | P: {test_overall['precision']:.4f} | "
            f"R: {test_overall['recall']:.4f} | F1: {test_overall['f1']:.4f}"
        )
        if 'mrr' in test_overall:
            test_msg += (
                f" | MRR: {test_overall['mrr']:.4f} | "
                f"H@1: {test_overall['hits_at_1']:.4f} | "
                f"H@5: {test_overall['hits_at_5']:.4f} | "
                f"H@10: {test_overall['hits_at_10']:.4f}"
            )
        print(test_msg)
    else:
        print("Test Results | Metrics unavailable")
    print_per_type_metrics("Test", test_per_type)

    test_csv_path = args.test_metrics_csv
    if test_csv_path is None:
        base_dir = os.path.dirname(os.path.abspath(args.dataset_path)) or '.'
        test_csv_path = os.path.join(base_dir, 'test_metrics.csv')
    save_test_metrics_csv(test_overall, test_per_type, test_csv_path)
    print(f"Test metrics CSV saved to {test_csv_path}")

    # ---------------------------------------------------------
    # Reconstruct Graph with Train/Val and Predicted Test Links
    # ---------------------------------------------------------
    print(f"\nReconstructing graph with predicted links (threshold={optimal_threshold:.4f})...")
    
    # 1. Get all Train and Val edges
    train_edges = train_set['pos_edge_index']
    train_types = train_set['pos_edge_type']
    val_edges = val_set['pos_edge_index']
    val_types = val_set['pos_edge_type']
    test_edges = test_set['pos_edge_index']
    test_types = test_set['pos_edge_type']
    
    # 2. Get Test edges that are predicted as positive (using optimal threshold)
    # Note: z_re, z_im are already computed above
    model.eval()
    with torch.no_grad():
        # Score positive test edges
        probs_test = torch.sigmoid(test_scores_pos)
        
        # Filter edges with score > optimal_threshold
        mask = (probs_test > optimal_threshold)
        predicted_test_edges = test_pos_edge_index[:, mask]
        predicted_test_types = test_pos_edge_type[mask]

        # Negative test edges predicted as positive (false positives)
        if test_neg_edge_index.numel() > 0:
            probs_test_neg = torch.sigmoid(test_scores_neg)
            neg_mask = (probs_test_neg > optimal_threshold)
            predicted_test_neg_edges = test_neg_edge_index[:, neg_mask]
            predicted_test_neg_types = test_neg_edge_type[neg_mask]
        else:
            predicted_test_neg_edges = test_neg_edge_index
            predicted_test_neg_types = test_neg_edge_type
        
    print(f"Test edges: {test_pos_edge_index.size(1)}")
    print(f"Predicted positive test edges (score > {optimal_threshold:.4f}): {predicted_test_edges.size(1)}")
    print(f"Test negative edges: {test_neg_edge_index.size(1)}")
    print(f"Predicted negative test edges (score > {optimal_threshold:.4f}): {predicted_test_neg_edges.size(1)}")
    
    # Move back to CPU for storage
    predicted_test_edges = predicted_test_edges.cpu()
    predicted_test_types = predicted_test_types.cpu()
    predicted_test_neg_edges = predicted_test_neg_edges.cpu()
    predicted_test_neg_types = predicted_test_neg_types.cpu()
    
    # 3. Combine all edges
    final_edge_index = torch.cat([train_edges, val_edges, predicted_test_edges], dim=1)
    final_edge_type = torch.cat([train_types, val_types, predicted_test_types], dim=0)
    
    print(f"Final graph edges: {final_edge_index.size(1)}")
    
    # Create new dataset dictionary to save
    trained_dataset = {
        'vocab': dataset['vocab'],
        'num_nodes': dataset['num_nodes'],
        'x': dataset['x'],
        'edge_index': final_edge_index,
        'edge_type': final_edge_type,
        'train_edges': train_edges,
        'val_edges': val_edges,
        'test_edges': test_edges,
        'test_edge_types': test_types,
        'predicted_test_edges': predicted_test_edges,
        'predicted_test_types': predicted_test_types,
        'predicted_test_neg_edges': predicted_test_neg_edges,
        'predicted_test_neg_types': predicted_test_neg_types,
        'optimal_threshold': optimal_threshold,  # Save optimal threshold for later use
    }
    
    output_path = os.path.join(os.path.dirname(args.dataset_path), 'trained_dataset.pt')
    torch.save(trained_dataset, output_path)
    print(f"Trained dataset saved to {output_path}")

if __name__ == '__main__':
    main()
