import torch
import random
import numpy as np

def split_edges_stratified(edge_index, edge_type, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split edges into train/val/test sets while maintaining the ratio of relation types.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    num_types = int(edge_type.max().item()) + 1
    
    train_edges_list = []
    train_types_list = []
    val_edges_list = []
    val_types_list = []
    test_edges_list = []
    test_types_list = []
    
    rng = random.Random(seed)
    
    for r in range(num_types):
        # Get indices for this relation type
        mask = (edge_type == r)
        indices = torch.nonzero(mask, as_tuple=True)[0]
        indices = indices.tolist()
        
        rng.shuffle(indices)
        
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # n_test = rest
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        if train_idx:
            train_edges_list.append(edge_index[:, train_idx])
            train_types_list.append(edge_type[train_idx])
        if val_idx:
            val_edges_list.append(edge_index[:, val_idx])
            val_types_list.append(edge_type[val_idx])
        if test_idx:
            test_edges_list.append(edge_index[:, test_idx])
            test_types_list.append(edge_type[test_idx])

    def concat_data(e_list, t_list):
        if not e_list:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
        return torch.cat(e_list, dim=1), torch.cat(t_list)

    train_edge_index, train_edge_type = concat_data(train_edges_list, train_types_list)
    val_edge_index, val_edge_type = concat_data(val_edges_list, val_types_list)
    test_edge_index, test_edge_type = concat_data(test_edges_list, test_types_list)
    
    return (train_edge_index, train_edge_type), (val_edge_index, val_edge_type), (test_edge_index, test_edge_type)


def sample_negative_triplets(
    edge_index,
    edge_type,
    num_nodes,
    seed=42,
    extra_positive_edges=None,
    return_stats=False,
    neg_ratio=1,
    hard_negative_mode='none',
    corrupt_mode='tail'
):
    """
    Generate negative samples by corrupting head and/or tail nodes.
    Existing positives from all provided splits are rejected.
    """
    if neg_ratio < 1:
        raise ValueError("neg_ratio must be >= 1")
    if hard_negative_mode not in {'none', 'degree', 'mixed'}:
        raise ValueError("hard_negative_mode must be one of: none, degree, mixed")
    if corrupt_mode not in {'tail', 'head', 'both'}:
        raise ValueError("corrupt_mode must be one of: tail, head, both")

    rng = np.random.default_rng(seed)
    
    # Existing positives are used to avoid false negatives.
    existing_triplets = set()
    degree = np.ones(num_nodes, dtype=np.float64)

    def add_triplets(e_index, e_type):
        for i in range(e_index.size(1)):
            s = e_index[0, i].item()
            o = e_index[1, i].item()
            r = e_type[i].item()
            existing_triplets.add((s, r, o))
            degree[s] += 1.0
            degree[o] += 1.0

    add_triplets(edge_index, edge_type)
    if extra_positive_edges:
        for e_index_extra, e_type_extra in extra_positive_edges:
            if e_index_extra.numel() == 0:
                continue
            add_triplets(e_index_extra, e_type_extra)

    degree_prob = degree / degree.sum()

    def choose_corrupt_side():
        if corrupt_mode == 'both':
            return 'head' if rng.random() < 0.5 else 'tail'
        return corrupt_mode

    def sample_node():
        use_degree = hard_negative_mode == 'degree' or (
            hard_negative_mode == 'mixed' and rng.random() < 0.5
        )
        if use_degree:
            return int(rng.choice(num_nodes, p=degree_prob))
        return int(rng.integers(0, num_nodes))

    def build_candidate(s, r, o, side, node):
        if side == 'head':
            return node, r, o
        return s, r, node

    def is_valid_negative(s, r, o, side, node):
        if side == 'head' and node == s:
            return False
        if side == 'tail' and node == o:
            return False
        return build_candidate(s, r, o, side, node) not in existing_triplets

    neg_edge_list = []
    neg_type_list = []
    
    fallback_count = 0
    rejected_positive_count = 0
    head_corrupt_count = 0
    tail_corrupt_count = 0

    for i in range(edge_index.size(1)):
        s = edge_index[0, i].item()
        o = edge_index[1, i].item()
        r = edge_type[i].item()
        
        for _ in range(neg_ratio):
            side = choose_corrupt_side()
            found_negative = False
            candidate = None

            for _ in range(100):
                node = sample_node()
                if is_valid_negative(s, r, o, side, node):
                    candidate = build_candidate(s, r, o, side, node)
                    found_negative = True
                    break
                rejected_positive_count += 1

            if not found_negative:
                fallback_count += 1
                for node in rng.permutation(num_nodes):
                    node = int(node)
                    if is_valid_negative(s, r, o, side, node):
                        candidate = build_candidate(s, r, o, side, node)
                        found_negative = True
                        break

            if candidate is None:
                # Last resort for extremely dense graphs.
                node = sample_node()
                candidate = build_candidate(s, r, o, side, node)

            neg_s, neg_r, neg_o = candidate
            neg_edge_list.append([neg_s, neg_o])
            neg_type_list.append(neg_r)
            if side == 'head':
                head_corrupt_count += 1
            else:
                tail_corrupt_count += 1

    if not neg_edge_list:
        empty_edges = torch.empty((2, 0), dtype=torch.long)
        empty_types = torch.empty((0,), dtype=torch.long)
        if return_stats:
            return empty_edges, empty_types, {
                "fallback_count": 0,
                "total": 0,
                "head_corrupt_count": 0,
                "tail_corrupt_count": 0,
                "rejected_positive_count": 0,
            }
        return empty_edges, empty_types
        
    neg_edge_index = torch.tensor(neg_edge_list, dtype=torch.long).t().contiguous()
    neg_edge_type = torch.tensor(neg_type_list, dtype=torch.long)
    
    if return_stats:
        return neg_edge_index, neg_edge_type, {
            "fallback_count": fallback_count,
            "total": int(neg_edge_index.size(1)),
            "head_corrupt_count": head_corrupt_count,
            "tail_corrupt_count": tail_corrupt_count,
            "rejected_positive_count": rejected_positive_count,
        }
    return neg_edge_index, neg_edge_type
