import torch
from torch_geometric.data import Data
from .data_loader import tokenize

COMPOSITION_EDGE_TYPE_ID = 0  # Displayed as Type 1 in reports.
TERM_LINK_EDGE_TYPE_ID = 1    # Displayed as Type 2 in reports.


def iter_edges_for_term(term, vocab):
    """
    Yield (src, dst, etype) edges for a single term using the RGNN rule.
    This preserves duplicates within a term; callers can deduplicate if desired.
    """
    tokens = tokenize(term)
    token_indices = [vocab[token] for token in tokens if token in vocab]
    if len(token_indices) == 1:
        idx = token_indices[0]
        yield (idx, idx, TERM_LINK_EDGE_TYPE_ID)
    elif len(token_indices) >= 2:
        for i in range(len(token_indices) - 1):
            yield (token_indices[i], token_indices[i + 1], COMPOSITION_EDGE_TYPE_ID)
        yield (token_indices[0], token_indices[-1], TERM_LINK_EDGE_TYPE_ID)

def build_rgnn_graph(terms, vocab, emb_model=None):
    """
    Build a heterogeneous graph for R-GCN.
    
    Relation type names:
    Type 1: composition link (internal id 0)
    Type 2: term link (internal id 1)
    """
    edge_index = []
    edge_type = []
    unique_edges = set()
    
    for term in terms:
        for edge in iter_edges_for_term(term, vocab):
            unique_edges.add(edge)
        
    # Convert set back to lists
    for src, dst, etype in unique_edges:
        edge_index.append([src, dst])
        edge_type.append(etype)

    # Handle empty edge list
    if not edge_index:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        
    num_nodes = len(vocab)
    
    # Node features
    if emb_model is not None:
        x_list = []
        # Ensure vocab iteration order matches indices
        # vocab is {token: index}
        # Sort by index to be safe
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        for token, idx in sorted_vocab:
            try:
                vector = emb_model.wv[token]
            except KeyError:
                vector = [0.0] * emb_model.vector_size
            x_list.append(torch.tensor(vector, dtype=torch.float))
        x = torch.stack(x_list, dim=0)
    else:
        # Identity features if no embeddings
        x = torch.eye(num_nodes, dtype=torch.float)
        
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    return data
