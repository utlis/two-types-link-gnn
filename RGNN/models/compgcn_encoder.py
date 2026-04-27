import torch
import torch.nn as nn
import torch.nn.functional as F


class CompGCNLayer(nn.Module):
    """
    Lightweight CompGCN layer for relation-aware message passing.

    The layer composes source node features with relation embeddings before
    aggregation, then updates relation embeddings for the next layer.
    """

    def __init__(self, in_dim, out_dim, composition="mult"):
        super().__init__()
        if composition not in {"mult", "sub"}:
            raise ValueError("composition must be one of: mult, sub")

        self.composition = composition
        self.message_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.rel_linear = nn.Linear(in_dim, out_dim, bias=False)

    def compose(self, node_features, relation_features):
        if self.composition == "sub":
            return node_features - relation_features
        return node_features * relation_features

    def forward(self, x, relation_embeddings, edge_index, edge_type):
        num_nodes = x.size(0)
        out = self.self_linear(x)

        if edge_index.numel() > 0:
            src = edge_index[0]
            dst = edge_index[1]
            composed = self.compose(x[src], relation_embeddings[edge_type])
            messages = self.message_linear(composed)

            aggregated = torch.zeros(num_nodes, messages.size(1), device=x.device, dtype=messages.dtype)
            aggregated.index_add_(0, dst, messages)

            degree = torch.zeros(num_nodes, device=x.device, dtype=messages.dtype)
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=messages.dtype))
            aggregated = aggregated / degree.clamp(min=1.0).unsqueeze(1)
            out = out + aggregated

        relation_embeddings = self.rel_linear(relation_embeddings)
        return out, relation_embeddings


class CompGCNEncoder(nn.Module):
    """
    CompGCN encoder with relation embeddings updated across layers.

    This keeps the same forward signature as the existing encoders:
    forward(x, edge_index, edge_type) -> node embeddings.
    """

    def __init__(
        self,
        num_features,
        hidden_dim,
        embedding_dim,
        num_relations,
        num_layers=2,
        dropout=0.2,
        composition="mult",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        dims = [num_features]
        if num_layers > 1:
            dims.extend([hidden_dim] * (num_layers - 1))
        dims.append(embedding_dim)

        self.layers = nn.ModuleList(
            CompGCNLayer(dims[i], dims[i + 1], composition=composition)
            for i in range(len(dims) - 1)
        )
        self.dropout = dropout
        self.relation_embeddings = nn.Parameter(torch.empty(num_relations, num_features))
        nn.init.xavier_uniform_(self.relation_embeddings)

    def forward(self, x, edge_index, edge_type):
        relation_embeddings = self.relation_embeddings
        for layer in self.layers[:-1]:
            x, relation_embeddings = layer(x, relation_embeddings, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x, _ = self.layers[-1](x, relation_embeddings, edge_index, edge_type)
        return x
