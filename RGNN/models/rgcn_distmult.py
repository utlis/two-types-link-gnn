import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, RGATConv

from models.compgcn_encoder import CompGCNEncoder

class RGCNEncoder(nn.Module):
    def __init__(self, num_features, hidden_dim, embedding_dim, num_relations, num_layers=2, dropout=0.2, num_bases=None):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        
        # First layer
        self.convs.append(RGCNConv(num_features, hidden_dim, num_relations, num_bases=num_bases))
        
        # Hidden layers (if any)
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases))
            
        # Output layer
        self.convs.append(RGCNConv(hidden_dim, embedding_dim, num_relations, num_bases=num_bases))

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index, edge_type)
        return x


class RGATEncoder(nn.Module):
    def __init__(self, num_features, hidden_dim, embedding_dim, num_relations, num_layers=2, dropout=0.2, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.heads = heads

        # First layer
        self.convs.append(
            RGATConv(
                num_features,
                hidden_dim,
                num_relations,
                heads=heads,
                concat=False,
                dropout=dropout,
            )
        )

        # Hidden layers (if any)
        for _ in range(num_layers - 2):
            self.convs.append(
                RGATConv(
                    hidden_dim,
                    hidden_dim,
                    num_relations,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                )
            )

        # Output layer
        self.convs.append(
            RGATConv(
                hidden_dim,
                embedding_dim,
                num_relations,
                heads=heads,
                concat=False,
                dropout=dropout,
            )
        )

    def forward(self, x, edge_index, edge_type):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_type)
        return x


class DistMultDecoder(nn.Module):
    def __init__(self, num_relations, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.rel_emb = nn.Parameter(torch.Tensor(num_relations, embedding_dim))
        nn.init.xavier_uniform_(self.rel_emb)

    def project(self, z):
        """
        Project node embeddings. For DistMult, this is identity (no projection needed).
        Returns (z_re, z_im) where z_im is None for DistMult.
        """
        return z, None

    def score(self, z_re, z_im, edge_index, edge_type):
        """
        Compute DistMult scores.
        z_im is ignored for DistMult.
        """
        s = z_re[edge_index[0]]
        o = z_re[edge_index[1]]
        r = self.rel_emb[edge_type]
        
        # DistMult score: <s, r, o> = sum(s * r * o)
        return torch.sum(s * r * o, dim=1)

    def forward(self, z, edge_index, edge_type):
        """
        z: Node embeddings [num_nodes, embedding_dim]
        edge_index: [2, num_edges]
        edge_type: [num_edges]
        """
        z_re, z_im = self.project(z)
        return self.score(z_re, z_im, edge_index, edge_type)


class ComplExDecoder(nn.Module):
    """
    ComplEx decoder for link prediction.
    Uses complex embeddings to handle asymmetric relations.
    
    Reference: Trouillon et al., "Complex Embeddings for Simple Link Prediction", ICML 2016
    """
    def __init__(self, num_relations, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Linear projections to create real and imaginary parts from encoder output
        self.proj_re = nn.Linear(embedding_dim, embedding_dim)
        self.proj_im = nn.Linear(embedding_dim, embedding_dim)
        
        # Real and imaginary parts of relation embeddings
        self.rel_emb_re = nn.Parameter(torch.Tensor(num_relations, embedding_dim))
        self.rel_emb_im = nn.Parameter(torch.Tensor(num_relations, embedding_dim))
        nn.init.xavier_uniform_(self.rel_emb_re)
        nn.init.xavier_uniform_(self.rel_emb_im)

    def project(self, z):
        """
        Project node embeddings into complex space (real and imaginary parts).
        Returns (z_re, z_im).
        """
        z_re = self.proj_re(z)
        z_im = self.proj_im(z)
        return z_re, z_im

    def score(self, z_re, z_im, edge_index, edge_type):
        """
        Compute ComplEx scores: Re(<s, r, conj(o)>)
        """
        # Get source and object embeddings
        s_re = z_re[edge_index[0]]
        s_im = z_im[edge_index[0]]
        o_re = z_re[edge_index[1]]
        o_im = z_im[edge_index[1]]
        
        # Get relation embeddings
        r_re = self.rel_emb_re[edge_type]
        r_im = self.rel_emb_im[edge_type]
        
        # ComplEx score: Re(<s, r, conj(o)>)
        # = s_re * r_re * o_re + s_re * r_im * o_im + s_im * r_re * o_im - s_im * r_im * o_re
        score = (
            torch.sum(s_re * r_re * o_re, dim=1) +
            torch.sum(s_re * r_im * o_im, dim=1) +
            torch.sum(s_im * r_re * o_im, dim=1) -
            torch.sum(s_im * r_im * o_re, dim=1)
        )
        return score

    def forward(self, z, edge_index, edge_type):
        """
        z: Node embeddings [num_nodes, embedding_dim]
        edge_index: [2, num_edges]
        edge_type: [num_edges]
        """
        z_re, z_im = self.project(z)
        return self.score(z_re, z_im, edge_index, edge_type)


class RGCNLinkPredictor(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_dim,
        embedding_dim,
        num_relations,
        num_layers=2,
        dropout=0.2,
        encoder_type="rgcn",
        decoder_type="complex",
        rgat_heads=4,
        num_bases=None,
        compgcn_composition="mult",
    ):
        super().__init__()
        
        # Encoder setup
        encoder_type = encoder_type.lower()
        if encoder_type == "rgcn":
            self.encoder = RGCNEncoder(num_features, hidden_dim, embedding_dim, num_relations, num_layers, dropout, num_bases=num_bases)
        elif encoder_type == "rgat":
            self.encoder = RGATEncoder(
                num_features,
                hidden_dim,
                embedding_dim,
                num_relations,
                num_layers=num_layers,
                dropout=dropout,
                heads=rgat_heads,
            )
        elif encoder_type == "compgcn":
            self.encoder = CompGCNEncoder(
                num_features,
                hidden_dim,
                embedding_dim,
                num_relations,
                num_layers=num_layers,
                dropout=dropout,
                composition=compgcn_composition,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Use 'rgcn', 'rgat', or 'compgcn'.")
        
        # Decoder setup
        decoder_type = decoder_type.lower()
        if decoder_type == "distmult":
            self.decoder = DistMultDecoder(num_relations, embedding_dim)
        elif decoder_type == "complex":
            self.decoder = ComplExDecoder(num_relations, embedding_dim)
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}. Use 'distmult' or 'complex'.")

    def forward(self, x, edge_index, edge_type, target_edge_index, target_edge_type):
        """
        x, edge_index, edge_type: Used for encoding (message passing)
        target_edge_index, target_edge_type: Edges to score
        """
        z = self.encoder(x, edge_index, edge_type)
        scores = self.decoder(z, target_edge_index, target_edge_type)
        return scores, z
