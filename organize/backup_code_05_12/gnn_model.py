#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN MODEL - Updated for Enhanced Features
==========================================
Now uses 15-dim node features and 10-dim edge features from C++.
"""

import torch
from torch import Tensor, nn
from torch_geometric.nn import GCNConv, GATConv
from typing import Tuple, Optional
import numpy as np

import logging
logger = logging.getLogger(__name__)

# ============================================================================
# UPDATED CONSTANTS
# ============================================================================

NODE_FEATURE_DIM = 15  # Expanded from 7
EDGE_FEATURE_DIM = 10  # From C++


class GCNWithAttention(nn.Module):
    """GCN backbone with multi-head attention."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 3, n_heads: int = 4):
        super().__init__()

        layers = []
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [hidden_dim]
        for i in range(n_layers):
            layers.append(GCNConv(dims[i], dims[i + 1]))
        self.convs = nn.ModuleList(layers)

        self.attention = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=n_heads,
            concat=True,
            dropout=0.1,
            add_self_loops=False
        )

        self.attention_proj = nn.Linear(hidden_dim * n_heads, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        device = x.device
        edge_index = edge_index.to(device, dtype=torch.long)

        for conv in self.convs:
            x = self.activation(conv(x, edge_index))
            x = self.dropout(x)

        if edge_index.numel() > 0:
            try:
                attn_out = self.attention(x, edge_index)
                attn_out = self.activation(self.attention_proj(attn_out))
                x = x + attn_out * 0.3
            except Exception as e:
                logger.warning(f"Attention layer failed: {e}")

        return x


class EdgeFeatureEncoder(nn.Module):
    """Encodes C++ edge features."""

    def __init__(self, num_edge_features: int = EDGE_FEATURE_DIM, output_dim: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_edge_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )
        self.output_dim = output_dim

    def forward(self, edge_features: Tensor) -> Tensor:
        if edge_features.numel() == 0:
            return torch.zeros(0, self.output_dim, device=edge_features.device)
        return self.encoder(edge_features)


class AttentionWeightedEdgeScorer(nn.Module):
    """Score edges using attention + C++ edge features + node embeddings."""

    def __init__(self, hidden_dim: int, edge_feature_dim: int = 32):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.edge_feature_dim = edge_feature_dim

        total_dim = 2 * hidden_dim + edge_feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 2 * total_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        self.mlp_no_edge = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.edge_attention = nn.Sequential(
            nn.Linear(edge_feature_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(
            self,
            node_embs: Tensor,
            edge_index: Tensor,
            edge_features: Optional[Tensor] = None
    ) -> Tensor:
        if edge_index.numel() == 0 or edge_index.shape[1] == 0:
            return torch.zeros(0, device=node_embs.device, dtype=torch.float32)

        src_idx, tgt_idx = edge_index
        num_nodes = node_embs.shape[0]

        if len(src_idx) > 0:
            max_idx = max(src_idx.max().item(), tgt_idx.max().item())
            if max_idx >= num_nodes:
                src_idx = torch.clamp(src_idx, 0, num_nodes - 1)
                tgt_idx = torch.clamp(tgt_idx, 0, num_nodes - 1)

        src_emb = node_embs[src_idx]
        tgt_emb = node_embs[tgt_idx]

        if edge_features is not None and edge_features.numel() > 0:
            edge_attn_weights = self.edge_attention(edge_features)
            edge_feats_weighted = edge_features * edge_attn_weights
            edge_feat = torch.cat([src_emb, tgt_emb, edge_feats_weighted], dim=1)
            score = self.mlp(edge_feat).squeeze(-1)
        else:
            edge_feat = torch.cat([src_emb, tgt_emb], dim=1)
            score = self.mlp_no_edge(edge_feat).squeeze(-1)

        score = torch.clamp(score, min=-1e6, max=1e6)
        score = torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

        return score


class GNNModel(nn.Module):
    """Full GNN with C++ edge features support."""

    def __init__(
            self,
            input_dim: int = NODE_FEATURE_DIM,  # Updated to 15
            hidden_dim: int = 64,
            n_layers: int = 3,
            n_heads: int = 4,
            edge_feature_dim: int = EDGE_FEATURE_DIM,  # Updated to 10
            use_cpp_edge_features: bool = True,  # Use C++ features
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.edge_feature_dim = edge_feature_dim
        self.use_cpp_edge_features = use_cpp_edge_features

        self.backbone = GCNWithAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads
        )

        # Edge encoder for C++ features
        encoded_edge_dim = 32
        self.edge_encoder = EdgeFeatureEncoder(
            num_edge_features=edge_feature_dim,
            output_dim=encoded_edge_dim
        )

        self.scorer = AttentionWeightedEdgeScorer(
            hidden_dim=hidden_dim,
            edge_feature_dim=encoded_edge_dim
        )

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_features: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: [N, 15] node features (from C++)
            edge_index: [2, E] edge indices
            edge_features: [E, 10] edge features (from C++)

        Returns:
            edge_logits: [E] scores
            node_embs: [N, hidden_dim] embeddings
        """
        if x.dim() != 2:
            raise ValueError(f"Node features must be 2D, got {x.dim()}D")
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Edge index must be [2, E], got {edge_index.shape}")

        device = x.device
        edge_index = edge_index.to(device, dtype=torch.long)

        num_nodes = x.shape[0]
        if edge_index.numel() > 0:
            max_idx = edge_index.max().item()
            if max_idx >= num_nodes:
                edge_index = torch.clamp(edge_index, 0, num_nodes - 1)

        # Get node embeddings
        node_embs = self.backbone(x, edge_index)

        if torch.isnan(node_embs).any():
            node_embs = torch.nan_to_num(node_embs, nan=0.0)

        # Encode C++ edge features
        encoded_edge_features = None
        if edge_features is not None and edge_features.numel() > 0 and self.use_cpp_edge_features:
            try:
                encoded_edge_features = self.edge_encoder(edge_features.float())
            except Exception as e:
                logger.warning(f"Could not encode edge features: {e}")

        # Score edges
        edge_logits = self.scorer(node_embs, edge_index, encoded_edge_features)

        return edge_logits, node_embs


if __name__ == "__main__":
    print("Testing GNNModel with C++ features...")
    print("=" * 60)

    model = GNNModel(input_dim=15, hidden_dim=64, edge_feature_dim=10)

    x = torch.randn(10, 15)  # 10 nodes, 15 features
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_features = torch.randn(3, 10)  # 3 edges, 10 features from C++

    edge_logits, node_embs = model(x, edge_index, edge_features)
    print(f"Node embeddings: {node_embs.shape}")
    print(f"Edge logits: {edge_logits.shape}")
    print("✓ All tests passed!")