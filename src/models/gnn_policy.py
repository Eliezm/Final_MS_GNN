#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN POLICY - Updated for Enhanced Features
===========================================
Now handles 15-dim node features and 10-dim edge features from C++.
"""

import numpy as np
import torch
from torch import nn, Tensor
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Tuple, Dict, Any, Optional

# from gnn_model import GNNModel, NODE_FEATURE_DIM, EDGE_FEATURE_DIM
from src.models.gnn_model import GNNModel, NODE_FEATURE_DIM, EDGE_FEATURE_DIM

import logging

logger = logging.getLogger(__name__)


class GNNExtractor(nn.Module):
    """Feature extractor using enhanced GNN."""

    def __init__(
            self,
            input_dim: int = NODE_FEATURE_DIM,  # 15
            hidden_dim: int = 64,
            n_layers: int = 3,
            n_heads: int = 4,
            edge_feature_dim: int = EDGE_FEATURE_DIM,  # 10
    ):
        # ✅ FIX 2: Force CPU device
        import torch
        torch.cuda.is_available = lambda: False  # Pretend CUDA doesn't exist

        super().__init__()

        self.gnn = GNNModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            edge_feature_dim=edge_feature_dim,
            use_cpp_edge_features=True,
        )
        self.hidden_dim = hidden_dim

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_features: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        return self.gnn(x, edge_index, edge_features)


class GNNPolicy(ActorCriticPolicy):
    """Actor-Critic policy using enhanced GNN."""

    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            net_arch=None,
            activation_fn=nn.ReLU,
            hidden_dim: int = 64,
            n_layers: int = 3,
            n_heads: int = 4,
            **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            activation_fn=activation_fn,
            **kwargs
        )

        self.node_feat_dim = NODE_FEATURE_DIM  # 15
        self.edge_feat_dim = EDGE_FEATURE_DIM  # 10
        self.hidden_dim = hidden_dim

        self.extractor = GNNExtractor(
            input_dim=self.node_feat_dim,
            hidden_dim=self.hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            edge_feature_dim=self.edge_feat_dim,
        )

        self.value_net = nn.Linear(self.hidden_dim, 1)
        self.action_net = nn.Identity()

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1)
        )

    def _get_data_from_obs(self, obs: Dict[str, Any]) -> Tuple[Tensor, Tensor, Optional[Tensor], int, int]:
        """Extract data including edge features."""
        device = self.device

        x = torch.as_tensor(obs["x"], dtype=torch.float32, device=device)
        edge_index = torch.as_tensor(obs["edge_index"], dtype=torch.long, device=device)

        # Get edge features from C++
        edge_features = None
        if "edge_features" in obs:
            edge_features = torch.as_tensor(obs["edge_features"], dtype=torch.float32, device=device)

        def to_python_int(val):
            if isinstance(val, torch.Tensor):
                return int(val.cpu().item())
            elif isinstance(val, np.ndarray):
                return int(val.flat[0])
            elif hasattr(val, 'item'):
                return int(val.item())
            return int(val)

        num_nodes = to_python_int(obs.get("num_nodes", x.shape[-2]))
        num_edges = to_python_int(obs.get("num_edges", edge_index.shape[-1]))

        if x.dim() == 3:
            x = x[0]
            edge_index = edge_index[0] if edge_index.dim() == 3 else edge_index
            if edge_features is not None and edge_features.dim() == 3:
                edge_features = edge_features[0]

        num_nodes = max(1, min(num_nodes, x.shape[0]))
        num_edges = max(0, min(num_edges, edge_index.shape[1] if edge_index.dim() == 2 else 0))

        x = x[:num_nodes]
        edge_index = edge_index[:, :num_edges] if num_edges > 0 else edge_index[:, :0]

        if edge_features is not None:
            edge_features = edge_features[:num_edges]

        if num_edges > 0 and num_nodes > 0:
            max_valid_idx = num_nodes - 1
            edge_index = torch.clamp(edge_index, min=0, max=max_valid_idx)

        return x, edge_index, edge_features, num_nodes, num_edges

    def _mask_invalid_edges(self, logits: Tensor, num_edges: int) -> Tensor:
        E = logits.size(0)
        num_edges = max(1, min(num_edges, E))
        mask = torch.arange(E, device=logits.device) < num_edges
        if not mask.any():
            mask[0] = True
        masked = logits.clone()
        masked[~mask] = float('-inf')
        return masked

    def _sample_action(self, logits: Tensor, deterministic: bool) -> Tuple[Tensor, Tensor]:
        logits = torch.clamp(logits, min=-100, max=100)
        if torch.isinf(logits).all():
            logits = torch.zeros_like(logits)
        probs = torch.softmax(logits, dim=0)
        if torch.isnan(probs).any():
            probs = torch.ones_like(logits) / len(logits)
        dist = Categorical(probs=probs)
        if deterministic:
            action = probs.argmax()
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    @torch.no_grad()
    def predict(self, observation: Dict[str, Any], state=None, episode_start=None, deterministic: bool = False):
        """
        ✅ FIXED: Ensure Python int action return
        """
        self.eval()
        try:
            x, edge_index, edge_features, num_nodes, num_edges = self._get_data_from_obs(observation)

            if num_edges == 0:
                # Return (numpy array with Python int, None)
                action_value = int(0)  # Ensure Python int
                return np.array([action_value], dtype=np.int64), None

            edge_logits, _ = self.extractor(x, edge_index, edge_features)
            masked_logits = self._mask_invalid_edges(edge_logits, num_edges)
            action_tensor, _ = self._sample_action(masked_logits, deterministic)

            # ✅ CRITICAL: Extract Python int from tensor
            action_python_int = int(action_tensor.cpu().item())

            # Validate
            if not isinstance(action_python_int, int) or isinstance(action_python_int, bool):
                raise TypeError(f"Action must be Python int, got {type(action_python_int)}")

            return np.array([action_python_int], dtype=np.int64), None

        except Exception as e:
            logger.error(f"Predict failed: {e}")
            return np.array([0], dtype=np.int64), None

    def forward(self, obs: Dict[str, Any], deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        device = self.device
        x, edge_index, edge_features, num_nodes, num_edges = self._get_data_from_obs(obs)

        edge_logits, node_embs = self.extractor(x, edge_index, edge_features)

        if node_embs.shape[0] > 0:
            value = self.value_net(node_embs.mean(dim=0, keepdim=True))
        else:
            value = torch.zeros(1, 1, device=device)

        if num_edges == 0:
            action = torch.zeros(1, dtype=torch.long, device=device)
            log_prob = torch.zeros(1, device=device)
        else:
            masked_logits = self._mask_invalid_edges(edge_logits, num_edges)
            action, log_prob = self._sample_action(masked_logits, deterministic)
            action = action.unsqueeze(0)
            log_prob = log_prob.unsqueeze(0)

        return action, value, log_prob

    def evaluate_actions(self, obs: Dict[str, Tensor], actions: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        device = self.device

        def to_python_int(val):
            if isinstance(val, torch.Tensor):
                return int(val.cpu().item())
            elif isinstance(val, np.ndarray):
                return int(val.flat[0])
            elif hasattr(val, 'item'):
                return int(val.item())
            return int(val)

        x = obs["x"].to(device, dtype=torch.float32)
        edge_index = obs["edge_index"].to(device, dtype=torch.long)

        edge_features = None
        if "edge_features" in obs:
            edge_features = obs["edge_features"].to(device, dtype=torch.float32)

        if x.dim() == 3:
            batch_size = x.shape[0]
        else:
            batch_size = 1
            x = x.unsqueeze(0)
            edge_index = edge_index.unsqueeze(0)
            if edge_features is not None:
                edge_features = edge_features.unsqueeze(0)

        num_nodes_raw = obs.get("num_nodes", None)
        if num_nodes_raw is None:
            num_nodes_arr = [x.shape[1]] * batch_size
        elif isinstance(num_nodes_raw, torch.Tensor):
            num_nodes_arr = num_nodes_raw.cpu().numpy().flatten().tolist()
        elif isinstance(num_nodes_raw, np.ndarray):
            num_nodes_arr = num_nodes_raw.flatten().tolist()
        else:
            num_nodes_arr = [int(num_nodes_raw)] * batch_size

        num_edges_raw = obs.get("num_edges", None)
        if num_edges_raw is None:
            num_edges_arr = [edge_index.shape[-1]] * batch_size
        elif isinstance(num_edges_raw, torch.Tensor):
            num_edges_arr = num_edges_raw.cpu().numpy().flatten().tolist()
        elif isinstance(num_edges_raw, np.ndarray):
            num_edges_arr = num_edges_raw.flatten().tolist()
        else:
            num_edges_arr = [int(num_edges_raw)] * batch_size

        values, log_probs, entropies = [], [], []

        for i in range(batch_size):
            x_i = x[i] if x.dim() == 3 else x
            ei_i = edge_index[i] if edge_index.dim() == 3 else edge_index
            ef_i = edge_features[i] if edge_features is not None and edge_features.dim() == 3 else edge_features
            action_i = actions[i] if actions.dim() > 0 else actions

            num_nodes_i = int(num_nodes_arr[i]) if i < len(num_nodes_arr) else x_i.shape[0]
            num_edges_i = int(num_edges_arr[i]) if i < len(num_edges_arr) else ei_i.shape[-1]

            num_nodes_i = max(1, min(num_nodes_i, x_i.shape[0]))
            num_edges_i = max(0, min(num_edges_i, ei_i.shape[-1]))

            x_i = x_i[:num_nodes_i]
            ei_i = ei_i[:, :num_edges_i]
            if ef_i is not None:
                ef_i = ef_i[:num_edges_i]

            if num_edges_i > 0 and num_nodes_i > 0:
                max_idx = num_nodes_i - 1
                ei_i = torch.clamp(ei_i, min=0, max=max_idx)

            edge_logits, node_embs = self.extractor(x_i, ei_i, ef_i)

            if node_embs.shape[0] > 0:
                val_i = self.value_net(node_embs.mean(dim=0, keepdim=True))
            else:
                val_i = torch.zeros(1, device=device)

            if num_edges_i == 0 or edge_logits.shape[0] == 0:
                log_prob_i = torch.zeros(1, device=device, requires_grad=True)
                ent_i = torch.zeros(1, device=device, requires_grad=True)
            else:
                masked_logits = self._mask_invalid_edges(edge_logits, num_edges_i)
                if torch.isinf(masked_logits).all():
                    masked_logits = torch.zeros_like(edge_logits)
                dist = Categorical(logits=masked_logits)
                action_clamped = torch.clamp(action_i, 0, edge_logits.shape[0] - 1)
                log_prob_i = dist.log_prob(action_clamped)
                ent_i = dist.entropy()

            values.append(val_i.squeeze())
            log_probs.append(log_prob_i)
            entropies.append(ent_i)

        return torch.stack(values), torch.stack(log_probs), torch.stack(entropies)

    def predict_values(self, obs: Dict[str, Tensor]) -> Tensor:
        device = self.device
        x = obs["x"].to(device, dtype=torch.float32)
        edge_index = obs["edge_index"].to(device, dtype=torch.long)
        edge_features = obs.get("edge_features")
        if edge_features is not None:
            edge_features = edge_features.to(device, dtype=torch.float32)

        if x.dim() == 2:
            x = x.unsqueeze(0)
            edge_index = edge_index.unsqueeze(0)
            if edge_features is not None:
                edge_features = edge_features.unsqueeze(0)

        batch_size = x.shape[0]
        values = []

        for i in range(batch_size):
            x_i = x[i]
            ei_i = edge_index[i]
            ef_i = edge_features[i] if edge_features is not None else None

            _, node_embs = self.extractor(x_i, ei_i, ef_i)

            if node_embs.shape[0] > 0:
                val = self.value_net(node_embs.mean(dim=0, keepdim=True))
            else:
                val = torch.zeros(1, 1, device=device)

            values.append(val)

        return torch.cat(values, dim=0)