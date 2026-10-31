"""Compact hierarchical DeepSets/GNN models for patient response."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax


class GatedAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        logits = self.score(torch.tanh(self.value(x)) * torch.sigmoid(self.gate(x))).squeeze(-1)
        weights = softmax(logits, group)
        return global_add_pool(weights.unsqueeze(-1) * x, group)


class ResidualGINEBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv = GINEConv(message_mlp, edge_dim=edge_dim, train_eps=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return x
        update = self.conv(self.norm(x), edge_index, edge_attr)
        return x + self.dropout(F.gelu(update))


class HierarchicalPatientModel(nn.Module):
    """Cell encoder -> optional spatial GNN -> ROI MIL -> patient MIL."""

    def __init__(
        self,
        expression_dim: int,
        num_cell_types: int,
        edge_dim: int,
        feature_mode: str = "combined",
        hidden_dim: int = 64,
        cell_type_dim: int = 24,
        num_gnn_layers: int = 2,
        dropout: float = 0.3,
        use_graph: bool = True,
    ):
        super().__init__()
        if feature_mode not in {"expressions", "celltypes", "combined"}:
            raise ValueError(f"Unknown feature mode: {feature_mode}")
        self.feature_mode = feature_mode
        self.use_graph = bool(use_graph)
        self.expression_encoder = None
        self.cell_type_embedding = None

        if feature_mode in {"expressions", "combined"}:
            self.expression_encoder = nn.Sequential(
                nn.Linear(expression_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        if feature_mode in {"celltypes", "combined"}:
            self.cell_type_embedding = nn.Embedding(num_cell_types, cell_type_dim)
            self.cell_type_encoder = nn.Sequential(
                nn.Linear(cell_type_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
            )
        combined_dim = hidden_dim * (2 if feature_mode == "combined" else 1)
        self.node_projection = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )

        self.gnn = nn.ModuleList(
            [ResidualGINEBlock(hidden_dim, edge_dim, dropout) for _ in range(num_gnn_layers)]
            if self.use_graph
            else []
        )
        self.roi_attention = GatedAttentionPool(hidden_dim)
        self.roi_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.patient_attention = GatedAttentionPool(hidden_dim)
        self.patient_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def encode_nodes(self, expression, cell_type):
        parts = []
        if self.expression_encoder is not None:
            parts.append(self.expression_encoder(expression))
        if self.cell_type_embedding is not None:
            parts.append(self.cell_type_encoder(self.cell_type_embedding(cell_type)))
        return self.node_projection(torch.cat(parts, dim=-1))

    def forward(self, graph_batch, roi_to_patient):
        x = self.encode_nodes(graph_batch.x, graph_batch.cell_type)
        for layer in self.gnn:
            x = layer(x, graph_batch.edge_index, graph_batch.edge_attr)

        roi_mean = global_mean_pool(x, graph_batch.batch)
        roi_max = global_max_pool(x, graph_batch.batch)
        roi_attention = self.roi_attention(x, graph_batch.batch)
        roi_embedding = self.roi_projection(torch.cat([roi_mean, roi_max, roi_attention], dim=-1))

        patient_mean = global_mean_pool(roi_embedding, roi_to_patient)
        patient_max = global_max_pool(roi_embedding, roi_to_patient)
        patient_attention = self.patient_attention(roi_embedding, roi_to_patient)
        patient_embedding = self.patient_projection(
            torch.cat([patient_mean, patient_max, patient_attention], dim=-1)
        )
        return self.classifier(patient_embedding).squeeze(-1)
