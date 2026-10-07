import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, EdgeConv, SAGEConv, GATConv, DynamicEdgeConv

from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import TopKPooling, SAGPooling, EdgePooling
from torch_geometric.nn import MLP
from torch_geometric.nn import GraphNorm, BatchNorm, LayerNorm

class ModelProbs(nn.Module):
	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, *args, **kwargs):
		"""
		Forward pass through the model.

		Args:
			args: Positional arguments.
			kwargs: Keyword arguments.

		Returns:
			tuple: Model output and hidden representation.
		"""
		return torch.sigmoid(self.model(*args, **kwargs))

class GCN(nn.Module):
	"""
	Graph Convolutional Network (GCN) model for node classification.

	Args:
		input_dim (int): Dimensionality of the input node features.
		hidden_dim (int): Dimensionality of the hidden layer.
		nm_class (int): Number of classes for node classification.
	"""
	def __init__(self, 
					input_dim, 
					hidden_dim, 
					drop_p=0.3):
		super(GCN, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
		self.norm1 = LayerNorm(self.hidden_dim)
		self.clf = nn.Sequential(
						nn.Linear(self.hidden_dim, 1)
					)
		self.dropout1 = nn.Dropout(0.5)
# 		self.dropout2 = nn.Dropout(0.5)

	def hidden_representation(self, x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index, edge_weight)
		x = self.dropout1(F.relu(self.norm1(x)))
		
		x = global_mean_pool(x, batch)
		return self.clf(x), x

	def forward(self, x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index, edge_weight)
		x = self.dropout1(F.relu(self.norm1(x)))
		
		x = global_mean_pool(x, batch)
		
		return self.clf(x)

	def induced_subgraph(self,  x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index, edge_weight)
		x = self.dropout(F.relu(x))
		x = self.conv2(x, edge_index, edge_weight)
		x_pooled, perm = global_sort_pool(x, batch, k=50, return_perm=True)
		mask_0 = torch.eq(edge_index[0].unsqueeze(1), topk_node_indices).any(dim=1)  # Check for source node
		mask_1 = torch.eq(edge_index[1].unsqueeze(1), topk_node_indices).any(dim=1)  # Check for target node

		# Create a mask for edges where both nodes are in the top k
		mask = mask_0 & mask_1

		# Select only the edges that connect top k nodes
		edge_index_subgraph = edge_index[:, mask]
		return edge_index_subgraph


class EdgeWeightedGCN(nn.Module):
	def __init__(self, input_dim, hidden_dim):
		super().__init__()
		self.conv1 = GCNConv(input_dim, hidden_dim)
		self.conv2 = GCNConv(hidden_dim, hidden_dim)

		self.edge_mlp = nn.Sequential(
			nn.Linear(1, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1),
			nn.Sigmoid()
		)

		self.classifier = nn.Sequential(
			nn.Linear(hidden_dim * 2, hidden_dim),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(hidden_dim, 1)
		)
		self.dropout = nn.Dropout(0.2)

	def hidden_representation(self, x, edge_index, edge_weight, batch):
		# Edge weights from edge features
		edge_weights = self.edge_mlp(edge_weight.unsqueeze(-1))

		# Graph convolutions with edge weights
		x = self.conv1(x, edge_index, edge_weight=edge_weights)
		x = F.relu(x)
		x = self.dropout(x)

		x = self.conv2(x, edge_index, edge_weight=edge_weights)

		# Multiple pooling strategies
		x_mean = global_mean_pool(x, batch)
		x_max = global_max_pool(x, batch)

		# Concatenate different pooling results
		x = torch.cat([x_mean, x_max], dim=1)

		return self.classifier(x), x

	def forward(self, x, edge_index, edge_weight, batch):
		# Edge weights from edge features
		edge_weights = self.edge_mlp(edge_weight)

		# Graph convolutions with edge weights
		x = self.conv1(x, edge_index, edge_weight=edge_weights)
		x = F.relu(x)
		x = self.dropout(x)

		x = self.conv2(x, edge_index, edge_weight=edge_weights)

		# Multiple pooling strategies
		x_mean = global_mean_pool(x, batch)
		x_max = global_max_pool(x, batch)

		# Concatenate different pooling results
		x = torch.cat([x_mean, x_max], dim=1)

		# Classification
		x = self.classifier(x)
		return x

class HierarchicalGCN(nn.Module):
	def __init__(self, input_dim, hidden_dim, k=8):
		super().__init__()
		self.k = k

		# Initial convolutions
		self.conv1 = GCNConv(input_dim, hidden_dim)

		# Dynamic edge convolution for hierarchical structure
		self.dynamic_edge_conv = DynamicEdgeConv(
			nn.Sequential(
				nn.Linear(2 * hidden_dim, hidden_dim),
				nn.ReLU(),
				nn.Linear(hidden_dim, hidden_dim)
			),
			k=self.k,
			aggr='max'
		)

		self.classifier = nn.Sequential(
			nn.Linear(hidden_dim * 2, hidden_dim),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(hidden_dim, 1)
		)

	def hidden_representation(self, x, edge_index, edge_weight, batch):
		# Initial feature processing
		x = self.conv1(x, edge_index)
		x = F.relu(x)

		# Hierarchical structure learning
		x = self.dynamic_edge_conv(x, batch)

		# Multiple pooling
		x_mean = global_mean_pool(x, batch)
		x_max = global_max_pool(x, batch)

		# Combine pooled features
		x = torch.cat([x_mean, x_max], dim=1)

		return self.classifier(x), x

	def forward(self, x, edge_index, edge_weight, batch):
		# Initial feature processing
		x = self.conv1(x, edge_index)
		x = F.relu(x)

		# Hierarchical structure learning
		x = self.dynamic_edge_conv(x, batch)

		# Multiple pooling
		x_mean = global_mean_pool(x, batch)
		x_max = global_max_pool(x, batch)

		# Combine pooled features
		x = torch.cat([x_mean, x_max], dim=1)

		# Classification
		x = self.classifier(x)
		return x

class AttentionGCN(nn.Module):
	def __init__(self, input_dim, hidden_dim, heads=4):
		super().__init__()
		self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
		self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)

		self.classifier = nn.Sequential(
			nn.Linear(hidden_dim * 3, hidden_dim),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(hidden_dim, 1)
		)
		self.dropout = nn.Dropout(0.2)

	def hidden_representation(self, x, edge_index, edge_weight, batch):
		# Multi-head attention
		x = self.conv1(x, edge_index)
		x = F.elu(x)
		x = self.dropout(x)

		x = self.conv2(x, edge_index)

		# Multiple pooling strategies
		x_mean = global_mean_pool(x, batch)
		x_max = global_max_pool(x, batch)
		x_sum = global_add_pool(x, batch)

		# Combine different pooling results
		x = torch.cat([x_mean, x_max, x_sum], dim=1)

		return self.classifier(x), x

	def forward(self, x, edge_index, batch):
		# Multi-head attention
		x = self.conv1(x, edge_index)
		x = F.elu(x)
		x = self.dropout(x)

		x = self.conv2(x, edge_index)

		# Multiple pooling strategies
		x_mean = global_mean_pool(x, batch)
		x_max = global_max_pool(x, batch)
		x_sum = global_add_pool(x, batch)

		# Combine different pooling results
		x = torch.cat([x_mean, x_max, x_sum], dim=1)

		# Classification
		x = self.classifier(x)
		return x


class SSGCN(nn.Module):
	"""
	Simple Spectral Graph Convolution (SSGCN) model for node classification.

	Args:
		input_dim (int): Dimensionality of the input node features.
		hidden_dim (int): Dimensionality of the hidden layer.
		nm_class (int): Number of classes for node classification.
	"""
	def __init__(self, 
					input_dim, 
					hidden_dim,
					K,
					alpha, 
					drop_p=0.3):
		super(SSGCN, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.K = int(K)
		self.alpha = alpha
		self.conv1 = SSGConv(self.input_dim, self.hidden_dim, self.alpha, self.K)
		self.conv2 = SSGConv(self.hidden_dim, self.hidden_dim, self.alpha, self.K)
		self.clf = nn.Linear(self.hidden_dim, 1)

	def hidden_representation(self, x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index, edge_weight)
		x = F.relu(x)
		x = self.conv2(x, edge_index, edge_weight)
		x = global_mean_pool(x, batch)
		return self.clf(x), x

	def forward(self, x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index, edge_weight)
		x = F.relu(x)
		x = self.conv2(x, edge_index, edge_weight)
		x = global_mean_pool(x, batch)
		return self.clf(x)



