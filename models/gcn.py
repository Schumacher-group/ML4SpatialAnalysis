from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv, SSGConv, global_sort_pool, TopKPooling, global_mean_pool, GATConv

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import numpy as np
import wandb
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from mainutils.utils import visualise_cellgraph
import io
from PIL import Image
from mainutils.utils import coords_to_graph


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph_networks import GCN, SSGCN, EdgeWeightedGCN, HierarchicalGCN, AttentionGCN

GCN_DICT = {
			'gcn': GCN,
			'ssgcn': SSGCN,
			'edgegcn': EdgeWeightedGCN,
			'hiergcn': HierarchicalGCN,
			'attngcn': AttentionGCN

}

class FocalLoss(nn.Module):
	def __init__(self, alpha=0.25, gamma=2):
		super(FocalLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma

	def forward(self, logits, targets):
		bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
		p_t = torch.exp(-bce_loss)
		focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
		return focal_loss.mean()

class GraphConvolutionalNetwork:
	"""
	Class for training GCN models on TNBC (Triple-Negative Breast Cancer) expression and spatial data.

	Args:
		logger (optional, Logger): Logger object for logging training information.
	"""

	def __init__(self,
				gconv, 
				lr,
				batch_size,
				fnorm,
				logger=None,
				device='cpu',
				class_weight=None,
				gmethod='knn',
				radius=7,
				batch_correct=False,
				nm_batch=6,
				**kwargs):
		"""
		Initializes the TNBC GCN model and optimizer.

		Args:
			logger (optional, Logger): Logger object for logging training information.
		"""
		self.gconv = gconv
		self.batch_size = batch_size
		self.fnorm = fnorm
		self.logger = logger
		self.class_weight = class_weight
		self.gmethod = gmethod
		self.radius = radius
		self.batch_correct = batch_correct
		self.nm_batch = nm_batch

		self.device = torch.device(device)
		gcn_params = kwargs.get(self.gconv, None)	
		self.model = GCN_DICT[self.gconv](**gcn_params).to(self.device)
		
		self.optim = torch.optim.AdamW(
					self.model.parameters(), 
					lr=lr, 
					weight_decay=1e-4,
					betas=(0.9, 0.999)
			)


		weights = None if class_weight is None else torch.tensor(
				[weight for target, weight in class_weight[1]],
				dtype=torch.float32
				).to(self.device)

		self.criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
		#self.criterion = FocalLoss(alpha=0.25, gamma=2)

	def to_pyg(self, data_dict):
		"""
			Converts input data (gene expression, graphs, and optional labels) into PyTorch Geometric Data objects.

		Args:
			data (dict): A dictionary containing: expressions, enrichments, graphs, labels, and markers.
			expressions (list): List of gene expression data for each sample.
			coords (list): List of coords of cells.
			labels (list): List of labels (0 or 1) for each sample.
			markers (list): List of feature names. 

		Returns:
			dataset (list): List of PyTorch Geometric Data objects representing the samples.
		"""
		dataset = []
		num_samples = len(data_dict['labels'])
		for i in range(num_samples):
			graph_attributes = torch.tensor(data_dict['expressions'][i]).float()
			if self.fnorm == 'log1p_minmax':
				graph_attributes = torch.log1p(graph_attributes)
				min_marker, _ = torch.min(graph_attributes, dim=0)
				max_marker, _ = torch.max(graph_attributes, dim=0)
				graph_attributes = (graph_attributes - min_marker)/(max_marker - min_marker + 1e-8)
			elif self.fnorm == 'log1p_zscore':
				graph_attributes = torch.log1p(graph_attributes)
				graph_attributes = (graph_attributes - torch.mean(graph_attributes, dim=0, keepdim=True))/(torch.std(graph_attributes, dim=0, keepdim=True) + 1e-8)
			elif self.fnorm == 'arctan':
				graph_attributes = torch.arctan(graph_attributes)
				graph_attributes = (graph_attributes - torch.mean(graph_attributes, dim=0, keepdim=True))/(torch.std(graph_attributes, dim=0, keepdim=True) + 1e-8)
			elif self.fnorm == 'znorm':
				graph_attributes = (graph_attributes - torch.mean(graph_attributes, dim=0, keepdim=True))/(torch.std(graph_attributes, dim=0, keepdim=True) + 1e-8)
			elif self.fnorm == 'asinh':
				cofactor = 5.0
				graph_attributes = torch.asinh(graph_attributes / cofactor)
			elif self.fnorm == 'robust_scale':
				median = torch.median(graph_attributes, dim=0)[0]
				q75 = torch.quantile(graph_attributes, 0.75, dim=0)
				q25 = torch.quantile(graph_attributes, 0.25, dim=0)
				iqr = q75 - q25
				graph_attributes = (graph_attributes - median) / (iqr + 1e-8)
			elif self.fnorm == 'log2_mad':
				# Log2 transform with Median Absolute Deviation scaling
				graph_attributes = torch.log2(graph_attributes + 1)
				median = torch.median(graph_attributes, dim=0)[0]
				mad = torch.median(torch.abs(graph_attributes - median), dim=0)[0]
				graph_attributes = (graph_attributes - median) / (mad + 1e-8)
			elif self.fnorm == 'clr':
				# Centered Log Ratio transformation
				# Good for compositional data
				epsilon = 1e-8
				log_transform = torch.log(graph_attributes + epsilon)
				geometric_mean = torch.mean(log_transform, dim=1, keepdim=True)
				graph_attributes = log_transform - geometric_mean

			elif self.fnorm == 'percentile':
				# Percentile normalization (rescale to 0-1 based on percentiles)
				q99 = torch.quantile(graph_attributes, 0.99, dim=0)
				q1 = torch.quantile(graph_attributes, 0.01, dim=0)
				graph_attributes = torch.clamp(graph_attributes, q1, q99)
				graph_attributes = (graph_attributes - q1) / (q99 - q1 + 1e-8)

			elif self.fnorm == 'hybrid':
				# Hybrid approach: asinh transform followed by robust scaling
				cofactor = 5.0
				graph_attributes = torch.asinh(graph_attributes / cofactor)
				median = torch.median(graph_attributes, dim=0)[0]
				q75 = torch.quantile(graph_attributes, 0.75, dim=0)
				q25 = torch.quantile(graph_attributes, 0.25, dim=0)
				iqr = q75 - q25
				graph_attributes = (graph_attributes - median) / (iqr + 1e-8)
			elif self.fnorm == 'raw':
				pass
			else:
				raise NotImplementedError(f"{self.fnorm} not implemented")
			coords = data_dict['coords'][i]
			graph = coords_to_graph(coords, gmethod=self.gmethod, radius=self.radius)
			graph = graph.tocoo()
			label = torch.tensor(data_dict['labels'][i]).float()
			row, col, data = graph.row, graph.col, graph.data
			edge_index = torch.tensor((row, col)).long()
			edge_attr = torch.tensor(data).float()

			data_pyg = Data(x=graph_attributes, edge_index=edge_index, edge_attr=edge_attr, y=label)

			dataset.append(data_pyg)
		return dataset

	def fit(self, data):
		"""
		Trains the GCN model on the provided data.

		Args:
			data (dict): A dictionary containing: expressions, enrichments, graphs, labels, and markers.
			expressions (list): List of gene expression data for each sample.
			graphs (list): List of spatial adjacency matrices representing connections between samples.
			labels (list): List of labels (0 or 1) for each sample.
		"""
		self.model.train()
		dataset = self.to_pyg(data)
		loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
		loss_epoch = 0.
		for x_batch in loader:
			x_batch = x_batch.to(self.device)
			self.optim.zero_grad()
			logits, latent_z = self.model.hidden_representation(
				x_batch.x, 
				x_batch.edge_index, 
				x_batch.edge_attr, 
				x_batch.batch
			)
			
			loss = self.criterion(logits, x_batch.y.unsqueeze(1))

			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
			self.optim.step()

			loss_epoch += loss.item()
			self.logger.log({'GCN Iteration Loss': loss.item()})
			
		loss_epoch = loss_epoch/(len(loader))
		self.logger.log({'GCN Epoch Loss': loss.item()})

	def predict(self, data, threshold=0.5):
		"""
		Predicts class labels using the trained GCN model.

		Args:
			X (list): List of gene expression data for each sample.
			graphs (list): List of spatial adjacency matrices representing connections between samples.

		Returns:
			preds (np.array) : Array of predicted class labels (0 or 1) for each sample.
		"""
		self.model.eval()

		with torch.no_grad():
			dataset = self.to_pyg(data)
			loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
			preds = []
			for x_batch in loader:
				x_batch = x_batch.to(self.device)
				preds_i, latent_z = self.model.hidden_representation(x_batch.x, x_batch.edge_index, x_batch.edge_weight, x_batch.batch)
				#preds_i = F.softmax(preds_i, dim=1)
				#preds_i = preds_i.argmax(dim=1).cpu().numpy()
				preds_i = torch.sigmoid(preds_i).squeeze().cpu().numpy()
				preds.append(preds_i)
			preds = np.concatenate(preds)
			preds = (preds>=threshold).astype(int)
			return preds

	def predict_proba(self, data):
		"""
		Predicts class probabilities using the trained GCN model.

		Args:
			data (dict): A dictionary containing: expressions, enrichments, graphs, labels, and markers.
			expressions (list): List of gene expression data for each sample.
			graphs (list): List of spatial adjacency matrices representing connections between samples.
			labels (list): List of labels (0 or 1) for each sample.
			markers (list): List of feature names. 
		Returns:
			preds (np.array) : Array of predicted probability for each sample.
		"""
		self.model.eval()

		with torch.no_grad():
			pyg_dataset = self.to_pyg(data)
			loader = DataLoader(pyg_dataset, batch_size=self.batch_size, shuffle=False)
			probs = []
			for x_batch in loader:
				x_batch = x_batch.to(self.device)
				prob, latent_z = self.model.hidden_representation(x_batch.x, x_batch.edge_index, x_batch.edge_attr, x_batch.batch)
				#score = F.softmax(score, dim=1)
				prob = torch.sigmoid(prob)
				probs.append(prob)
			probs = torch.cat(probs, dim=0).cpu().numpy()
			return probs.squeeze()