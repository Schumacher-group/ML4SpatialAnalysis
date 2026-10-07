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
from patientgnn.normalization import MarkerNormalizer


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
				gmethod='atmostk',
				gmode='connectivity',
				k=7,
				radius=7.0,
				node_f='expressions',
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
		self.gmode = gmode
		self.k = k
		self.radius = radius
		self.node_f = node_f
		self.feature_normalizer = None

		self.device = torch.device(device)
		gcn_params = kwargs.get(self.gconv, None)	
		self.model = GCN_DICT[self.gconv](**gcn_params).to(self.device)
		
		self.optim = torch.optim.AdamW(
					self.model.parameters(), 
					lr=lr, 
					weight_decay=1e-5,
					betas=(0.9, 0.999)
			)

		pos_weight = None
		if class_weight is not None:
			pos_weight = torch.tensor(
				[float(class_weight[1]) / float(class_weight[0])],
				dtype=torch.float32,
				device=self.device,
			)
		self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
		unique_celltypes = data_dict['celltypes']
		celltype_to_index = {celltype: idx for idx, celltype in enumerate(unique_celltypes)}
		for i in range(num_samples):
			raw_attributes = np.asarray(data_dict['expressions'][i], dtype=np.float32)
			if self.fnorm == 'raw':
				graph_attributes = torch.from_numpy(raw_attributes)
			else:
				if self.feature_normalizer is None:
					raise RuntimeError('Training feature normalization has not been fitted')
				graph_attributes = torch.from_numpy(self.feature_normalizer.transform(raw_attributes))

			cell_labels = data_dict['cell_labels'][i]
			celltype_one_hot = torch.zeros((len(cell_labels), len(unique_celltypes)))
			for j, label in enumerate(cell_labels):
				celltype_one_hot[j, celltype_to_index[label]] = 1.0
			# Combine features based on the selected option
			if self.node_f == 'expressions':
				node_features = graph_attributes
			elif self.node_f == 'celltypes':
				node_features = celltype_one_hot
			elif self.node_f == 'expressions_celltypes':
				node_features = torch.cat([graph_attributes, celltype_one_hot], dim=1)
			else:
				raise ValueError(f"Invalid option for use_features: {self.node_f}. Choose from 'expressions', 'celltypes', or 'both'.")

			coords = data_dict['coords'][i]
			graph = coords_to_graph(
				coords, gmethod=self.gmethod, mode=self.gmode, k=self.k, radius=self.radius
			)
			graph = graph.tocoo()
			label = torch.tensor(data_dict['labels'][i]).float()
			row, col, data = graph.row, graph.col, graph.data
			edge_index = torch.tensor((row, col)).long()
			edge_attr = torch.tensor(data).float()

			data_pyg = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=label)

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
		if self.fnorm != 'raw' and self.feature_normalizer is None:
			transform_name = 'identity'
			if self.fnorm.startswith('log'):
				transform_name = 'log1p'
			elif self.fnorm in {'asinh', 'hybrid'}:
				transform_name = 'asinh'
			patients = {}
			for index, expression in enumerate(data['expressions']):
				patient_id = str(data.get('patient', range(len(data['expressions'])))[index])
				patients.setdefault(patient_id, {'patient_id': patient_id, 'rois': []})
				patients[patient_id]['rois'].append({'expression': expression})
			self.feature_normalizer = MarkerNormalizer(
				transform_name=transform_name, max_cells_per_patient=20000
			).fit(list(patients.values()))
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
		self.logger.log({'GCN Epoch Loss': loss_epoch})

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
				preds_i, latent_z = self.model.hidden_representation(x_batch.x, x_batch.edge_index, x_batch.edge_attr, x_batch.batch)
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
