import pickle
import numpy as np
import wandb
import matplotlib.pyplot as plt
import io
from PIL import Image
from abc import ABC
from sklearn.preprocessing import StandardScaler
import os
from mainutils.utils import compute_scores, patient_level_scores
from mainutils.utils import graph_feature_vector, feature_normalisation

class AbstractModel(ABC):
	"""
		Abstract class for handling common functionalities of model training.

		Attributes:
		config (dict): Configuration dictionary containing training parameters.
		logger (wandb.Logger): W&B logger object (optional).
	"""

	def __init__(self, config, logger=None):
		self.config = config
		self.logger = logger
		self.scaler = None

	def cellcell_to_featurisation(self, X):
		"""
		Performs feature extraction based on averaging expression across genes for cell-cell contact.

		Args:
			X (list): List of expression data for each cell-cell interaction.

		Returns:
			np.ndarray: Array containing the extracted features.
		"""
		data_mat = feature_normalisation(X, self.config['fnorm'])
		data_mat = np.asarray([expr.mean(axis=0) for expr in data_mat])
		return data_mat


	def graph_features(self, graphs):
		"""
		Computes graph features based on the specified criterion.

		Args:
			graphs (list): List of graphs.

		Returns:
			np.ndarray: Array containing the computed graph features.
		"""
		gfeature_all = []
		for graph in graphs:
			gfeature, gfname = graph_feature_vector(graph, self.config['gcriterion'], self.config['gf_dim']) 
			gfeature_all.append(gfeature)
		return np.array(gfeature_all), gfname

	def fit(self, data):
		"""
		Fits the model to the training data.

		Args:
			X (np.ndarray): Feature matrix or a list of node attribute matrix.
			y (np.ndarray): Labels.
			graphs (list, optional): List of graphs (for GCN models). Defaults to None.
		"""
		if self.config['name'] == 'gnn':
			self.classifier.fit(data)
			return
		if self.config['name'] in ['logistic', 'randomforest', 'xgboost']:
			X = self.cellcell_to_featurisation(data['expressions'])
			if self.config['normalise_features']:
				self.scaler = StandardScaler()
				self.scaler.fit(X)
				X = self.scaler.transform(X)
			self.classifier.fit(X, data['labels'])
			return 

	def predict(self, data):
		"""
		Predicts labels for new data points.

		Args:
			X (np.ndarray): Feature matrix or a list of node attribute matrix.
			graphs (list, optional): List of graphs (for GCN models). Defaults to None.

		Returns:
			np.ndarray: Predicted labels.
		"""
		if self.config['name'] == 'gnn':
			return self.classifier.predict(data)
		if self.config['name'] in ['logistic', 'randomforest', 'xgboost']:
			X = self.cellcell_to_featurisation(data['expressions'])
			if self.config['normalise_features']:
				X = self.scaler.transform(X)
			return self.classifier.predict(X)

	def predict_proba(self, data):
		"""
		Predicts class probabilities for new data points.

		Args:
			X (np.ndarray): Feature matrix or a list of node attribute matrix.
			graphs (list, optional): List of graphs (for GCN models). Defaults to None.

		Returns:
			np.ndarray: Predicted class probabilities.
		"""
		if self.config['name'] == 'gnn':
			return self.classifier.predict_proba(data)
		if self.config['name'] in ['logistic', 'randomforest', 'xgboost']:
			X = self.cellcell_to_featurisation(data['expressions'])
			if self.config['normalise_features']:
				X = self.scaler.transform(X)
			return self.classifier.predict_proba(X)


	def evaluate(self, data, mode='Test'):
		"""
		Tests the model on new data and returns evaluation metrics.

		Args:
			data (dict): A dictionary containing: expressions, enrichments (None for cell-cell case), graphs, labels, and feature names.
							expressions is a Feature matrix or a list of node attribute matrix.
			graphs (list, optional): List of graphs (for GCN models). Defaults to None.
		"""
		y_pred = self.predict(data)
		y_proba = self.predict_proba(data)
		#y_proba = y_proba[:, 1]
		metrics_roi = compute_scores(data['labels'], y_pred, y_proba, mode)
		self.log_metrics(metrics_roi,  mode=f"ROILevel{mode}")
		print('Metrics at ROI Level', metrics_roi)

		metrics_patient = patient_level_scores(data['labels'], y_pred, y_proba, data['patient'], mode=mode, pcriterion=self.config['pcriterion'])
		self.log_metrics(metrics_patient, mode=f"Patient Level {mode}")
		print('Metrics at Patient Level', metrics_patient)

	def save_model(self, logname):
		"""
		Saves the trained model and feature names to a file.

		Args:
			fold (str, optional): Fold number for cross-validation (optional). Defaults to None.
		"""
		filename = f"{self.config['LOG_PATH']}/{self.config['name']}_{logname}.pkl"

		if not os.path.exists(self.config['LOG_PATH']):
			os.makedirs(self.config['LOG_PATH'])
		
		out = {
				'model': self.classifier,
				'scaler': self.scaler
				}
		
		with open(filename, 'wb') as f:
			pickle.dump(out, f)


	def save_leaveOO(self, test, logname=None):
		"""
		Saves the trained model and feature names to a file.

		Args:
			fold (str, optional): Fold number for cross-validation (optional). Defaults to None.
		"""
		filename = f"{self.config['LOG_PATH']}/{logname}.pkl"

		if not os.path.exists(self.config['LOG_PATH']):
			os.makedirs(self.config['LOG_PATH'])
		
		out = {
				'model': self.classifier,
				'scaler': self.scaler,
				'data': test
				}
		
		with open(filename, 'wb') as f:
			pickle.dump(out, f)

	def load_model(self, filename):
		with open(filename, 'rb') as f:
			load = pickle.load(f)
		self.classifier = load['model']
		self.scaler = load['scaler']

	def log_metrics(self, metrics, mode='Train'):
		"""
		Logs the evaluation metrics to W&B.

		Args:
			metrics (dict): Dictionary containing evaluation metrics.
		"""
		metrics_table=[[key, value] for key, value in metrics.items()]
		self.logger.log({
					f"{mode} Metrics": 
							wandb.Table(
									data=metrics_table, 
								columns=['Metric', 'Value'])
					})
