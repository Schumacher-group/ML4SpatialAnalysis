import pickle
import numpy as np
from mainutils.utils import compute_scores_train, compute_scores
from mainutils.utils import leave_one_out_split, patient_level_scores
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import wandb
import pandas as pd
from torch_geometric.seed import seed_everything
import matplotlib.pyplot as plt
import io
from PIL import Image
from models.gcn import GraphConvolutionalNetwork
from models.abstract import AbstractModel

MODELS_DICT = {
	'logistic' : LogisticRegression,
	'randomforest' : RandomForestClassifier,
	'xgboost' : XGBClassifier,
	'gnn' : GraphConvolutionalNetwork,
}

class ModelTrainer(AbstractModel):
	"""
	This class handles training, testing, and evaluation of models.

	Attributes:
		config (dict): Configuration dictionary containing training parameters.
		logfile (str): Name of the log file (optional).
		logger (wandb.Logger): W&B logger object.
		classifier (object): Trained model classifier.
	"""
	def __init__(self, config, 
						logger,
						logfile=None,
						seed=42,
						class_weight=None):
		"""
		Initializes the ModelTrainer object.

		Args:
			config (dict): Configuration dictionary containing training parameters.
			logger (wandb.Logger): W&B logger object.
			feature_names (list, optional): List of feature names. Defaults to None.
			logfile (str, optional): Name of the log file. Defaults to None.
		"""
		super().__init__(config, logger)
		self.seed = seed
		seed_everything(self.seed)
		# Choose and initialize classifier based on configuration
		self.config[self.config['name']]['random_state'] = seed
		if (class_weight is not None) and (self.config['name'] in ['logistic', 'randomforest', 'gnn']):
			self.config[self.config['name']]['class_weight'] = class_weight

		if (class_weight is not None) and self.config['name'] == 'xgboost':
			self.config[self.config['name']]['scale_pos_weight'] = class_weight[1]

		if self.config['name'] == 'gnn':
			self.config['gnn']['fnorm'] = self.config['fnorm']
			self.config['gnn']['logger'] = self.logger
			self.config[self.config['name']][self.config[self.config['name']]['gconv']]['input_dim'] = self.config['feature_dim']
			#self.config[self.config['name']][self.config[self.config['name']]['gconv']] ['hidden_dim'] = self.config['feature_dim']

		self.classifier = MODELS_DICT[self.config['name']](**self.config[self.config['name']])


	def optimise(self, dataset, logname):
		"""
		Optimizes the model by fitting, evaluating, and potentially saving it.

		Args:
			data (dict): A dictionary containing: expressions, enrichments (None for cell-cell case), graphs, labels, and feature names.
							expressions is a Feature matrix or a list of node attribute matrix.
			graphs (list, optional): List of graphs (for GCN models). Defaults to None.
		"""

		print(f"Fitting {self.config['name']} with {self.config['eval']} training")
		if self.config['eval'] == 'split':
			for epoch in range(self.config['nm_epochs']):
				self.fit(dataset['train'])
				if (epoch + 1) % self.config['test_every'] == 0:
					self.evaluate(dataset['train'], mode='Train')
					print(f"Evaluating on Test Set")
					self.evaluate(dataset['test'], mode='Test')
				if (epoch + 1) % self.config['save_every'] == 0:			
					print(f"Saving the {self.config['name']} Model")
					self.save_model(logname=logname)

		elif self.config['eval'] == 'leaveOneOut':
			y_test = np.array([])
			y_pred_test = np.array([])
			y_pred_test_prob = np.array([])
			patient_label = np.array([])
			for i, (train_i, test_i) in enumerate(leave_one_out_split(dataset)):
				for epoch in range(self.config['nm_epochs']):
					self.fit(train_i)

				y_pred_train_i = self.predict(train_i)
				y_pred_train_prob_i = self.predict_proba(train_i)[:, 1]

				print(f"Evaluating the {self.config['name']} Model on Training Set")
				print(f"[Fold {i+1}] Evaluating {self.config['name']} Model on TRAIN set")
				train_metrics = compute_scores(
								train_i['labels'],          # ground truth
								y_pred_train_i,             # predicted labels
								y_pred_train_prob_i,        # predicted probabilities
								mode='Train'
				)
				self.log_metrics(train_metrics, mode=f"LOO_ROI_Train_Fold{i+1}")

				metrics_train = patient_level_scores(
								train_i['labels'], 
								y_pred_train_i, 
								y_pred_train_prob_i, 
								train_i['patient'], 
								mode='Train', 
								pcriterion=self.config['pcriterion'])
				print('Metrics at Patient Level', metrics_train)
				self.log_metrics(metrics_train, mode=f"LeaveOneOutPatientLevelTrain {i+1}")

				y_pred_test_i = self.predict(test_i)
				y_pred_test_prob_i = self.predict_proba(test_i)[:, 1] 

				y_test = np.concatenate([y_test, test_i['labels']])
				y_pred_test = np.concatenate([y_pred_test, y_pred_test_i])
				y_pred_test_prob = np.concatenate([y_pred_test_prob, y_pred_test_prob_i])
				patient_label = np.concatenate([patient_label, test_i['patient']])

				self.save_leaveOO(test_i, logname=f"{logname}_leaveoneout_{i+1}_patient_{test_i['patient'][0]}")
				self.classifier = MODELS_DICT[self.config['name']](**self.config[self.config['name']])

			print(f"[All Folds] Evaluating {self.config['name']} Model on the concatenated TEST folds")

			metrics = compute_scores(y_test, y_pred_test, y_pred_test_prob, mode='Test')
			self.log_metrics(metrics, mode=f"LeaveOneOutROILevel {i+1}")

			metrics_test = patient_level_scores(y_test, y_pred_test, y_pred_test_prob, patient_label, mode='Test', pcriterion=self.config['pcriterion'])
			self.log_metrics(metrics_test, mode='LeaveOneOutPatientLevelTest')
			print('Metrics at Patient Level', metrics_test)
		else:
			raise NotImplementedError(f"{self.config['eval']} Evaluation not implemented")


