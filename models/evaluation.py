import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import wandb
from models.abstract import AbstractModel
from mainutils.utils import compute_scores_train, compute_scores
from mainutils.utils import leave_one_out_split, patient_level_scores

class ModelEvaluation(AbstractModel):
	"""
	This class performs model evaluation and attribution.
	Attributes:
		config (dict): Configuration dictionary containing training parameters.
		classifier (object): Trained logistic regression model.
		feature_names (list): List of feature names.
		clf_name (str): Name of the classifier type.
	"""
	def __init__(self, config, logname, logger=None):
		"""
		Initializes the ModelEvaluation object.

		Args:
			config (dict): Configuration dictionary containing attribution parameters.
		"""
		super().__init__(config, logger)
		self.classifier = None
		self.scaler = None

	def run(self, dataset, logname):
		"""
		Logs the coefficients of the logistic regression model to W&B.

		Args:
			logger (wandb.Logger): W&B logger object.
		"""
		if self.config['eval'] == 'split':
			filename = f"{self.config['LOG_PATH']}/{self.config['name']}_{logname}.pkl"
			self.load_model(logname)
			self.evaluate(dataset['test'], mode='Test')

		elif self.config['eval'] == 'leaveOneOut':
			y_test = np.array([])
			y_pred_test = np.array([])
			y_pred_test_prob = np.array([])
			patient_label = np.array([])
			for i, (train_i, test_i) in enumerate(leave_one_out_split(dataset)):
				filename = f"{self.config['LOG_PATH']}/{logname}_leaveoneout_{i+1}_patient_{test_i['patient'][0]}.pkl"
				self.load_model(filename)

				# y_pred_train_i = self.predict(train_i)
				# y_pred_train_prob_i = self.predict_proba(train_i)[:, 1]

				# print(f"Evaluating the {self.config['name']} Model on Training Set")
				# print(f"[Fold {i+1}] Evaluating {self.config['name']} Model on TRAIN set")
				# train_metrics = compute_scores(
				# 				train_i['labels'],          # ground truth
				# 				y_pred_train_i,             # predicted labels
				# 				y_pred_train_prob_i,        # predicted probabilities
				# 				mode='Train'
				# )
				# self.log_metrics(train_metrics, mode=f"LOO_ROI_Train_Fold{i+1}")

				# metrics_train = patient_level_scores(
				# 				train_i['labels'], 
				# 				y_pred_train_i, 
				# 				y_pred_train_prob_i, 
				# 				train_i['patient'], 
				# 				mode='Train', 
				# 				pcriterion=self.config['pcriterion'])
				# print('Metrics at Patient Level', metrics_train)
				# self.log_metrics(metrics_train, mode=f"LeaveOneOutPatientLevelTrain {i+1}")

				y_pred_test_i = self.predict(test_i)
				y_pred_test_prob_i = self.predict_proba(test_i)[:, 1] 

				y_test = np.concatenate([y_test, test_i['labels']])
				y_pred_test = np.concatenate([y_pred_test, y_pred_test_i])
				y_pred_test_prob = np.concatenate([y_pred_test_prob, y_pred_test_prob_i])
				patient_label = np.concatenate([patient_label, test_i['patient']])

			print(f"[All Folds] Evaluating {self.config['name']} Model on the concatenated TEST folds")

			metrics = compute_scores(y_test, y_pred_test, y_pred_test_prob, mode='Test')
			self.log_metrics(metrics, mode=f"LeaveOneOutROILevel {i+1}")

			metrics_test = patient_level_scores(y_test, y_pred_test, y_pred_test_prob, patient_label, mode='Test', pcriterion=self.config['pcriterion'])
			self.log_metrics(metrics_test, mode='LeaveOneOutPatientLevelTest')
			print('Metrics at Patient Level', metrics_test)

		else:
			raise NotImplementedError(f"{self.config['eval']} Evaluation not implemented")

	def wandb_log_figure(self, fig, name):
		"""
		Helper to log a matplotlib figure to W&B or any logger with .log(...).
		"""
		buf = io.BytesIO()
		fig.savefig(buf, format='png')
		buf.seek(0)
		if self.logger:
			self.logger.log({name: wandb.Image(Image.open(buf))})
		plt.close(fig)

	def visualize_latent_space(self, data, method='PCA'):
		from sklearn.decomposition import PCA
		from sklearn.preprocessing import StandardScaler
		import umap

		with torch.no_grad():
			dataset = self.to_pyg(data)
			loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
			all_rois, all_labels = [], []
			for x_batch in loader:
				x_batch = x_batch.to(self.device)
				_, latent_z = self.model.hidden_representation(x_batch.x, x_batch.edge_index, x_batch.edge_weight, x_batch.batch)
				all_rois.append(latent_z)
				all_labels.append(x_batch.stain_y)
			all_rois = torch.cat(all_rois, dim=0).cpu().numpy()
			all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

		if method == 'PCA':
			scaler = StandardScaler()
			latent_patients = scaler.fit_transform(all_rois)
			reducer = PCA(n_components=2)
			projections = reducer.fit_transform(all_rois)
		elif method == 'UMAP':
			reducer = umap.UMAP()
			projections = reducer.fit_transform(all_rois)
		else:
			raise NotImplementedError('Not Implemented')
		
		unique_labels = np.unique(all_labels)
		colors = plt.cm.get_cmap('tab10', len(unique_labels))
		fig, ax = plt.subplots()

		for i, label in enumerate(unique_labels):
			mask = all_labels == label
			ax.scatter(projections[mask, 0], projections[mask, 1], color=colors(i), label=label)
		ax.set_title(f"ROI Clustering Visualization ({method})")
		ax.set_xlabel(f"{method} 1")
		ax.set_ylabel(f"{method} 2")
		ax.legend()
		self.wandb_log_figure(fig, f"Latent Encoding {method}")
		
	def pyg_attribution(self, data, topk=10):
		from torch_geometric.explain import Explainer, GNNExplainer
		import torch_geometric.utils as utils
		self.model.eval()
		explainer = Explainer(
						model=self.model,
						algorithm=GNNExplainer(epochs=100),
						explanation_type='phenomenon',
						node_mask_type='attributes',
						edge_mask_type='object',
						model_config=dict(
						mode='multiclass_classification',
						task_level='graph',
						return_type='log_probs',
						),
					)

		pyg_dataset = self.to_pyg(data)
		loader = DataLoader(pyg_dataset, batch_size=1, shuffle=False)
		
		feature_names = np.array(data['markers'])
		celltypes = data.get('celltypes', [])

		top_k_count_celltypespositive = {name: 0 for name in celltypes}
		top_k_count_celltypesnegative = {name: 0 for name in celltypes}

		top_k_count_positive = {name: 0 for name in feature_names}
		top_k_count_negative = {name: 0 for name in feature_names}

		labels_dict = {1:'Responder', 0:'Non-Responder'}
		
		for i, x_batch in enumerate(loader):
			x_batch = x_batch.to(self.device)

			kwargs = {'edge_weight':x_batch.edge_attr, 'batch': x_batch.batch}
			if hasattr(x_batch, 'batch'):
				kwargs['batch'] = x_batch.batch

			explanation = explainer(
					x_batch.x, 
					x_batch.edge_index, 
					target=x_batch.y.long(), 
					**kwargs
				)

			nm = explanation.node_mask
			em = explanation.edge_mask

			node_thresh = torch.quantile(nm, 0.85)
			edge_thresh = torch.quantile(em, 0.85)

			node_mask = (nm > node_thresh)
			edge_mask = (em > edge_thresh)

			# Feature importance
			scores = explanation.node_mask.mean(0).cpu().numpy()
			sorted_indices = scores.argsort()[::-1][:topk]
			sorted_scores = scores[sorted_indices]
			sorted_feature_names = feature_names[sorted_indices]

			label_val = int(x_batch.y.item())
			for feature in sorted_feature_names:
				if label_val == 1:
					top_k_count_positive[feature] += 1
				else:
					top_k_count_negative[feature] += 1

			sub_graph = x_batch.edge_index[:,edge_mask]
			retained_nodes = torch.nonzero((node_mask.sum(1) != 0), as_tuple=False).squeeze()

			src_dst = sub_graph.t().tolist()
			mask = [(src in retained_nodes) and (dst in retained_nodes) for src, dst in src_dst]
			mask = torch.tensor(mask)
			filtered_sub_graph = sub_graph[:, mask]
			unique_nodes = torch.unique(filtered_sub_graph)

			
			fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
			
			coo_adj_full = utils.to_scipy_sparse_matrix(
				x_batch.edge_index, 
				edge_attr=x_batch.edge_attr, 
				num_nodes=x_batch.x.shape[0]
			)
			if hasattr(x_batch, 'cell_labels'):
				node_labels = x_batch.cell_labels
			else:
				node_labels = None

			csr_adj_full = coo_adj_full.tocsr()

			_, _, pos, label_to_color, _ = visualise_cellgraph(csr_adj_full, random_state=42, node_labels=x_batch.cell_labels, show=False, spatial_coords=data['coords'][i], ax=ax1, add_legend=True)

			coo_adj_sub = utils.to_scipy_sparse_matrix(
				filtered_sub_graph, 
				edge_attr= torch.ones(filtered_sub_graph.size(1), dtype=torch.float), 
				num_nodes=x_batch.x.shape[0])
			csr_adj_sub = coo_adj_sub.tocsr()

			_, _, _, _, freq = visualise_cellgraph(csr_adj_sub, random_state=42, node_labels=x_batch.cell_labels, show=False, ax=ax2, pos=pos, label_to_color=label_to_color, largest_comp=True)

			self.wandb_log_figure(fig, f"results/SubGraph_{data['leapid'][i]}_{data['patient'][i]}_{labels_dict[x_batch.y.cpu().int().item()]}")


			fig2 = plt.figure(figsize=(32, 10))
			ax_left = fig2.add_subplot(1, 2, 1)
			ax_right = fig2.add_subplot(1, 2, 2)

			# Full Graph
			ax_left.bar(range(len(sorted_scores)), sorted_scores, tick_label=sorted_feature_names)
			ax_left.set_xlabel('Proteins', fontsize=28)
			ax_left.set_ylabel('Importance Score', fontsize=28)
			ax_left.set_title(f"GCN {labels_dict[x_batch.y.cpu().int().item()]} {data['patient'][i]}", fontsize=32)
			ax_left.tick_params(axis='x', labelrotation=45, labelsize=28)

			avg_expr = data['expressions'][i].mean(0)
			topk_expr = avg_expr[sorted_indices]
			# Subgraph
			ax_right.bar(range(len(topk_expr)), topk_expr, tick_label=sorted_feature_names)
			ax_right.set_xlabel('Proteins', fontsize=28)
			ax_right.set_ylabel('Average Expression Across Cells', fontsize=28)
			ax_right.set_title(f"GCN {labels_dict[x_batch.y.cpu().int().item()]} {data['patient'][i]}", fontsize=32)
			ax_right.tick_params(axis='x', labelrotation=45, labelsize=28)

			plt.tight_layout()
			self.wandb_log_figure(fig2, f"GNNExplainer ROI {data['leapid'][i]} {data['patient'][i]}")

			if label_val == 1:
				for celltype, count in freq.items():
					top_k_count_celltypespositive[celltype] += count
			else:
				for celltype, count in freq.items():
					top_k_count_celltypesnegative[celltype] += count			

		fig3, ax3 = plt.subplots(figsize=(12, 6))
		indices = np.arange(len(top_k_count_positive))
		# Plot the bars
		ax3.bar(indices, top_k_count_positive.values(), 0.35, label='Responder', color='skyblue')
		ax3.bar(indices + 0.35, top_k_count_negative.values(), 0.35, label='Non-responder', color='salmon')

		# Add some text for labels, title and axes ticks
		ax3.set_xlabel('Features', fontsize=14)
		ax3.set_ylabel(f"Top  {topk} Frequency", fontsize=14)
		ax3.set_title(f"Top {topk} Markers for Responder vs Non-Responder", fontsize=16)
		ax3.set_xticks(indices + 0.35 / 2)
		ax3.set_xticklabels(list(top_k_count_positive.keys()), rotation=45, ha='right')
		ax3.legend()
		plt.tight_layout()
		
		self.wandb_log_figure(fig3, 'Most Frequent Marker GNNExplainer across ROIs')


		fig4, ax4 = plt.subplots(figsize=(12, 6))
		indices = np.arange(len(top_k_count_celltypespositive))
		# Plot the bars
		val_pos = [el/(i+1) for el in list(top_k_count_celltypespositive.values())]
		val_neg = [el/(i+1) for el in list(top_k_count_celltypespositive.values())]
		
		ax4.bar(indices, val_pos, 0.35, label='Responder', color='skyblue')
		ax4.bar(indices + 0.35, val_neg, 0.35, label='Non-responder', color='salmon')

		# Add some text for labels, title and axes ticks
		ax4.set_xlabel('Features', fontsize=14)
		ax4.set_ylabel(f"Top  {topk} Frequency", fontsize=14)
		ax4.set_title(f"Top {topk} Markers for Responder vs Non-Responder", fontsize=16)
		ax4.set_xticks(indices + 0.35 / 2)
		ax4.set_xticklabels(list(top_k_count_celltypespositive.keys()), rotation=45, ha='right')
		ax4.legend()
		plt.tight_layout()
		self.wandb_log_figure(fig3, 'Most Frequent Celltypes GNNExplainer across ROIs')


	def gradient_attribution(self, data, topk=10, target_class=1):
		feature_names = data['markers']
		import matplotlib.pyplot as plt
		self.model.eval()
		pyg_dataset = self.to_pyg(data)
		loader = DataLoader(pyg_dataset, batch_size=1, shuffle=False)
		avg_node_gradients, std_node_gradients = [], []
		y_all = []
		for i, x_batch in enumerate(loader):
			x_batch = x_batch.to(self.device)
			self.model.zero_grad()
			x_batch.x.requires_grad = True
			logits = self.model.hidden_representation(x_batch.x, x_batch.edge_index, x_batch.edge_attr, x_batch.batch)
			logits[target_class].backward()
			y_all.append(x_batch.y)
			node_gradients = x_batch.x.grad

			avg_node_gradients.append(node_gradients.abs().mean(0))
			std_node_gradients.append(node_gradients.abs().std(0))

			node_gradients = node_gradients.T.abs().cpu().numpy() 
			top_k_idx = node_gradients.argsort(1)[:,-topk:]
			top_k_value = node_gradients[np.arange(node_gradients.shape[0])[:, None], top_k_idx]
			plt.imshow(top_k_value, cmap='hot', aspect='auto')
			plt.xlabel('Node Index')
			plt.ylabel('Attribute Index')
			plt.yticks(ticks=range(len(feature_names)), labels=feature_names, fontsize=8)
			#plt.xticks(range(len(feature_names)), feature_names, rotation=90)
			plt.title('Gradient of Node Attributes with Respect to log prob of a responder')
			plt.tight_layout()
			plt.colorbar(label='Gradient')
			self.logger.log({f"Attribution GCN batch {i}": plt})
			plt.clf()
			plt.cla()
		avg_node_gradients = torch.stack(avg_node_gradients).cpu().numpy()
		std_node_gradients = torch.stack(std_node_gradients).cpu().numpy()
		y_all = torch.stack(y_all).cpu().squeeze().numpy().astype('int')

		avg_node_gradients_resp = avg_node_gradients[y_all].mean(0)
		std_node_gradients_resp = std_node_gradients[y_all].mean(0)
		avg_node_gradients_noresp = avg_node_gradients[1 - y_all].mean(0)
		std_node_gradients_noresp = std_node_gradients[1 - y_all].mean(0)

		grad_df = pd.DataFrame({'Feature': np.array(feature_names), 
									'Mean Gradients pCR': avg_node_gradients_resp,
									'Std Gradients pCR': std_node_gradients_resp,
									'Mean Gradients Non-Responder': avg_node_gradients_noresp,
									'Std Gradients Non-Responder': std_node_gradients_noresp,
								})
		grad_df.set_index('Feature', inplace=True)
		fig = plt.figure(figsize=(16, 10))
		x = np.arange(len(feature_names))
		bar_width = 0.35
		plt.bar(x - bar_width/2, grad_df['Mean Gradients pCR'], yerr=grad_df['Std Gradients pCR'], width=bar_width, label='Responder', capsize=5, alpha=0.7)
		plt.bar(x + bar_width/2, grad_df['Mean Gradients pCR'], yerr=grad_df['Std Gradients Non-Responder'], width=bar_width, label='Non-Responder', capsize=5, alpha=0.7)
		#grad_df.plot(kind='bar', rot=0)
		#table = wandb.Table(dataframe=grad_df)
		# plt.xlabel('Protein Expres', fontsize=12)
		plt.ylabel('Gradient Averaged Across ROIs', fontsize=28)
		plt.title('Comparison of Gradients Averaged Across ROIs', fontsize=32)
		plt.legend(fontsize=12, prop={'size': 8})
		plt.xticks(ticks=np.arange(len(feature_names)), labels=grad_df.index, rotation=45, ha='right')
		plt.tight_layout()
		buffer = io.BytesIO()
		buffer.seek(0)
		plt.savefig(buffer, format='png')
		self.logger.log({'Average gradients across ROIs': wandb.Image(Image.open(buffer))})
