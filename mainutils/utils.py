import yaml
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph, NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix
from matplotlib.cm import ScalarMappable
import matplotlib.patches as patches
from collections import Counter
from sklearn.metrics import (
	accuracy_score, 
	balanced_accuracy_score, 
	f1_score,
	roc_auc_score, 
	roc_curve
	)
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
from scipy.spatial import QhullError
import warnings
import pandas as pd
import wandb

def load_config(filename):
	"""
	Loads configuration from a YAML file.

	Args:
		filename (str): Path to the configuration YAML file.

	Returns:
		dict: Dictionary containing the loaded configuration.
	"""
	with open(filename, 'r') as f:
		return yaml.safe_load(f)


def edge_index_to_adj(edge_index, num_nodes):
	row = edge_index[0].cpu().numpy()
	col = edge_index[1].cpu().numpy()
	data = np.ones_like(row)
	adj = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
	return adj


def distance_to_similarity(
	G, 
	method='inverse', 
	eps=1e-9, 
	sigma=1.0,
	copy=True
):
	"""
	Converts the nonzero distances in a CSR matrix G into similarity values, 
	using either an inverse or Gaussian transform (or others, if extended).

	Args:
		G (sp.csr_matrix): A sparse matrix whose .data hold distance values.
		method (str): 'inverse' or 'Gaussian'. Defaults to 'inverse'.
		eps (float): Small constant to avoid division by zero in 'inverse'.
		sigma (float): Gaussian sigma. Used only if method='Gaussian'.
		copy (bool): If True, make a copy of G before modifying. 
	     If False, do in-place transformation.

	Returns:
		sp.csr_matrix: The same shape NxN matrix with .data now storing similarity values.
	"""
	# If desired, create a copy so we don't overwrite the original
	if copy:
		G = G.copy()

	# Ensure we're working with CSR format
	G_csr = G.tocsr()

	# Pull out the distance data
	distances = G_csr.data

	# Apply the requested similarity transform
	if method == 'inverse':
		G_csr.data = 1.0 / (distances + eps)
	elif method == 'Gaussian':
		G_csr.data = np.exp(-(distances**2) / (2.0 * sigma**2))
	else:
		raise ValueError(f"Unsupported similarity method: {method}")
	return G_csr


def adjacency_to_laplacian(A, normalised=True):
	"""
	Computes the Laplacian matrix (normalized or unnormalized) from an adjacency matrix.

	Args:
		A (numpy.ndarray or scipy.sparse.csr_matrix): Adjacency matrix of a graph.
		normalised (bool, optional): True to compute the normalized Laplacian, False for unnormalized.
		Defaults to True.

	Returns:
		Lnorm (numpy.ndarray or scipy.sparse.csr_matrix): The computed Laplacian matrix.
	"""
	D = np.squeeze(np.asarray(A.sum(axis=1)))
	L = sp.diags(D) - A if sp.issparse(A) else np.diag(D) - A
	if not normalised:
		return L
	Dsqrt = 1.0/ np.sqrt(D)
	Dsqrt[Dsqrt==np.inf] = 0
	Dsqrt = sp.diags(Dsqrt) if sp.issparse(A) else np.diag(Dsqrt)
	Lnorm = Dsqrt.dot(L).dot(Dsqrt)
	return Lnorm


def delaunay_graph(coords, mode='connectivity', dtype=np.float32):
	"""
	Build an NxN adjacency matrix from the Delaunay triangulation of coordinates.

	Parameters:
	-----------
		coords : array-like, shape (n_points, n_dimensions)
				The coordinates of points to triangulate
		mode : {'connectivity', 'distance'}, default='connectivity'
				The type of adjacency matrix to construct:
				- 'connectivity': Binary adjacency matrix (1 for connected points)
				- 'distance': Weighted adjacency matrix with Euclidean distances
		dtype : numpy.dtype, default=np.float32
		Data type for the output matrix
	Returns:
	--------
	scipy.sparse.csr_matrix
	The adjacency matrix in CSR format
	"""
	# Input validation
	if mode not in ['connectivity', 'distance']:
		raise ValueError(f"Unsupported mode: {mode}")

	coords = np.asarray(coords)
	if coords.ndim != 2:
		raise ValueError("coords must be a 2D array")

	n_points = len(coords)

	# Delaunay is undefined for fewer than three points; connect what is there.
	if n_points < 3:
		A = sp.lil_matrix((n_points, n_points), dtype=dtype)
		if n_points == 2:
			val = 1.0 if mode == 'connectivity' else float(np.linalg.norm(coords[0] - coords[1]))
			A[0, 1] = A[1, 0] = val
		return A.tocsr()

	# Compute triangulation. Degenerate ROIs (collinear or coincident centroids)
	# make Qhull fail on the initial simplex; joggling the input breaks the tie.
	try:
		triangulation = Delaunay(coords)
	except QhullError:
		triangulation = Delaunay(coords, qhull_options='QJ')

	# Pre-allocate arrays for COO matrix construction
	# Each triangle (in 2D) has 3 edges, and we add each edge twice (i->j, j->i) => 6 entries
	n_triangles = len(triangulation.simplices)
	rows = np.zeros(n_triangles * 6, dtype=np.int32)
	cols = np.zeros(n_triangles * 6, dtype=np.int32)
	data = np.zeros(n_triangles * 6, dtype=dtype)

	idx = 0
	for simplex in triangulation.simplices:
		# Process each edge in the triangle
		for i in range(3):
			for j in range(i+1, 3):
				p1, p2 = simplex[i], simplex[j]
				if mode == 'connectivity':
					val = 1.0
				else:  # mode == 'distance'
					diff = coords[p1] - coords[p2]
					val = np.sqrt(np.sum(diff * diff))
	
				# Add edge in both directions for a symmetric adjacency
				rows[idx] = p1
				cols[idx] = p2
				data[idx] = val
				idx += 1

				rows[idx] = p2
				cols[idx] = p1
				data[idx] = val
				idx += 1

	# Trim arrays to the actual size used
	rows = rows[:idx]
	cols = cols[:idx]
	data = data[:idx]

	# Create a COO matrix (potentially with duplicate edges)
	A_coo = sp.coo_matrix((data, (rows, cols)), shape=(n_points, n_points), dtype=dtype)

	# Average duplicate triangle contributions without halving boundary edges.
	A = A_coo.tocsr()
	counts = sp.coo_matrix(
		(np.ones_like(data), (rows, cols)), shape=(n_points, n_points)
	).tocsr()
	A.data = A.data / counts.data
	A = A.maximum(A.T)
	return A

def clamp_k(k, n_samples, context=''):
	"""
	Caps the neighbourhood size at what an ROI supports.

	A node can have at most n_samples - 1 distinct neighbours, and both
	kneighbors_graph and the KDTree query raise once k exceeds that. 

	Args:
		k (int): Requested number of neighbours.
		n_samples (int): Number of cells in the ROI.
		context (str, optional): Label used in the warning message.

	Returns:
		int: The usable number of neighbours, at least 1.
	"""
	k = int(k)
	if k < 1:
		raise ValueError(f"k must be a positive integer, got {k}")
	k_max = n_samples - 1
	if k > k_max:
		warnings.warn(
			f"{context or 'graph'}: k={k} exceeds the {k_max} neighbours available "
			f"in an ROI of {n_samples} cells; using k={k_max}",
			RuntimeWarning,
			stacklevel=2,
		)
		return k_max
	return k


def atmostk_neighbors_graph(
	coords,
	k,
	mode='connectivity',
	include_self=False,
	):
	"""
	Constructs a k-nearest neighbor graph where each node has at most k neighbors,
	and edges are only created if they're within a global distance threshold.
	...
	"""
	if not isinstance(coords, np.ndarray):
		coords = np.asarray(coords)

	if coords.ndim != 2:
		raise ValueError("coords must be a 2D array")

	n_samples = coords.shape[0]
	if n_samples < 2:
		return csr_matrix((n_samples, n_samples), dtype=np.float64)

	k = clamp_k(k, n_samples, context='atmostk')

	# Query exactly k candidates.
	n_neighbors = min(n_samples - 1, k)
	tree = KDTree(coords)
	distances, indices = tree.query(coords, n_neighbors + 1)

	# 2) Remove or keep self references
	if not include_self:
		distances = distances[:, 1:]
		indices = indices[:, 1:]
	else:
		distances = distances[:, :n_neighbors]
		indices = indices[:, :n_neighbors]

	# Robust physical-density threshold derived from first-neighbour spacing.
	# Nodes may have fewer than k neighbours and GCNConv supplies self-loops.
	first_neighbor = distances[:, 0]
	threshold = 1.5 * np.quantile(first_neighbor, 0.90)

	# 4) Build adjacency in COO format
	rows = []
	cols = []
	data = []

	for i in range(n_samples):
		# Filter neighbors by threshold
		valid_mask = distances[i] <= threshold
		valid_neighbors = indices[i][valid_mask]
		valid_dists = distances[i][valid_mask]

		# Keep only up to k of the closest among those within threshold
		if len(valid_neighbors) > k:
			valid_neighbors = valid_neighbors[:k]
			valid_dists = valid_dists[:k]

		rows.extend([i] * len(valid_neighbors))
		cols.extend(valid_neighbors)

		if mode == 'distance':
			data.extend(valid_dists)
		else:  # 'connectivity'
			data.extend([1.0] * len(valid_neighbors))


	adjacency = coo_matrix(
		(data, (rows, cols)),
		shape=(n_samples, n_samples)
	)

	adjacency = adjacency.maximum(adjacency.T)
	return adjacency.tocsr()


GRAPH_METHODS = ('knn', 'atmostk', 'delaunay', 'radius')


def coords_to_graph(coords, gmethod='atmostk', mode='connectivity', k=7, radius=7.0):
	"""
	Constructs a graph from coordinates using specified method (k-nearest neighbors, k-atmost neighbors,
	Delaunay triangulation or radius-based).

	Args:
		coords (numpy.ndarray): Array of coordinates representing nodes.
		mode: ‘connectivity’, ‘distance’
		gmethod (str, optional): Graph construction method. Defaults to 'atmostk',
									matching configs/config.yaml.
		k (int, optional): Number of nearest neighbours. Used by 'knn' and 'atmostk', ignored by
									'delaunay' and 'radius'. Clamped to n_cells - 1 for small ROIs.
									Defaults to 7.
		radius (float, optional): Radius for radius-based graph construction. Used only if gmethod='radius'.
									Defaults to 7.

	Returns:
		G (scipy.sparse.csr_matrix): The constructed graph adjacency matrix.
	"""
	coords = np.asarray(coords)
	if coords.ndim != 2:
		raise ValueError("coords must be a 2D array")
	n_samples = coords.shape[0]

	if gmethod == 'radius':
		G = radius_neighbors_graph(
			coords,
			radius,
			mode=mode,
			include_self=False)
	elif gmethod == 'knn':
		if n_samples < 2:
			return csr_matrix((n_samples, n_samples), dtype=np.float64)
		G = kneighbors_graph(
			coords,
			clamp_k(k, n_samples, context='knn'),
			mode=mode,
			include_self=False)
		G = G.maximum(G.T)
	elif gmethod == 'atmostk':
		G = atmostk_neighbors_graph(
			coords,
			k,
			mode=mode,
			include_self=False)
	elif gmethod == 'delaunay':
		G = delaunay_graph(
			coords,
			mode=mode)
	else:
		raise ValueError(f"Unsupported gmethod {gmethod!r}; choose one of {GRAPH_METHODS}")
	if mode == 'distance':
		G = distance_to_similarity(G)
	return G

def graph_feature_vector(graph, gcriterion='heat_trace', feature_dim=10):
	"""
	Extracts graph features based on provided criteria.

	Args:
		graph (scipy.sparse.csr_matrix): Adjacency matrix of the graph.
		gcriterion (str, optional): Feature extraction criterion, either 'heat_trace', or
											'laplacian_spectrum'. Defaults to 'heat_trace'.
		feature_dim (int, optional): Desired dimensionality of the feature vector. Defaults to 10.

	Returns:
		feature_vector (numpy.ndarray): Array containing the extracted graph features.
	"""
	if not isinstance(graph, csr_matrix):
		raise ValueError('nn_graph must be a scipy sparse CSR matrix.')

	num_nodes = graph.shape[0]
	feature_vector = np.zeros(feature_dim, dtype=np.float64)
	if gcriterion == 'heat_trace':
		# Compute heat trace on the graph at different timescales
		timescales = np.logspace(-2, 2, num=feature_dim)

		# Compute normalized Laplacian matrix
		laplacian = adjacency_to_laplacian(graph)
		# Compute eigenvalues of normalized Laplacian
		k = min(feature_dim, num_nodes-2)
		eivals, _ = eigs(laplacian, k=k, which='SM')

		feature_names = []
		# Compute heat trace at different timescales
		for t, timescale in enumerate(timescales):
			feature_vector[t] = np.sum(np.exp(-timescale * eivals.real))
			feature_names.append(f"_HeatTrace_{t:.3f}")

	elif gcriterion == 'laplacian_spectrum':
		# Compute normalized Laplacian matrix
		laplacian = adjacency_to_laplacian(graph)
		k = min(feature_dim, num_nodes-1)
		eivals = sp.linalg.svds(laplacian, k=k, return_singular_vectors=False)
		feature_vector = np.zeros(feature_dim)
		if len(eivals)>1:
			feature_vector[-k:] = sorted(eivals)
		feature_names = [f"_eigen_{i}" for i in range(feature_dim)]

	elif gcriterion == 'graphproperties':
		num_edges = graph.getnnz() / 2.0  # Divide by 2 since the matrix is symmetric
		num_nodes = graph.shape[0]
		density = num_edges / (num_nodes * (num_nodes - 1) / 2)  # Complete graph denominator

		average_degree = np.mean(np.sum(graph != 0, axis=0))
		graphnx = nx.from_numpy_array(graph.toarray())
		clustering_coefficient = nx.average_clustering(graphnx)
		#avg_path = nx.average_shortest_path_length(graphnx)
		connectivity = 1.0 if nx.is_connected(graphnx) else 0.0

		feature_vector = np.array([num_nodes, num_edges, density, clustering_coefficient, average_degree, connectivity])
		feature_names = ['_num_nodes', '_num_edges', '_density', '_clustcoeff', '_avgdegree', '_connectivity']
	else:
		raise NotImplementedError(f" {gcriterion} Not implemented. Valid options are `degree`, or `heat_trace`.")

	return feature_vector, feature_names


def compute_scores_train(y_train, y_pred_train, y_test, y_pred_test):
	"""
	Computes various evaluation scores for both training and test sets.

	Args:
		y_train (list or numpy.ndarray): True labels for training set.
		y_pred_train (list or numpy.ndarray): Predicted labels for training set.
		y_test (list or numpy.ndarray): True labels for test set.
		y_pred_test (list or numpy.ndarray): Predicted labels for test set.

	Returns:
		metrics (dict): Dictionary containing evaluation metrics for both sets.
	"""
	accuracy_train = accuracy_score(y_train, y_pred_train)
	balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)
	auc_train = roc_auc_score(y_train, y_pred_train)
	f1_train = f1_score(y_train, y_pred_train)

	accuracy_test = accuracy_score(y_test, y_pred_test)
	balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)
	auc_test = roc_auc_score(y_test, y_pred_test)
	f1_test = f1_score(y_test, y_pred_test)
	metrics = {
			'Accuracy Train': accuracy_train,
			'Balanced_Accuracy Train': balanced_accuracy_train,
			'Accuracy Test': accuracy_test,
			'Balanced_Accuracy Test': balanced_accuracy_test,
			'AUC Train': auc_train,
			'AUC Test': auc_test,
			'F1 Score Train': f1_train,
			'F1 Score Test': f1_test,			
	}
	return metrics



def compute_scores(y, y_pred, y_proba, mode='Train'):
	"""
	Computes evaluation scores for a single test set.

	Args:
		y (list or numpy.ndarray): True labels.
		y_pred (list or numpy.ndarray): Predicted labels.

	Returns:
		metrics (dict): Dictionary containing evaluation metrics.
	"""
	accuracy_ = accuracy_score(y, y_pred)
	auc_ = roc_auc_score(y, y_proba)
	bal_acc_ = balanced_accuracy_score(y, y_pred)
	metrics = {
			f"{mode} Accuracy": accuracy_,
			f"{mode} AUC": auc_,
			f"{mode} Balanced Accuracy": bal_acc_,

	}
	return metrics

def patient_level_scores(y, y_pred, y_proba, patients, mode='Test', pcriterion='majority'):
	unique_patients = list(set(patients))
	patients_preds = {patient : [] for patient in unique_patients}
	patients_labels = {patient : [] for patient in unique_patients}
	patients_probs = {patient : [] for patient in unique_patients}

	for patient, label, pred, prob in zip(patients, y, y_pred, y_proba):
		patients_preds[patient].append(int(pred))
		patients_probs[patient].append(float(prob))
		patients_labels[patient].append(int(label))

	unique_pred_patients_prob, unique_pred_patients_label, unique_patients_label = [], [], []
	for patient in unique_patients:
		roi_probs = np.array(patients_probs[patient])
		roi_labels = np.array(patients_labels[patient])

		patient_label = Counter(roi_labels).most_common(1)[0][0]
		unique_patients_label.append(patient_label)

		if pcriterion == 'majority':
			# A patient score must not be the most extreme ROI, because that makes
			# AUC depend on the number of ROIs available for a patient.
			prob_patient = float(np.mean(roi_probs))
			vote_patient = int(prob_patient >= 0.5)
			unique_pred_patients_label.append(vote_patient)
			unique_pred_patients_prob.append(prob_patient)

		elif pcriterion == 'weighted_mean': 
			confidences = np.abs(roi_probs - 0.5) + 0.5
			weights = confidences / confidences.sum()
			prob_patient = np.average(roi_probs, weights=weights)
			pred_patient = int(prob_patient >= 0.5)
			unique_pred_patients_label.append(pred_patient)
			unique_pred_patients_prob.append(prob_patient)

		elif pcriterion == 'geometric_mean':
			# Use geometric mean of odds ratios
			eps = 1e-7  # Small epsilon to prevent division by zero
			odds = (roi_probs + eps) / (1 - roi_probs + eps)
			geometric_odds = np.exp(np.mean(np.log(odds)))
			prob_patient = geometric_odds / (1 + geometric_odds)
			pred_patient = int(prob_patient >= 0.5)
			unique_pred_patients_label.append(pred_patient)
			unique_pred_patients_prob.append(prob_patient)

		elif pcriterion == 'consensus':
			# Require strong consensus for positive prediction
			consensus_threshold = 0.75
			roi_preds = (roi_probs >= 0.5).astype(int)
			positive_ratio = np.mean(roi_preds)
			pred_patient = int(positive_ratio >= consensus_threshold)
			# Use mean probability of the consensus class
			prob_patient = np.mean(roi_probs[roi_preds == pred_patient])
			unique_pred_patients_label.append(pred_patient)
			unique_pred_patients_prob.append(prob_patient)

		else:
			raise NotImplementedError(f"{pcriterion} Not Implemented")

	unique_pred_patients_prob = np.array(unique_pred_patients_prob)
	unique_patients_label = np.array(unique_patients_label)

	patient_auc = roc_auc_score(unique_patients_label, unique_pred_patients_prob)
	accuracy = accuracy_score(unique_patients_label, unique_pred_patients_label)
	bal_acc_ = balanced_accuracy_score(unique_patients_label, unique_pred_patients_label)

	metrics = {
			f"{mode} Accuracy {pcriterion}": accuracy,
			f"{mode} AUC {pcriterion}": patient_auc,
			f"{mode} Balanced Accuracy {pcriterion}": bal_acc_,
	}

	# Plot & log Patient-level ROC
	fpr_patient, tpr_patient, _ = roc_curve(unique_patients_label, unique_pred_patients_prob)

	plt.figure()
	plt.plot(fpr_patient, tpr_patient, label=f"Patient-Level ROC (AUC={patient_auc:.3f})")
	plt.plot([0, 1], [0, 1], 'r--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(f"{mode} Patient-Level ROC Curve")
	plt.legend(loc="lower right")
	wandb.log({f"{mode}_Patient_Level_ROC": wandb.Image(plt)})
	plt.close()

	# ROI-level metrics & ROC
	#  The question specifically mentions plotting AUC-ROC at ROI level as well.
	roi_auc = roc_auc_score(y, y_proba)
	fpr_roi, tpr_roi, _ = roc_curve(y, y_proba)

	# Plot & log ROI-level ROC
	plt.figure()
	plt.plot(fpr_roi, tpr_roi, label=f"ROI-Level ROC (AUC={roi_auc:.3f})")
	plt.plot([0, 1], [0, 1], 'r--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(f"{mode} ROI-Level ROC Curve")
	plt.legend(loc="lower right")
	wandb.log({f"{mode}_ROI_Level_ROC": wandb.Image(plt)})
	plt.close()
	return metrics



def train_test_split(dataset, test_size=0.2, random_state=None):
	"""
	Custom train-test split for a dictionary-like dataset.

	Args:
		dataset (dict): Dictionary where keys represent different data matrices or labels.
		test_size (float): Ratio of the dataset to include in the test set.
		random_state (int or None): Random seed for reproducibility.

	Returns:
		train_set (dict): Dictionary containing train split for each key.
		test_set (dict): Dictionary containing test split for each key.
	"""
	np.random.seed(random_state)

	nm_samples =  len(dataset['labels'])
	patient_ids = dataset['patient']
	df = pd.DataFrame({'patient_id': dataset['patient'], 'label': dataset['labels']})

	unique_patients = df.groupby('patient_id')['label'].agg(lambda x: x.iloc[0])
	unique_patient_ids = unique_patients.index
	unique_patient_labels = unique_patients.values

	# Determine the test size based on the proportion of unique patients
	test_size = int(test_size * len(unique_patient_ids))

	from sklearn.model_selection import train_test_split as sk_split
	# Split the unique patients into training and test sets with stratification
	train_patient_ids, test_patient_ids = sk_split(
		unique_patient_ids, 
		test_size=test_size, 
		stratify=unique_patient_labels,  # Ensure the split is proportional based on labels
		random_state=random_state
		)

	train_indices = [i for i, patient in enumerate(patient_ids) if patient in train_patient_ids]
	test_indices = [i for i, patient in enumerate(patient_ids) if patient in test_patient_ids]

	train_set = {}
	test_set = {}

	for key, data in dataset.items():
		if data is None:
			train_set[key] = None
			test_set[key] = None
		elif key == 'markers' or key=='celltypes':
			train_set[key] = data
			test_set[key] = data
		else:
			train_set[key] = [data[i] for i in train_indices]
			test_set[key] = [data[i] for i in test_indices]

	return train_set, test_set


def k_fold_split(dataset, k=5, random_state=None):
	"""
	Custom k-fold split for a dictionary-like dataset.

	Args:
		dataset (dict): Dictionary where keys represent different data matrices or labels.
		k (int): Number of folds.
		random_state (int or None): Random seed for reproducibility.

	Returns:
		fold_sets (list): List of k fold sets, where each fold set is a tuple containing train and test splits for each key.
	"""
	np.random.seed(random_state)
	num_samples = len(dataset['labels'])
	unique_ids, inverse_indices = np.unique(np.array(dataset['patient']), return_inverse=True)

	fold_indices = np.array_split(np.random.permutation(len(unique_ids)), k)

	fold_sets = []
	for fold_idx in range(k):
		test_unique_ids = unique_ids[fold_indices[fold_idx]]
		train_unique_ids = unique_ids[np.concatenate([fold_indices[i] for i in range(k) if i != fold_idx])]

		test_indices = np.where(np.isin(dataset['patient'], test_unique_ids))[0]
		train_indices = np.where(np.isin(dataset['patient'], train_unique_ids))[0]

		train_set = {}
		test_set = {}

		for key, data in dataset.items():
			if data is None:
				train_set[key] = None
				test_set[key] = None
			elif key in ['markers', 'celltypes']:
				train_set[key] = data
				test_set[key] = data
			else:
				train_set[key] = [data[i] for i in train_indices]
				test_set[key] = [data[i] for i in test_indices]

		fold_sets.append((train_set, test_set))
	return fold_sets


def leave_one_out_split(data):
	unique_patients = list(set(data['patient']))

	for leave_out_patient in unique_patients:
		print(f"Leave One Out Validation on a patient {leave_out_patient}")
		train_patients = [patient for patient in unique_patients if patient != leave_out_patient]
		train_idx = [idx for idx, patient in enumerate(data['patient']) if patient != leave_out_patient]
		test_idx = [idx for idx, patient in enumerate(data['patient']) if patient == leave_out_patient]

		train_set, test_set = {}, {}
		for key, values in data.items():
			if values is None:
				train_set[key] = None
				test_set[key] = None
			elif key in ['markers', 'celltypes']:
				train_set[key] = values
				test_set[key] = values
			else:
				train_set[key] = [values[i] for i in train_idx]
				test_set[key] = [values[i] for i in test_idx]

		yield train_set, test_set


def feature_normalisation(X, fnorm):
	if fnorm == 'raw':
		return X
	if fnorm == 'znorm':
		return [((expr - np.mean(expr, axis=0, keepdims=True))/(1e-8 + np.std(expr, axis=0, keepdims=True))) for expr in X]
	if fnorm == 'log1p':
		X_norm = []
		for expr in X:
			expr_s = np.log1p(expr)
			expr_ns = (expr_s - np.mean(expr_s, axis=0, keepdims=True))/(1e-8 + np.std(expr_s, axis=0, keepdims=True))
			X_norm.append(expr_ns)
		return X_norm
	if fnorm == 'minmax':
		X_norm = []
		for expr in X:
			expr_s = np.log1p(expr)
			expr_ns = (expr - expr.min(axis=0))/(expr.max(axis=0) - expr.min(axis=0) + 1e-8)
			X_norm.append(expr_ns)
		return X_norm
	if fnorm == 'arctan':
		X_norm = []
		for expr in X:
			expr_s = np.arctan(expr)
			expr_ns = (expr_s - np.mean(expr_s, axis=0, keepdims=True))/(1e-8 + np.std(expr_s, axis=0, keepdims=True))
			X_norm.append(expr_ns)
		return X_norm
		

def visualise_cellgraph(graph, random_state=42, node_labels=None, show=True, spatial_coords=None, ax=None, pos=None, add_legend=False, label_to_color=None, largest_comp=None):
	"""
	Visualizes a cell graph using NetworkX and Matplotlib.

	Args:
		graph: A scipy.sparse matrix representing the cell graph.
		random_state: An integer seed for reproducibility of the layout algorithm (default: 42).
		node_labels: An optional numpy array of node labels to color-code the nodes.
	"""
	np.random.seed(random_state)

	edges = []
	for i in range(graph.shape[0]):
		for j in graph.indices[graph.indptr[i]:graph.indptr[i+1]]:
			edges.append((i, j))
	# Create a NetworkX graph and add edges
	G = nx.Graph()
	G.add_edges_from(edges)
	G.remove_edges_from(nx.selfloop_edges(G))

	if largest_comp is not None:
		# Identify the largest connected component
		largest_cc = max(nx.connected_components(G), key=len)
		G = G.subgraph(largest_cc).copy()
	if node_labels is not None:
		node_labels = np.array(node_labels)
		node_labels = [node_labels[i] for i in G.nodes]

	if ax is None:
		# Set appropriate figure size for large graphs
		fig, ax = plt.subplots(figsize=(10, 6))
	else:
		fig = ax.get_figure()
	# Use a layout that handles large graphs relatively well
	if pos is None:
		if spatial_coords is None:
			pos = nx.spring_layout(G, seed=random_state, k=0.15, iterations=200)
		else:
			pos = {i: (spatial_coords[i][0], spatial_coords[i][1]) for i in range(graph.shape[0])}
	# Create a color mapper for normalization
	unique_labels = list(set(node_labels))
	if node_labels is not None and label_to_color is None:
		cmap = plt.cm.tab20  
		norm = plt.Normalize(vmin=0, vmax=len(unique_labels) - 1) 
		sm = ScalarMappable(cmap=cmap, norm=norm)
		label_to_color = {label: sm.to_rgba(i) for i, label in enumerate(unique_labels)}

	if node_labels is not None:
		# Draw nodes with colors based on labels and colormap
		node_colors = [label_to_color[label] for label in node_labels]
		node_freq =  {}
		for label in node_labels:
			if label in node_freq:
				node_freq[label] += 1
			else:
				node_freq[label] = 1
		node_freq = {k: v*1.0/len(node_labels) for k,v in node_freq.items()}
		nx.draw_networkx_nodes(G, pos, node_size=5, node_color=node_colors, ax=ax)
	else:
		nx.draw_networkx_nodes(G, pos, node_size=5, ax=ax)

	nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.5, edge_color='gray', ax=ax)
	
	if node_labels is not None and add_legend:
		legend_handles = [patches.Patch(color=sm.to_rgba(i), label=label) for i, label in enumerate(unique_labels)]
		ax.legend(handles=legend_handles, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.15), borderaxespad=0.)
	ax.axis('off')
	if show:
		plt.show()
	return fig, ax, pos, label_to_color, node_freq
