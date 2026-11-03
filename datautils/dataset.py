import numpy as np
from .utils import load_cell_data, cellcell_to_features
import hashlib
import os
import pickle
from mainutils.utils import train_test_split, k_fold_split

class SpatialCellToFeatures:
	CACHE_VERSION = 'v2_qc_trainfit'
	"""
	This class loads and prepares spatial gene expression data for further analysis.

	Attributes:
		config (dict): Configuration dictionary containing data and processing parameters.
		expressions (np.ndarray): Array containing gene expression data.
		enrichments (np.ndarray, optional): Array containing enrichment features (if applicable). Defaults to None.
		graphs (list, optional): List of graphs representing spatial relationships (if applicable). Defaults to None.
		labels (list): List of ground truth labels.
		feature_names (list): List of feature names.
		label_vec (np.ndarray): One-hot encoded labels.
	"""
	def __init__(self, config, random_state=42):
		"""
		Initializes the SpatialCellToFeatures object.

		Args:
			config (dict): Configuration dictionary containing data and processing parameters.
		"""

		self.config = config
		self.seed = random_state
		self.unique_labels = {'pCR': 1, 'Responder': 1, 'Non-Responder': 0}
		#filename = f"{self.config['DATA_PATH']}/cellcell_processed_split.pkl"
		# Old file
		#filename = f"{self.config['DATA_PATH']}/RCB_cellcell_processed_{self.config['datasplit']}_{self.seed}.pkl"
		# New file
		#filename = f"{self.config['DATA_PATH']}/RCB_cellcell_processed_new_qcpass7_{self.config['datasplit']}_{self.seed}.pkl"

		split_detail = f"_test{self.config.get('test_ratio', 'na')}" if self.config['datasplit'] == 'split' else ''
		self.stratify_by_batch = bool(self.config.get('stratify_by_batch', False))
		self.drop_single_class_batches = bool(self.config.get('drop_single_class_batches', False))
		if self.stratify_by_batch:
			split_detail += '_batchstrat'
		if self.drop_single_class_batches:
			split_detail += '_purebatchdrop'
		filename = f"{self.base_cache_path(self.config)[:-4]}_{self.config['datasplit']}{split_detail}_{self.seed}.pkl"

		if os.path.exists(f"{filename}"):		
			self.data = self.load_split_data(filename)
		else:
			self.data = self.prepare_data(filename)
		

	@staticmethod
	def cohort_options(config):
		"""qc_pass switch and cohort exclusions, as read from the dataset config."""
		from patientgnn.config import load_exclusions
		leap_ids, fovs = load_exclusions(config)
		return bool(config.get('require_qc_pass', True)), leap_ids, fovs

	@classmethod
	def base_cache_path(cls, config):
		"""Seed-independent cache path. Non-default cohort options get their own
		suffix so a cache built with other settings is never silently reused."""
		require_qc, leap_ids, fovs = cls.cohort_options(config)
		suffix = ''
		if not require_qc:
			suffix += '_noqc'
		if leap_ids or fovs:
			digest = hashlib.sha1(repr((sorted(leap_ids), sorted(fovs))).encode()).hexdigest()[:8]
			suffix += f'_excl{digest}'
		return (f"{config['DATA_PATH']}/{config['cell_filename']}_{config['response_filename']}"
				f"_{cls.CACHE_VERSION}{suffix}.pkl")

	def load_data(self, filename):
		"""
		Loads pre-processed data from a pickle file.

		Args:
			filename (str): Path to the pickle file containing the data.

		Returns:
			dict: A dictionary containing: expressions, enrichments (default None), graphs (default None), labels, and feature names.
		"""
		print('Loading Expression Data From File')
		with open(filename, 'rb') as f:
			data = pickle.load(f)
			return data

	def load_split_data(self, filename):
		"""
		Loads pre-processed data from a pickle file.

		Args:
			filename (str): Path to the pickle file containing the data.

		Returns:
			dict: A dictionary containing: expressions, enrichments (default None), graphs (default None), labels, and feature names.
					train and test data accessed as: data['train'], data['test']
		"""
		print('Loading Expression Data From File')
		with open(filename, 'rb') as f:
			data = pickle.load(f)
			return data

	def filter_single_class_batches(self, data):
		"""
		Drops ROIs from stain batches that contain only one response class.
		"""
		stain = np.asarray(data['stain'])
		labels = np.asarray(data['labels'])
		keep_batches = [b for b in np.unique(stain)
						if len(np.unique(labels[stain == b])) > 1]
		mask = np.isin(stain, keep_batches)
		dropped = int((~mask).sum())
		print(f'Dropping {dropped} ROIs from single-class stain batches; '
			  f'keeping batches {sorted(int(b) for b in keep_batches)}')
		indices = np.flatnonzero(mask)
		filtered = {}
		for key, values in data.items():
			if values is None or key in ('markers', 'celltypes'):
				filtered[key] = values
			else:
				filtered[key] = [values[i] for i in indices]
		filtered['labels'] = np.asarray(filtered['labels'])
		return filtered

	def prepare_data(self, filename):
		"""
		Loads raw cell data, preprocesses it, and saves the processed data to a file.

		Args:
			filename (str): Path to the pickle file where the processed data will be saved.

		Returns:
			dict: A dictionary containing: expressions, enrichments (default None), graphs (default None), labels, and feature names.
		"""
		
		# Old Data
		#datafile = f"{self.config['DATA_PATH']}/RCB_processed_data_cellcell.pkl"
		# New Data
		#datafile = f"{self.config['DATA_PATH']}/RCB_processed_data_cellcell_new_qcpass7.pkl"

		datafile = self.base_cache_path(self.config)
		if os.path.exists(datafile):
			data = self.load_data(datafile)
		else:
			print('Loading Cell Table')
			require_qc, leap_ids, fovs = self.cohort_options(self.config)
			cell_table = load_cell_data(self.config['DATA_PATH'],
											self.config['cell_filename'],
											self.config['response_filename'],
											require_qc_pass=require_qc,
											exclude_leap_ids=leap_ids,
											exclude_fovs=fovs,
											)

			print('Preparing Expression Data From Cell Table and saving to disk')
			data = cellcell_to_features(cell_table, 
											filename=datafile)
		print('Split Expression Data and save to disk')
		data['labels'] = np.asarray([self.unique_labels[label] for label in data['labels']])
		if self.drop_single_class_batches:
			data = self.filter_single_class_batches(data)
		if self.config['datasplit'] == 'split':
			data_train, data_test = train_test_split(data,
													test_size=self.config['test_ratio'],
													random_state=self.seed,
													stratify_by_batch=self.stratify_by_batch)

			dataset = {
						'train': data_train,
						'test': data_test
			}

			with open(filename, 'wb') as f:
				pickle.dump(dataset, f)
			return dataset

		if self.config['datasplit'] == 'leaveOneOut':
			return data
		raise ValueError(
			f"Unsupported data split {self.config['datasplit']!r}; "
		)
