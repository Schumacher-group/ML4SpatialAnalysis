import numpy as np
from .utils import load_cell_data, cellcell_to_features
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
		filename = f"{self.config['DATA_PATH']}/{self.config['cell_filename']}_{self.config['response_filename']}_{self.CACHE_VERSION}_{self.config['datasplit']}{split_detail}_{self.seed}.pkl"

		if os.path.exists(f"{filename}"):		
			self.data = self.load_split_data(filename)
		else:
			self.data = self.prepare_data(filename)
		

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

		datafile = f"{self.config['DATA_PATH']}/{self.config['cell_filename']}_{self.config['response_filename']}_{self.CACHE_VERSION}.pkl"
		if os.path.exists(datafile):
			data = self.load_data(datafile)
		else:
			print('Loading Cell Table')
			cell_table = load_cell_data(self.config['DATA_PATH'],
											self.config['cell_filename'], 
											self.config['response_filename']
											)

			print('Preparing Expression Data From Cell Table and saving to disk')
			data = cellcell_to_features(cell_table, 
											filename=datafile)
		print('Split Expression Data and save to disk')
		data['labels'] = np.asarray([self.unique_labels[label] for label in data['labels']])
		if self.config['datasplit'] == 'split':
			data_train, data_test = train_test_split(data, 
													test_size=self.config['test_ratio'], 
													random_state=self.seed)

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
