import numpy as np
import pandas as pd
import os
import pickle
from scipy.sparse import csr_matrix
import squidpy as sq
from collections import OrderedDict

from functools import partial
from tqdm import tqdm
from joblib import Parallel, delayed
import threading

##
# Based On Giuseppe's Code

ALLMARKERS = ['Alpha-SMA', 'B7-H4', 'Beta-Catenin', 'CD107a', 'CD11b', 'CD14', 'CD16', 
			'CD163', 'CD20', 'CD27', 'CD3', 'CD31', 'CD366', 'CD38', 'CD4', 'CD44', 
			'CD45', 'CD45RO', 'CD68', 'CD8a', 'Carboplatin', 'Collage-Type_I', 
			'DNA1', 'DNA2', 'E-Cadherin', 'EGFR', 'FOXP3', 'Granzyme-B', 'HLA-DR-DQ-DP', 
			'Ki-67', 'PD-1', 'PD-L1', 'PD-L2', 'Pan-keratin', 'Tbet', 'VEGF', 
			'Vimentin', 'p53']

EXCLUDE_MARKERS = ['Carboplatin']

MARKERS = list(filter(lambda x: x not in EXCLUDE_MARKERS, ALLMARKERS))

def quality_control(data, low_gene_active=0.2, high_gene_active=0.5, dna_quantile=0.05):
	"""
	Performs quality control filtering on intensity data, ensuring efficiency and readability.

	Args:
		data (pd.DataFrame): DataFrame containing intensity data.
		low_gene_active (float, optional): Threshold for minimum active genes per cell (default: 0.2).
		high_gene_active (float, optional): Threshold for maximum active genes per cell (default: 0.5).
		dna_quantile (float, optional): Quantile for DNA content filtering (default: 0.05).

	Returns:
		pd.Series: Boolean Series indicating cells that pass quality control.
	"""

	if 'pass_qc' in data.columns:
		return data['pass_qc']

	# Create efficient boolean masks for filtering
	is_marker = data.columns.isin(MARKERS)

	# Perform filtering and quantile calculation efficiently using broadcasting
	active_genes_few = ((data.loc[:,is_marker]>low_gene_active).sum(axis=1)>0)
	active_genes_many = ((data.loc[:,is_marker]>high_gene_active).sum(axis=1)<11)
	dna_thr = np.quantile(data[['DNA1', 'DNA2']].sum(axis=1), dna_quantile)
	passed_qc = active_genes_few & active_genes_many & (data[['DNA1', 'DNA2']].sum(axis=1) > dna_thr)
	return passed_qc

def load_cell_data(datapath, filename_celldata, filename_biosamples):
	"""
	Loads cell data from CSV files and performs quality control if needed.

	Args:
		datapath (str): Path to the data directory.
		filename_celldata (str): Name of the cell data CSV file.
		filename_biosamples (str): Name of the biosamples CSV file.

	Returns:
		pd.DataFrame: A combined dataframe containing cell table and biosamples DataFrames.
	"""
	cell_table = pd.read_csv(f'{datapath}/{filename_celldata}.csv', sep=',')
	if 'qc_pass' not in cell_table.columns:
		qc_pass = quality_control(cell_data)
		cell_table['qc_pass'] = qc_pass
		cell_table.to_csv(f'{datapath}/{filename_celldata}.csv', index=False)
	biosamples = pd.read_csv(f'{datapath}/{filename_biosamples}.csv', sep=',')
	biosamples.drop(['FORCE_TRIAL?_(Y/N)'],axis = 1,inplace = True)        

	if 'cell_meta_cluster' in cell_table:
		cell_table = cell_table[cell_table['cell_meta_cluster']!='Unassigned']

	cell_table['LEAP_ID'] = cell_table.fov.str.split('_', n=1).str[0].str.upper()
	cell_table['LEAP_ID'] = cell_table.LEAP_ID.str[:7]#leap_ID should be Leap123, anything more is stripped

	cell_table = cell_table.reset_index().merge(biosamples, left_on='LEAP_ID', right_on= 'LEAP_ID').drop(['LEAP_ID'], axis = 1).set_index('index')

	# get fovs having more than 1000 cells
	#fovs = cell_table.fov.value_counts()[cell_table.fov.value_counts()>=1000].index
	cell_table = cell_table[cell_table.fov.isin(fovs)]
	cell_table[MARKERS] = cell_table[MARKERS].fillna(0)
	cell_table = cell_table.dropna(subset=['Stain'])
	cell_table = filter_data(cell_table, use_core=True)
	cell_table.dropna(subset=['NACT_treatment _group'], inplace=True)
	return cell_table


def filter_data(cell_table, qc_pass=False, use_core=True):
	"""
	Filters the pandas based on user-defined criteria.

	Args:
		cell_table (pd.DataFrame): The pandas object containing the data.
		qc_pass (bool, optional): If True, filter to include only high-quality cells based on the 'qc_pass' label. Defaults to False.
		use_core (bool, optional): If True, filter to include only core biopsies based on the 'SAMPLE_TYPE_(CORE/RESECTION)' label. Defaults to True.

	Returns:
		pd.DataFrame: The filtered pd frame.
	"""

	if use_core:
		cell_table = cell_table[cell_table['SAMPLE_TYPE_(CORE/RESECTION)']=='CORE']
	if qc_pass:
		cell_table = cell_table[cell_table['qc_pass']]
	return cell_table

# from memory_profiler import profile

# @profile
def process_roi(roi, cell_table, min_cells):
	roi_cells = cell_table[cell_table.fov == roi]
	if len(roi_cells) < min_cells:
		return None

	coords = roi_cells[['centroid-0', 'centroid-1']].values
	expressions = roi_cells[MARKERS].values
	cell_labels = roi_cells.cell_meta_cluster.values
	label = set(roi_cells.Response.values)
	if len(label) != 1:
		raise ValueError(f"Acquisition {roi} has non-unique labels")
	label = label.pop()
	patient = roi_cells.Patient.iloc[0]
	stain = int(roi_cells.Stain.iloc[0]) - 1 #Offset by 1 for labels
	return expressions, coords, label, patient, cell_labels, stain, roi


def cellcell_to_features(cell_table, min_cells=10, filename='./data.pkl', num_cores=4):
	"""
	Extracts features based on interactions between individual cells within each acquisition.

	Args:
		adata (anndata.AnnData): AnnData object containing spatial transcriptomics data.
		min_cells (int, optional): Minimum number of cells required in an acquisition for processing. Defaults to 10.
		gmethod (str, optional): Method for constructing the adjacency matrix (e.g., 'knn'). Defaults to 'knn'.
		k (int, optional): Number of nearest neighbors for the kNN method. Defaults to 7.
		filename (str, optional): Filename to save processed data. Defaults to './data.pkl'.

	Returns:
		dict: A dictionary containing expressions, graphs, labels, markers and patientid.
	"""
	unique_rois = cell_table.fov.unique()
	#unique_leaps = set(roi.split('_')[0] for roi in unique_rois)
	#leap_to_patient = {leap: f"patient_{i+1}" for i, leap in enumerate(unique_leaps)}
	celltypes = np.array(cell_table.cell_meta_cluster.unique())

	dataset = {
		'expressions': [],
		'coords': [],
		'labels': [],
		'patient': [],
		'cell_labels': [],
		'stain': [],
		'markers': MARKERS,
		'celltypes': celltypes,
		'leapid': []
		}
	semaphore = threading.Semaphore()
	with semaphore:
		results = Parallel(n_jobs=num_cores, timeout=120)(
				delayed(process_roi)(roi, cell_table, min_cells)
				for roi in tqdm(unique_rois, desc="Processing ROIs")
		)

	for result in results:
		if result is not None:
			expressions, coords, label, patient, cell_labels, stain, leapid = result
			dataset['expressions'].append(expressions)
			dataset['coords'].append(coords)
			dataset['labels'].append(label)
			dataset['patient'].append(patient)
			dataset['cell_labels'].append(cell_labels)
			dataset['stain'].append(stain)
			dataset['leapid'].append(leapid)

	with open(filename, 'wb') as f:
		pickle.dump(dataset, f)
	return dataset
