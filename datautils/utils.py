import numpy as np
import pandas as pd
import os
import pickle
from scipy.sparse import csr_matrix
from collections import OrderedDict

from functools import partial
from tqdm import tqdm
from joblib import Parallel, delayed


ALLMARKERS = ['Alpha-SMA', 'B7-H4', 'Beta-Catenin', 'CD107a', 'CD11b', 'CD14', 'CD16',
            'CD163', 'CD20', 'CD27', 'CD3', 'CD31', 'CD366', 'CD38', 'CD4', 'CD44',
            'CD45', 'CD45RO', 'CD68', 'CD8a', 'Carboplatin', 'Collage-Type_I',
            'DNA1', 'DNA2', 'E-Cadherin', 'EGFR', 'FOXP3', 'Granzyme-B', 'HLA-DR-DQ-DP',
            'Ki-67', 'PD-1', 'PD-L1', 'PD-L2', 'Pan-keratin', 'Tbet', 'VEGF',
            'Vimentin', 'p53']

EXCLUDE_MARKERS = ['CD27', 'CD38', 'Tbet', 'p53', 'EGFR', 'VEGF', 'PD-1', 'PD-L1', 'PD-L2', 'Carboplatin']

MARKERS = list(filter(lambda x: x not in EXCLUDE_MARKERS, ALLMARKERS))

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
    biosamples = pd.read_csv(f'{datapath}/{filename_biosamples}.csv', sep=',')

    if 'cell_meta_cluster' in cell_table:
        cell_table = cell_table[cell_table['cell_meta_cluster']!='Unassigned']
    if 'qc_pass' in cell_table:
        qc_pass = cell_table['qc_pass'].astype(str).str.strip().str.lower().isin(['true', '1', 'yes', 'y'])
        cell_table = cell_table[qc_pass]

    cell_table['LEAP_ID'] = cell_table.fov.str.split('_', n=1).str[0].str.upper()
    cell_table['LEAP_ID'] = cell_table.LEAP_ID.str[:7]  # leap_ID should be Leap123, anything more is stripped

    cell_table = cell_table.reset_index().merge(biosamples, left_on='LEAP_ID', right_on='LEAP_ID').drop(['LEAP_ID'], axis=1).set_index('index')

    cell_table[MARKERS] = cell_table[MARKERS].fillna(0)
    cell_table = cell_table[cell_table['Sample_Type_(pre/post treatment)'] == 'pre']
    cell_table.dropna(subset=['NACT_Treatment _Group'], inplace=True)
    return cell_table

def process_roi(roi, cell_table, min_cells):
    roi_cells = cell_table[cell_table.fov == roi]
    if len(roi_cells) < min_cells:
        return None
    coords = roi_cells[['centroid-0', 'centroid-1']].values
    expressions = roi_cells[MARKERS].values
    cell_labels = roi_cells.cell_meta_cluster.values
    label = roi_cells.Response.unique()
    if len(label) != 1:
        raise ValueError(f"Acquisition {roi} has non-unisque labels")
    label = label[0]
    patient = roi_cells.Patient_ID.iloc[0]
    stain = int(roi_cells.Stain_Batch.iloc[0])
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

    # Filter out ROIs with fewer cells before parallel processing
    valid_rois = [roi for roi in unique_rois if len(cell_table[cell_table.fov == roi]) >= min_cells]

    results = Parallel(n_jobs=num_cores, timeout=120)(
        delayed(process_roi)(roi, cell_table, min_cells)
        for roi in tqdm(valid_rois, desc="Processing ROIs")
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
