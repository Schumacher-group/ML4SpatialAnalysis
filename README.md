# ML4SpatialAnalysis: A Machine Learning Framework to do Spatial Analysis of Single Cell TNBC Data

This repository implements machine learning (ML) models in the context of spatial single-cell data analysis. It streamlines the entire process, from data preparation and model training to evaluation and interpretability.

## Citation
This pipeline was used in the following preprint:

**"Identifying tissue states by spatial protein patterns related to chemotherapy response in triple-negative breast cancer"**
bioRxiv (2025). DOI: [10.1101/2025.10.06.680783](https://www.biorxiv.org/content/10.1101/2025.10.06.680783)

If you use this pipeline in your work, please consider citing the preprint. 

## Preceding analysis
### Data Preprocessing and Segmentation
Data preprocessing and segmentation are performed using the separate [IMC_preprocessing](https://github.com/Schumacher-group/IMC_preprocessing) pipeline.
### Cell phenotyping and spatial analysis
This repository handles phenotyping and other analysis shown in the above manuscript: https://github.com/Schumacher-group/IMC_TNBC_analysis

The output of the preprocessing pipeline (processed images and cell tables) serves as input for the phenotyping and analysis, and the output of the phenotyping (annotated cell table) serves as the input for the GNN prediction in this repository.

## Getting Started

1. **Create a Conda Environment:**

```bash
conda env create -f requirements.yaml
```

## Running the Code

Use the conda environment from `requirements.yaml` (python 3.8, torch 1.12, torch_geometric 2.6). ROI-level GNN or classical model, one run per config:

```bash
python main.py --config configs/config.yaml
```

`main.py` takes only `--config`; to sweep settings write one yaml per cell with `scripts/make_variant_configs.py` (gconv x gmethod x k x node_f x seed, plus a manifest for `scripts/run_manifest_cpu.sbatch`) and summarise the metric sinks with `scripts/collect_variant_results.py`.

Patient-level benchmark (package `patientgnn/`, config `configs/patient_benchmark.yaml`):

```bash
# Build the versioned QC cache and run all patient-level comparisons.
python -m patientgnn.run_patient_benchmark \
  --config configs/patient_benchmark.yaml \
  --build-cache

bash scripts/submit_regression.sh   # slurm: cache -> classical baselines -> DeepSets/GNN
```

Compare `expressions`, `celltypes`, and `combined` inputs for:

- Regularized logistic regression on patient-level ROI summaries.
- Random forest and XGBoost on exactly the same patient-level summaries and
  outer folds, matching the classical models used in the workshop analysis.
- A spatial logistic baseline with abundance-normalized cell-type contacts.
- Hierarchical DeepSets (same encoders and pooling as the GNN, no edges).
- A two-layer residual GINE using physical contact distances.
- An edge-shuffled negative control for the combined GNN.

## Data

`dataset.DATA_PATH` holds `<cell_filename>.csv` (one row per cell: `fov`, `cell_meta_cluster`, `centroid-0`, `centroid-1`, the marker columns, optionally `qc_pass`) and `<response_filename>.csv`. The loader expects the `CleanCohort_Metadata` schema: `LEAP_ID`, `Sample_Type_(pre/post treatment)`, `NACT_Treatment _Group`, `Response`, `Patient_ID`, `Stain_Batch`. The older `processed_response_RCB` file is not supported. The manuscript cohort has 63 patients (37 responders / 26 non-responders). Patient metadata is not tracked in this repository; place the file in `DATA_PATH` and pin the version you mean in `response_filename`, as later revisions have different labels.

## Attribution Analysis

Evaluate and attribute a saved run. `main.py` stamps run names with `time.time()`; the newest matching model is used unless `--run <timestamp>` is given:

```bash
python run_evaluation.py --config configs/config.yaml [--run 1753249753.4735453]
```

## Code Repository Structure

1. **data/**: Contains all datasets within a folder.
2. **datautils/**: Handles preprocessing of cell table and graph computations:
    - `dataset.py`: Implements the dataset class.
    - `utils.py`: Contains basic utilities required for dataset construction.

3. **logmodels/**: Stores pretrained models.

4. **models/**: Contains all ML models and their training:
    - `graph_networks.py`, `gcn.py`: GNN architectures and their training wrapper.
    - `trainer.py`: Defines a class for feature preparation, training, and evaluation.
    - `evaluation.py`: Analyzes weights/features for interpretability.

5. **mainutils/**: Graph construction, splits, metrics (`utils.py`) and train-fit marker normalisation (`normalization.py`).

6. **patientgnn/**: Patient-level benchmark (cohort cache, classical baselines, DeepSets/GINE).

7. **main.py**: Used for dispatching experiments.

8. **run_evaluation.py**: Dispatches attribution methods on a saved run.
```
