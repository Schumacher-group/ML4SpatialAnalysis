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

To run the code with specified hyperparameters in a config file:

```bash
python main.py configs/config.yaml
```

```bash
# Build the versioned QC cache and run all patient-level comparisons.
python run_patient_benchmark.py \
  --config configs/patient_benchmark.yaml \
  --build-cache

bash scripts/submit_regression.sh
```

Compare `expressions`, `celltypes`, and `combined` inputs for:

- Regularized logistic regression on patient-level ROI summaries.
- Random forest and XGBoost on exactly the same patient-level summaries and
  outer folds, matching the classical models used in the workshop analysis.
- A spatial logistic baseline with abundance-normalized cell-type contacts.
- Hierarchical DeepSets (same encoders and pooling as the GNN, no edges).
- A two-layer residual GINE using physical contact distances.
- An edge-shuffled negative control for the combined GNN.

## Attribution Analysis

Attribution analysis of a pretrained model:

```bash
python run_attribution.py configs/config.yaml
```

## Code Repository Structure

1. **data/**: Contains all datasets within a folder.
2. **datautils/**: Handles preprocessing of cell table and graph computations:
    - `dataset.py`: Implements the dataset class.
    - `utils.py`: Contains basic utilities required for dataset construction.

3. **logmodels/**: Stores pretrained models.

4. **models/**: Contains all ML models and their training:
    - `factory.py`: Implements ML models.
    - `trainer.py`: Defines a class for feature preparation, training, and evaluation.
    - `attribution.py`: Analyzes weights/features for interpretability.

5. **utils/**: Holds common functions used in different parts of the code:
    - `utils.py`

6. **main.py**: Used for dispatching experiments.

    ./response_prediction.sh config/config.yaml model.name=xgboost model.fnorm=log1p

7. **run_attribution.py**: Dispatches attribution methods.
```
