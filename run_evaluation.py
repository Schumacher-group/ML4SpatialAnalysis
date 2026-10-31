import argparse
import glob
from mainutils.utils import load_config
from models.evaluation import ModelEvaluation
from datautils.dataset import SpatialCellToFeatures
import wandb
import os
import time


def find_run(config, run_prefix, run_id=None):
	"""
	Resolves the logname of the training run to evaluate. main.py stamps every run
	name with time.time(), so without an explicit id the newest saved model wins.
	"""
	if run_id is not None:
		return f"{run_prefix}_{run_id}"
	log_path = config['model']['LOG_PATH']
	if config['dataset']['datasplit'] == 'leaveOneOut':
		pattern = f"{log_path}/{run_prefix}_*_leaveoneout_1_patient_*.pkl"
	else:
		pattern = f"{log_path}/{config['model']['name']}_{run_prefix}_*.pkl"
	saved = sorted(glob.glob(pattern), key=os.path.getmtime)
	if not saved:
		raise FileNotFoundError(f"no saved model matches {pattern}; pass --run <timestamp>")
	logname = os.path.basename(saved[-1])[:-len('.pkl')].split('_leaveoneout_')[0]
	prefix = f"{config['model']['name']}_"
	return logname[len(prefix):] if logname.startswith(prefix) else logname


def run(config, run_id=None):
	"""
	Performs model attribution and logs results to Weights & Biases.

	Args:
		config (dict): Configuration dictionary containing model and dataset settings.
		run_id (str): time.time() stamp of the run to evaluate, newest run if None.
	"""

	if config['model']['name'] == 'gnn':
		config['model']['gcriterion'] = 'gnn'

	node_feature_mode = config['model'].get(config['model']['name'], {}).get('node_f', 'expressions')
	run_prefix = f"model_{config['model']['name']}"\
				f"_graphtype_{config['model']['gnn']['gmethod']}"\
				f"_gmode_{config['model']['gnn']['gmode']}"\
				f"_fnorm_{config['model']['fnorm']}"\
				f"_graphfeats_{config['model']['gcriterion']}"\
				f"_node_f_{node_feature_mode}"\
				f"_eval_{config['dataset']['datasplit']}"\
				f"_seed_{config['seed']}_balanced_train_{config['balanced_train']}"
	logname = find_run(config, run_prefix, run_id)
	print(f"Evaluating run {logname}")
	os.environ['WANDB_DIR'] = config['model']['LOG_PATH']
	# Initialize W&B logger with project name, entity, configuration, and log name
	logger = wandb.init(entity=None, project="Delta Tissue", config=config, name=logname)
	print('Preparing Features')
	dataset = SpatialCellToFeatures(config['dataset'], random_state=config['seed'])

	feature_data = dataset.data if config['dataset']['datasplit'] == 'leaveOneOut' else dataset.data['train']
	if node_feature_mode == 'expressions':
		config['model']['feature_dim'] = len(feature_data['markers'])
	elif node_feature_mode == 'celltypes':
		config['model']['feature_dim'] = len(feature_data['celltypes'])
	elif node_feature_mode == 'expressions_celltypes':
		config['model']['feature_dim'] = len(feature_data['markers']) + len(feature_data['celltypes'])
	else:
		raise ValueError(f"Invalid node feature type {node_feature_mode} specified in the configuration.")

	config['model']['eval'] = config['dataset']['datasplit']

	print('Feature Class Labels')
	print(dataset.unique_labels)

	print('Configuring models')
	explainer = ModelEvaluation(config['model'], logger)
	explainer.run(dataset.data, logname)
	wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
    parser.add_argument('--run', type=str, default=None, help='time.time() stamp of the run to evaluate; default is the newest saved model')
    args = parser.parse_args()
    config = load_config(args.config)
    run(config, args.run)
