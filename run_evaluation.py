import argparse
from mainutils.utils import load_config
from models.evaluation import ModelEvaluation
from datautils.dataset import SpatialCellToFeatures
import wandb
import os
import time
def run(config):
	"""
	Performs model attribution and logs results to Weights & Biases.

	Args:
		config (dict): Configuration dictionary containing model and dataset settings.
	"""

	if config['model']['name'] == 'gnn':
		config['model']['gcriterion'] = 'gnn'

	node_feature_mode = config['model'].get(config['model']['name'], {}).get('node_f', 'expressions')
	logname = f"model_{config['model']['name']}"\
				f"_graphtype_{config['model']['gnn']['gmethod']}"\
				f"_gmode_{config['model']['gnn']['gmode']}"\
				f"_fnorm_{config['model']['fnorm']}"\
				f"_graphfeats_{config['model']['gcriterion']}"\
				f"_node_f_{node_feature_mode}"\
				f"_eval_{config['dataset']['datasplit']}"\
				f"_seed_{config['seed']}_balanced_train_{config['balanced_train']}_1753249753.4735453"
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
    args = parser.parse_args()
    config = load_config(args.config)
    run(config)
