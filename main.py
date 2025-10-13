import argparse
import pandas as pd
import numpy as np
import wandb
from mainutils.utils import load_config
from models.trainer import ModelTrainer
from datautils.dataset import SpatialCellToFeatures
from sklearn.utils.class_weight import compute_class_weight
import os

def log_features(features, labels, logger):
	"""
	Logs features and labels as a table in W&B for better visualization.

	Args:
		features (np.array): Array of features.
		labels (np.array): Array of labels.
		logger (wandb.Logger): W&B logger object.
	"""
	columns = [f"Feature_{i}" for i in range(features.shape[1])] + ["Label"]
	df = pd.DataFrame(np.column_stack((features, labels)), columns=columns)
	logger.log({"Feature_Matrix": wandb.Table(dataframe=df)})


def run(config):
	"""
	Runs the entire training process based on the provided configuration.

	Args:
		config (dict): Configuration dictionary containing training parameters.
	"""

	if config['model']['name'] == 'gnn':
		config['model']['gcriterion'] = 'gcn'

	logname = f"model_{config['model']['name']}"\
				f"_graphtype_{config['model']['gnn']['gmethod']}"\
				f"_gmode_{config['model']['gnn']['gmode']}"\
				f"_fnorm_{config['model']['fnorm']}"\
				f"_graphfeats_{config['model']['gcriterion']}"\
				f"_eval_{config['dataset']['datasplit']}"\
				f"_seed_{config['seed']}_balanced_train_{config['balanced_train']}"
	os.environ['WANDB_DIR'] = config['model']['LOG_PATH']
	# Initialize W&B logger with project name, entity, configuration, and log name
	logger = wandb.init(entity=None, project="ML on TNBC Data", config=config, name=logname)
	print('Preparing Features')
	dataset = SpatialCellToFeatures(config['dataset'], random_state=config['seed'])
	if config['dataset']['datasplit'] == 'leaveOneOut':
		config['model']['feature_dim'] = len(dataset.data['markers'])
	else:
		config['model']['feature_dim'] = len(dataset.data['train']['markers'])
	config['model']['eval'] = config['dataset']['datasplit']

	print('Feature Class Labels')
	print(dataset.unique_labels)

	if config['balanced_train'] and config['dataset']['datasplit']=='split':
		class_weight = compute_class_weight('balanced', classes=np.unique(dataset.data['train']['labels']), y=dataset.data['train']['labels'])
		class_weight = dict(zip(np.unique(dataset.data['train']['labels']), class_weight))
	else:
		class_weight = None
	print('Configuring models')
	# Configure the model trainer
	model = ModelTrainer(config['model'],
							class_weight=class_weight, 
							logger=logger,
							logfile=logname,
							seed=config['seed'])
	
	# Train the model and get evaluation metrics
	model.optimise(dataset.data, logname)
	wandb.finish()


# Entry point for the script, parses arguments and loads configuration
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
	args = parser.parse_args()
	config_file = load_config(args.config)
	run(config_file)