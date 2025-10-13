import argparse
from mainutils.utils import load_config
from models.evaluation import ModelEvaluation
from datautils.dataset import SpatialCellToFeatures
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def run(config):
	"""
	Performs model attribution and logs results to Weights & Biases.

	Args:
		config (dict): Configuration dictionary containing model and dataset settings.
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

	print('Configuring models')
	explainer = ModelEvaluation(config['model'], logname, logger)
	explainer.run(dataset.data, logname)
	wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
    args = parser.parse_args()
    config = load_config(args.config)
    run(config)