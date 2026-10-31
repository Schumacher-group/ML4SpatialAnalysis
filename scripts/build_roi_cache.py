"""Build the ROI-level SpatialCellToFeatures cache for the main.py pipeline.

Every main.py run rebuilds this cache from the 4.2M-row cell table if it is
missing. Launching a grid of jobs with no cache present would have each of them
rebuild it concurrently and race on the same pickle path, so build it once here
and let the grid start against a warm cache.

The seed-independent base pickle is built first and then reused by every split.
"""

import argparse

from mainutils.utils import load_config
from datautils.dataset import SpatialCellToFeatures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = parser.parse_args()

    config = load_config(args.config)
    for seed in args.seeds:
        print(f'=== building cache for seed {seed} ===', flush=True)
        dataset = SpatialCellToFeatures(config['dataset'], random_state=seed)
        data = dataset.data
        train = data['train'] if 'train' in data else data
        print(f"seed {seed}: {len(train['labels'])} train ROIs, "
              f"{len(data['test']['labels']) if 'test' in data else 0} test ROIs, "
              f"{len(train['markers'])} markers, {len(train['celltypes'])} celltypes",
              flush=True)


if __name__ == '__main__':
    main()
