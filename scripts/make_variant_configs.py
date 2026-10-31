"""Generate one config per cell of a GNN comparison grid.

Cells are gconv x gmethod x k x node_f x seed; anything not listed on the
command line is left as in the base config. hiergcn is deliberately absent:
it needs DynamicEdgeConv from torch-cluster, which is not installed in the
ML4SA env, so it cannot be launched.
"""

import argparse
import copy
from itertools import product
from pathlib import Path

import yaml

GCONVS = ['gcn', 'ssgcn', 'edgegcn', 'attngcn']
SEEDS = [42, 43, 44]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--outdir', default='logs/variants')
    parser.add_argument('--gconvs', nargs='+', default=GCONVS)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--gmethods', nargs='+', default=['atmostk'])
    parser.add_argument('--ks', type=int, nargs='+', default=None,
                        help='neighbourhood sizes for knn/atmostk; default keeps the config k')
    parser.add_argument('--node-fs', nargs='+', default=None,
                        help='node feature sets; default keeps the config node_f')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', default=None,
                        help="override model.gnn.device, e.g. 'cpu'")
    parser.add_argument('--log-path', default=None,
                        help='override model.LOG_PATH to isolate a campaign')
    parser.add_argument('--val-ratio', type=float, default=None,
                        help='carve a patient-grouped validation split out of train')
    parser.add_argument('--epoch-curve', action='store_true',
                        help='evaluate val and test every epoch (diagnostic trajectory)')
    parser.add_argument('--manifest', default=None,
                        help='manifest filename inside outdir (default manifest.txt)')
    args = parser.parse_args()

    with open(args.config) as handle:
        base = yaml.safe_load(handle)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []

    grid = product(args.gconvs, args.gmethods, args.ks or [None], args.node_fs or [None], args.seeds)
    for gconv, gmethod, k, node_f, seed in grid:
        cfg = copy.deepcopy(base)
        cfg['seed'] = seed
        cfg['model']['nm_epochs'] = args.epochs
        gnn = cfg['model']['gnn']
        gnn['gconv'] = gconv
        gnn['gmethod'] = gmethod
        if k is not None:
            gnn['k'] = k
        if node_f is not None:
            gnn['node_f'] = node_f
        if args.device:
            gnn['device'] = args.device
        if args.log_path:
            cfg['model']['LOG_PATH'] = args.log_path
        if args.val_ratio is not None:
            cfg['model']['val_ratio'] = args.val_ratio
        if args.epoch_curve:
            cfg['model']['epoch_curve'] = True
        tag = f"{gconv}_{gmethod}_k{gnn['k']}_{gnn['node_f']}_s{seed}"
        path = outdir / f'cfg_{tag}.yaml'
        with path.open('w') as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        manifest.append(str(path))

    listing = outdir / (args.manifest or 'manifest.txt')
    listing.write_text('\n'.join(manifest) + '\n')
    print(f'wrote {len(manifest)} configs to {outdir}')
    print(f'manifest: {listing}')
    for line in manifest:
        print(' ', line)


if __name__ == '__main__':
    main()
