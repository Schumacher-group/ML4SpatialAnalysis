"""Nested-CV architecture/hyperparameter search for the ROI-level GNN.

Why nested. With ~50 patients a random classifier scores AUC 0.50 +/- 0.14, so
evaluating a grid of N configurations and reporting the best one manufactures a
result near 0.75 from noise alone. Here the configuration is chosen inside each
outer fold, using only that fold's training patients, and the reported score comes
from outer-fold patients the selection never saw. The number this prints is
therefore an estimate of "pick a config this way, then apply it", which is the
thing you would actually deploy.

Controls applied to the data before any modelling:
  - single-class stain batches dropped (batch identity alone scored AUC 0.66)
  - exactly N ROIs sampled per patient (ROI count alone scored AUC 0.74)
Both confounders are re-scored on the same folds so the GNN has an honest bar.
"""

import argparse
import collections
import copy
import json
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from mainutils.utils import load_config

BASE_CACHE = ('/n/data1/hms/cellbio/sander/asif/KCLData/data/newdata3/'
              'updated_cell_table_CleanCohort_Metadata_v2_qc_trainfit.pkl')
LABELS = {'pCR': 1, 'Responder': 1, 'Non-Responder': 0}

# Levers with a mechanism behind them, not an exhaustive sweep: depth (a 1-layer
# GCN is provably the ROI mean), edge geometry (connectivity discards all
# distances), class weighting (73% positive ROIs collapse the model to majority),
# and training length (test AUC decays monotonically past ~epoch 2-4).
GRID = [
    dict(gconv='deepgcn', num_layers=3, hidden_dim=128, gmode='connectivity', balanced=True,  lr=1e-3, epochs=5),
    dict(gconv='deepgcn', num_layers=3, hidden_dim=128, gmode='distance',     balanced=True,  lr=1e-3, epochs=5),
    dict(gconv='deepgcn', num_layers=2, hidden_dim=64,  gmode='connectivity', balanced=True,  lr=1e-3, epochs=3),
    dict(gconv='deepgcn', num_layers=4, hidden_dim=128, gmode='distance',     balanced=True,  lr=3e-4, epochs=8),
    dict(gconv='deepgcn', num_layers=3, hidden_dim=64,  gmode='connectivity', balanced=False, lr=1e-3, epochs=3),
    dict(gconv='gcn',     num_layers=1, hidden_dim=256, gmode='connectivity', balanced=True,  lr=1e-3, epochs=3),
    # Indices 6+ are for augmented runs: a fresh ROI draw and cell subsample each
    # epoch slows memorisation, so longer training is worth trying there.
    dict(gconv='deepgcn', num_layers=3, hidden_dim=128, gmode='connectivity', balanced=True,  lr=1e-3, epochs=20),
    dict(gconv='deepgcn', num_layers=3, hidden_dim=128, gmode='distance',     balanced=True,  lr=1e-3, epochs=20),
    dict(gconv='deepgcn', num_layers=4, hidden_dim=128, gmode='distance',     balanced=True,  lr=3e-4, epochs=30),
    # Indices 9-11: epoch curve under augmentation. Augmentation at 5 epochs beat
    # no-augmentation at 5 epochs (0.523 vs 0.446) while 20 epochs collapsed to
    # 0.287, so the optimum is short. These bracket it; all points get reported.
    dict(gconv='deepgcn', num_layers=3, hidden_dim=128, gmode='distance', balanced=True, lr=1e-3, epochs=2),
    dict(gconv='deepgcn', num_layers=3, hidden_dim=128, gmode='distance', balanced=True, lr=1e-3, epochs=3),
    dict(gconv='deepgcn', num_layers=3, hidden_dim=128, gmode='distance', balanced=True, lr=1e-3, epochs=8),
    # Index 12: one epoch. The augmented curve decays monotonically with training
    # length (2ep 0.592 -> 20ep 0.296), so the endpoint tests whether ANY training
    # helps. If 1 epoch beats 2, the optimiser is moving away from generalisation.
    dict(gconv='deepgcn', num_layers=3, hidden_dim=128, gmode='distance', balanced=True, lr=1e-3, epochs=1),
    # 13-16: loss x schedule for the full-data configuration (all patients, all ROIs
    # at evaluation, balanced draw per epoch in training).
    dict(gconv='deepgcn', num_layers=3, hidden_dim=128, gmode='distance', balanced=True, lr=1e-3, epochs=2, loss='bce'),
    dict(gconv='deepgcn', num_layers=3, hidden_dim=128, gmode='distance', balanced=True, lr=1e-3, epochs=2, loss='focal', focal_gamma=2.0),
    dict(gconv='deepgcn', num_layers=3, hidden_dim=128, gmode='distance', balanced=True, lr=1e-3, epochs=5, loss='bce'),
    dict(gconv='deepgcn', num_layers=3, hidden_dim=128, gmode='distance', balanced=True, lr=1e-3, epochs=5, loss='focal', focal_gamma=2.0),
    # 17-24: architecture, cell-type encoding and regularisation. Balanced
    # node_f_scale equalises the marker block (93.6% of input variance) against
    # the cell-type one-hot (6.4%), so cell identity reaches the first layer.
    dict(gconv='deepgine', num_layers=3, hidden_dim=128, gmode='distance', balanced=True, lr=1e-3, epochs=2, loss='focal'),
    dict(gconv='deepgine', num_layers=3, hidden_dim=128, gmode='distance', balanced=True, lr=1e-3, epochs=2, loss='focal', node_f_scale='balanced'),
    dict(gconv='deepgcn',  num_layers=3, hidden_dim=128, gmode='distance', balanced=True, lr=1e-3, epochs=2, loss='focal', node_f_scale='balanced'),
    dict(gconv='deepgcn',  num_layers=3, hidden_dim=128, gmode='distance', balanced=True, lr=1e-3, epochs=2, loss='focal', node_f_scale='balanced', drop_p=0.5, weight_decay=1e-3),
    dict(gconv='deepgine', num_layers=4, hidden_dim=128, gmode='distance', balanced=True, lr=3e-4, epochs=4, loss='focal', node_f_scale='balanced'),
    dict(gconv='attngcn',  num_layers=1, hidden_dim=64,  gmode='distance', balanced=True, lr=1e-3, epochs=2, loss='focal', node_f_scale='balanced'),
    dict(gconv='edgegcn',  num_layers=1, hidden_dim=64,  gmode='distance', balanced=True, lr=1e-3, epochs=2, loss='focal', node_f_scale='balanced'),
    dict(gconv='deepgcn',  num_layers=3, hidden_dim=128, gmode='connectivity', balanced=True, lr=1e-3, epochs=2, loss='focal', node_f_scale='balanced'),
]


def fit_batch_correction(data, indices):
    """Per-marker location and scale for every stain batch, from training ROIs only.

    Returns global statistics plus per-batch statistics so the transform can map
    each batch onto the common scale. Fitting on training ROIs only keeps the
    correction free of held-out information.
    """
    stain = np.asarray(data['stain'])[indices]
    blocks = [np.asarray(data['expressions'][i], dtype=np.float64) for i in indices]
    allcells = np.concatenate(blocks, axis=0)
    g_mean, g_std = allcells.mean(0), allcells.std(0)
    per = {}
    for b in np.unique(stain):
        sel = [blk for blk, sb in zip(blocks, stain) if sb == b]
        cells = np.concatenate(sel, axis=0)
        per[int(b)] = (cells.mean(0), cells.std(0))
    return {'global': (g_mean, g_std), 'per_batch': per}


def apply_batch_correction(data, stats, eps=1e-6):
    """Map every ROI's markers from its own batch distribution onto the global one."""
    g_mean, g_std = stats['global']
    per = stats['per_batch']
    out = dict(data)
    corrected = []
    for expr, b in zip(data['expressions'], np.asarray(data['stain'])):
        x = np.asarray(expr, dtype=np.float64)
        b_mean, b_std = per.get(int(b), (g_mean, g_std))
        z = (x - b_mean) / (b_std + eps)
        corrected.append((z * g_std + g_mean).astype(np.float32))
    out['expressions'] = corrected
    return out


def subset(data, indices):
    out = {}
    for key, values in data.items():
        if values is None or key in ('markers', 'celltypes'):
            out[key] = values
        else:
            out[key] = [values[i] for i in indices]
    out['labels'] = np.asarray(out['labels'])
    return out


def drop_single_class_batches(data):
    stain = np.asarray(data['stain'])
    labels = np.asarray(data['labels'])
    keep = [b for b in np.unique(stain) if len(np.unique(labels[stain == b])) > 1]
    idx = np.flatnonzero(np.isin(stain, keep))
    print(f'batch control: keeping batches {sorted(int(b) for b in keep)} -> '
          f'{len(idx)} ROIs')
    return subset(data, idx)


def fix_rois_per_patient(data, n_per_patient, seed):
    """Sample exactly n ROIs per patient; drop patients with fewer.

    ROI count is itself a strong predictor (responders were imaged more), so
    equalising it is the only way a model cannot read the sampling design.
    """
    rng = np.random.default_rng(seed)
    by_patient = collections.defaultdict(list)
    for i, pid in enumerate(data['patient']):
        by_patient[pid].append(i)
    keep = []
    dropped = 0
    for pid, idx in sorted(by_patient.items()):
        if len(idx) < n_per_patient:
            dropped += 1
            continue
        keep.extend(rng.choice(idx, n_per_patient, replace=False).tolist())
    keep.sort()
    print(f'ROI-count control: {n_per_patient} ROIs/patient, dropped {dropped} '
          f'patients with fewer -> {len(keep)} ROIs, '
          f'{len(keep) // n_per_patient} patients')
    return subset(data, keep)


def train_subset_of_pool(pool, train_split):
    """All ROIs from the full pool belonging to this fold's training patients."""
    patients = set(train_split['patient'])
    idx = [i for i, pid in enumerate(pool['patient']) if pid in patients]
    return subset(pool, idx)


def permute_patient_labels(data, seed):
    """Shuffle response labels across patients, keeping each patient's ROIs consistent.

    The null must be built at the level the label lives at. Permuting ROI labels
    would break the within-patient correlation that real data has and produce an
    unrealistically tight null, making the pipeline look more significant than it
    is. Permuting whole patients preserves every other structure -- cohort size,
    class ratio, ROIs per patient, batch composition -- so the only thing destroyed
    is the association being tested.
    """
    rng = np.random.default_rng(seed)
    current = {}
    for pid, label in zip(data['patient'], np.asarray(data['labels'])):
        current[pid] = int(label)
    pids = sorted(current)
    values = [current[p] for p in pids]
    rng.shuffle(values)
    shuffled = dict(zip(pids, values))
    out = dict(data)
    out['labels'] = np.asarray([shuffled[p] for p in data['patient']])
    return out


def patient_table(data):
    per = collections.defaultdict(lambda: {'labels': [], 'stains': [], 'ncells': []})
    for pid, lab, st, co in zip(data['patient'], np.asarray(data['labels']),
                                np.asarray(data['stain']), data['coords']):
        per[pid]['labels'].append(int(lab))
        per[pid]['stains'].append(int(st))
        per[pid]['ncells'].append(len(co))
    pids = sorted(per)
    y = np.array([collections.Counter(per[p]['labels']).most_common(1)[0][0] for p in pids])
    st = np.array([collections.Counter(per[p]['stains']).most_common(1)[0][0] for p in pids])
    nroi = np.array([len(per[p]['labels']) for p in pids], dtype=float)
    ncell = np.array([float(np.mean(per[p]['ncells'])) for p in pids])
    return pids, y, st, nroi, ncell


def aggregate_probs(values, mode='mean'):
    """Pool a patient's ROI probabilities into one patient score.

    Response may be driven by a subset of the tissue, so the mean over every ROI
    can dilute a real signal that is present in only a few regions. These are the
    standard multiple-instance pooling alternatives.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float('nan')
    if mode == 'mean':
        return float(values.mean())
    if mode == 'max':
        return float(values.max())
    if mode == 'top2':
        k = max(1, min(2, values.size))
        return float(np.sort(values)[-k:].mean())
    if mode == 'top25pct':
        k = max(1, int(np.ceil(0.25 * values.size)))
        return float(np.sort(values)[-k:].mean())
    if mode == 'noisy_or':
        # P(at least one responsive region), the classic MIL pooling.
        return float(1.0 - np.prod(np.clip(1.0 - values, 1e-6, 1.0)))
    if mode == 'logit_mean':
        v = np.clip(values, 1e-6, 1 - 1e-6)
        return float(np.mean(np.log(v / (1 - v))))
    if mode == 'median':
        return float(np.median(values))
    if mode == 'q75':
        return float(np.quantile(values, 0.75))
    if mode == 'q90':
        return float(np.quantile(values, 0.90))
    if mode == 'trimmed':
        # 20% trimmed mean: robust to a few aberrant ROIs without chasing the max.
        lo, hi = np.quantile(values, [0.2, 0.8])
        kept = values[(values >= lo) & (values <= hi)]
        return float(kept.mean() if kept.size else values.mean())
    if mode == 'conf_weighted':
        # Weight each ROI by how confident it is, normalised so the weights sum to
        # one -- normalisation is what keeps this independent of the ROI count.
        w = np.abs(values - 0.5) + 1e-3
        return float(np.sum(w * values) / np.sum(w))
    if mode == 'majority_frac':
        # Fraction of a patient's ROIs voting positive: a proportion, so it does
        # not grow with the number of ROIs the way max or noisy-or do.
        return float(np.mean(values >= 0.5))
    if mode == 'majority_conf':
        # Majority vote, then the mean probability among only the winning ROIs.
        vote = np.mean(values >= 0.5) >= 0.5
        side = values[values >= 0.5] if vote else values[values < 0.5]
        return float(side.mean() if side.size else values.mean())
    raise ValueError(f'unknown aggregation {mode!r}')


def roi_to_patient(data, probs, agg='mean'):
    per = collections.defaultdict(lambda: {'labels': [], 'probs': []})
    for pid, lab, pr in zip(data['patient'], np.asarray(data['labels']), probs):
        per[pid]['labels'].append(int(lab))
        per[pid]['probs'].append(float(pr))
    pids = sorted(per)
    y = np.array([collections.Counter(per[q]['labels']).most_common(1)[0][0] for q in pids])
    p = np.array([aggregate_probs(per[q]['probs'], agg) for q in pids])
    return pids, y, p


def bootstrap_ci(y, score, n_boot=5000, seed=0):
    y, score = np.asarray(y), np.asarray(score)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            stats.append(roc_auc_score(y[i], score[i]))
    if not stats:
        return float('nan'), float('nan')
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def build_model_config(base_model_cfg, spec, train_labels, markers, celltypes, logger):
    cfg = copy.deepcopy(base_model_cfg)
    gnn = cfg['gnn']
    gnn['gconv'] = spec['gconv']
    gnn['device'] = 'cpu'
    gnn['logger'] = logger
    gnn['fnorm'] = cfg['fnorm']
    gnn['gmode'] = spec['gmode']
    gnn['lr'] = spec['lr']

    node_f = gnn['node_f']
    dim = {'expressions': len(markers), 'celltypes': len(celltypes),
           'expressions_celltypes': len(markers) + len(celltypes)}[node_f]
    cfg['feature_dim'] = dim
    if spec['gconv'] not in gnn:
        gnn[spec['gconv']] = dict(gnn.get('deepgcn', {'hidden_dim': 128, 'num_layers': 3,
                                                      'drop_p': 0.3}))
    gnn[spec['gconv']]['input_dim'] = dim
    if 'hidden_dim' in gnn[spec['gconv']]:
        gnn[spec['gconv']]['hidden_dim'] = spec['hidden_dim']
    if spec['gconv'] in ('deepgcn', 'deepgine'):
        gnn[spec['gconv']]['num_layers'] = spec['num_layers']

    if spec['balanced']:
        counts = collections.Counter(np.asarray(train_labels).tolist())
        total = sum(counts.values())
        gnn['class_weight'] = {c: total / (2.0 * n) for c, n in counts.items()}
    else:
        gnn['class_weight'] = None
    gnn['node_f_scale'] = spec.get('node_f_scale', 'raw')
    gnn['weight_decay'] = spec.get('weight_decay', 1e-5)
    if 'drop_p' in spec and 'drop_p' in gnn.get(spec['gconv'], {}):
        gnn[spec['gconv']]['drop_p'] = spec['drop_p']
    gnn['loss'] = spec.get('loss', 'bce')
    gnn['focal_gamma'] = spec.get('focal_gamma', 2.0)
    gnn['focal_alpha'] = spec.get('focal_alpha', 0.25)
    cfg['eval'] = 'split'
    cfg['nm_epochs'] = spec['epochs']
    cfg['val_ratio'] = 0
    cfg['epoch_curve'] = False
    return cfg


def resample_balanced(data, n_per_patient, seed):
    """Draw exactly n ROIs per patient, with replacement when a patient has fewer.

    Called fresh every epoch. Unlike the one-shot control this keeps every patient
    and, across epochs, shows the model every ROI -- while each individual epoch
    still carries an equal number per patient, so ROI count cannot be learned.
    """
    rng = np.random.default_rng(seed)
    by_patient = collections.defaultdict(list)
    for i, pid in enumerate(data['patient']):
        by_patient[pid].append(i)
    keep = []
    for _, idx in sorted(by_patient.items()):
        keep.extend(rng.choice(idx, n_per_patient,
                               replace=len(idx) < n_per_patient).tolist())
    return subset(data, sorted(keep))


def augment_rois(data, seed, max_cells=0, jitter=0.0):
    """Per-epoch graph and feature augmentation.

    Cell subsampling is the load-bearing one: it resamples which cells exist, so
    the neighbourhood graph is rebuilt differently every epoch (a DropNode-style
    regulariser). Multiplicative jitter models per-cell staining variability.
    """
    rng = np.random.default_rng(seed)
    out = dict(data)
    exprs, coords, cell_labels = [], [], []
    for expr, coord, labels in zip(data['expressions'], data['coords'],
                                   data['cell_labels']):
        expr = np.asarray(expr, dtype=np.float32)
        coord = np.asarray(coord)
        labels = np.asarray(labels)
        n = len(coord)
        if max_cells and n > max_cells:
            sel = rng.choice(n, max_cells, replace=False)
            expr, coord, labels = expr[sel], coord[sel], labels[sel]
        if jitter > 0:
            expr = expr * rng.normal(1.0, jitter, size=expr.shape).astype(np.float32)
        exprs.append(expr)
        coords.append(coord)
        cell_labels.append(labels)
    out['expressions'] = exprs
    out['coords'] = coords
    out['cell_labels'] = cell_labels
    return out


def fit_predict(base_model_cfg, spec, train, test, seed, logger, aug=None):
    from models.trainer import ModelTrainer
    cfg = build_model_config(base_model_cfg, spec, train['labels'],
                             train['markers'], train['celltypes'], logger)
    trainer = ModelTrainer(cfg, logger=logger, logfile=None, seed=seed)
    for epoch in range(spec['epochs']):
        if aug and aug.get('enabled'):
            # Fresh balanced draw plus fresh cell subsample every epoch, so the
            # model sees all ROIs over training but never an unbalanced one.
            epoch_data = resample_balanced(train, aug['n_per_patient'],
                                           seed * 1000 + epoch)
            epoch_data = augment_rois(epoch_data, seed * 1000 + epoch,
                                      aug.get('max_cells', 0), aug.get('jitter', 0.0))
        else:
            epoch_data = train
        trainer.fit(epoch_data)
    test_probs = np.asarray(trainer.predict_proba(test)).reshape(-1)
    # Reference distribution for per-fold calibration. Uses ONLY training patients,
    # so it leaks nothing about the held-out ones.
    train_probs = np.asarray(trainer.predict_proba(train)).reshape(-1)
    return test_probs, train_probs


def calibrate_to_fold(test_probs, train_probs):
    """Map raw probabilities to their percentile within this fold's own training scores.

    Each CV fold trains a separate model with its own bias and scale. Pooling raw
    probabilities across folds compares models, not patients -- and in leave-one-out
    the held-out patient's own label shifts its fold's class balance, so that bias
    anti-correlates with the truth by construction. Converting to a within-fold
    percentile removes fold-level location and scale while preserving the ranking
    the model actually learned.
    """
    reference = np.sort(np.asarray(train_probs, dtype=float))
    if reference.size == 0:
        return np.asarray(test_probs, dtype=float)
    ranks = np.searchsorted(reference, np.asarray(test_probs, dtype=float), side='left')
    return ranks / float(reference.size)


def inner_select(base_model_cfg, train, seed, logger, inner_folds, grid, aug=None, agg='mean'):
    """Choose a config using only training patients."""
    y = np.asarray(train['labels'])
    groups = np.asarray(train['patient'])
    n_groups = len(set(groups))
    folds = min(inner_folds, n_groups)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for spec in grid:
        fold_aucs = []
        for tr, va in splitter.split(np.zeros(len(y)), y, groups=groups):
            sub_tr, sub_va = subset(train, tr), subset(train, va)
            if len(np.unique(np.asarray(sub_va['labels']))) < 2:
                continue
            try:
                probs, tr_probs = fit_predict(base_model_cfg, spec, sub_tr, sub_va,
                                              seed, logger, aug)
                probs = calibrate_to_fold(probs, tr_probs)
                _, vy, vp = roi_to_patient(sub_va, probs, agg=agg)
                if len(np.unique(vy)) > 1:
                    fold_aucs.append(roc_auc_score(vy, vp))
            except Exception as exc:  # a bad config must not kill the search
                print(f'      inner fit failed for {spec}: {type(exc).__name__}: {exc}')
        scores.append(float(np.mean(fold_aucs)) if fold_aucs else float('nan'))
    best = int(np.nanargmax(scores)) if not np.all(np.isnan(scores)) else 0
    return grid[best], scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--cache', default=BASE_CACHE)
    parser.add_argument('--cv', choices=['grouped', 'loo'], default='grouped')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--inner-folds', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--rois-per-patient', type=int, default=4)
    parser.add_argument('--keep-all-batches', action='store_true')
    parser.add_argument('--batch-correct', action='store_true',
                        help='regress stain-batch location/scale out of the markers, '
                             'fitted per fold on training ROIs; keeps all 63 patients')
    parser.add_argument('--fixed-spec', type=int, default=None,
                        help='skip inner selection and use GRID[i] directly')
    parser.add_argument('--augment', action='store_true',
                        help='resample a balanced ROI set per epoch and augment cells')
    parser.add_argument('--aug-max-cells', type=int, default=0,
                        help='subsample at most this many cells per ROI each epoch')
    parser.add_argument('--aug-jitter', type=float, default=0.0,
                        help='multiplicative marker noise sd')
    parser.add_argument('--no-calibrate', action='store_true',
                        help='pool raw per-fold probabilities (the old, biased behaviour)')
    parser.add_argument('--eval-all-rois', action='store_true',
                        help='keep every patient and every ROI for evaluation; the '
                             'ROI-count control then lives in training only')
    parser.add_argument('--agg', default='mean',
                        choices=['mean', 'max', 'top2', 'top25pct', 'noisy_or', 'logit_mean',
                                 'median', 'q75', 'q90', 'trimmed', 'conf_weighted',
                                 'majority_frac', 'majority_conf'])
    parser.add_argument('--loss', default='bce', choices=['bce', 'focal'])
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--permute', type=int, default=None,
                        help='shuffle patient labels with this seed (null distribution)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    with open(args.cache, 'rb') as handle:
        data = pickle.load(handle)
    data['labels'] = np.asarray([LABELS[l] for l in data['labels']])

    if not args.keep_all_batches:
        data = drop_single_class_batches(data)
    train_pool = data
    if args.eval_all_rois:
        # Mean ROI probability is an unbiased patient score whatever the ROI count
        # -- count changes the variance of that mean, not its expectation -- and the
        # model scores one ROI at a time so it cannot see the count. Balancing the
        # TRAINING draw is what stops count being learnable; discarding ROIs and
        # patients at evaluation only costs statistical power.
        print(f'eval-all-rois: keeping {len(data["labels"])} ROIs, '
              f'{len(set(data["patient"]))} patients for evaluation; training draws '
              f'{args.rois_per_patient} ROIs/patient per epoch')
    elif args.rois_per_patient > 0:
        if args.augment:
            # Training resamples a balanced set from the full pool every epoch, so
            # keep every ROI and every patient here; only evaluation is fixed.
            print('augment: training draws a fresh balanced set per epoch from '
                  f'{len(data["labels"])} ROIs, {len(set(data["patient"]))} patients')
            data = fix_rois_per_patient(data, args.rois_per_patient, args.seed)
        else:
            data = fix_rois_per_patient(data, args.rois_per_patient, args.seed)
            train_pool = data

    if args.permute is not None:
        # After the confound controls so the cohort is identical to the real run.
        data = permute_patient_labels(data, args.permute)
        train_pool = permute_patient_labels(train_pool, args.permute)
        print(f'PERMUTED patient labels with seed {args.permute} (null run)')

    pids, py, pstain, pnroi, pncell = patient_table(data)
    print(f'cohort: {len(data["labels"])} ROIs, {len(pids)} patients, '
          f'labels {dict(collections.Counter(py.tolist()))}')
    pindex = {p: i for i, p in enumerate(pids)}

    os.environ.setdefault('WANDB_MODE', 'disabled')
    import wandb
    logger = wandb.init(mode='disabled')

    roi_y = np.asarray(data['labels'])
    roi_groups = np.asarray(data['patient'])

    oof = np.full(len(pids), np.nan)
    roi_records = []
    chosen = []
    for repeat in range(args.repeats):
        if args.cv == 'loo':
            splitter = LeaveOneGroupOut()
            splits = list(splitter.split(np.zeros(len(roi_y)), roi_y, groups=roi_groups))
        else:
            splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                            random_state=100 + repeat + 1000 * args.seed)
            splits = list(splitter.split(np.zeros(len(roi_y)), roi_y, groups=roi_groups))

        for fold, (tr, te) in enumerate(splits):
            train, test = subset(data, tr), subset(data, te)
            if args.batch_correct:
                stats = fit_batch_correction(data, tr)
                train, test = (apply_batch_correction(train, stats),
                               apply_batch_correction(test, stats))
            te_pids = sorted(set(test['patient']))
            if args.fixed_spec is not None:
                spec, inner_scores = GRID[args.fixed_spec], None
            else:
                spec, inner_scores = inner_select(
                    config['model'], train, 100 + repeat + 1000 * args.seed, logger,
                    args.inner_folds, GRID,
                    dict(enabled=args.augment, n_per_patient=args.rois_per_patient,
                         max_cells=args.aug_max_cells, jitter=args.aug_jitter),
                    agg=args.agg)
            chosen.append(spec)
            aug = dict(enabled=args.augment, n_per_patient=args.rois_per_patient,
                       max_cells=args.aug_max_cells, jitter=args.aug_jitter)
            fit_on = (train_subset_of_pool(train_pool, train)
                      if (args.augment or args.eval_all_rois) else train)
            if args.batch_correct:
                fit_on = apply_batch_correction(fit_on, stats)
            probs, tr_probs = fit_predict(config['model'], spec, fit_on, test,
                                          100 + repeat + 1000 * args.seed, logger, aug)
            if not args.no_calibrate:
                probs = calibrate_to_fold(probs, tr_probs)
            roi_records.extend({'patient': str(pid), 'label': int(lab), 'prob': float(pr)}
                               for pid, lab, pr in zip(test['patient'],
                                                       np.asarray(test['labels']), probs))
            _, _, pat_prob = roi_to_patient(test, probs, agg=args.agg)
            for p, v in zip(te_pids, pat_prob):
                oof[pindex[p]] = v
            msg = (f'  fold {fold + 1}/{len(splits)}: {len(te_pids)} test patients, '
                   f'spec={spec["gconv"]}/L{spec["num_layers"]}/{spec["gmode"]}/'
                   f'bal{int(spec["balanced"])}/e{spec["epochs"]}')
            if inner_scores is not None:
                msg += f'  inner={["%.2f" % s for s in inner_scores]}'
            print(msg, flush=True)

    mask = ~np.isnan(oof)
    print(f'\nscored {mask.sum()}/{len(pids)} patients out-of-fold')
    rows = {}
    gnn_label = ('GNN (fixed spec)' if args.fixed_spec is not None
                 else 'GNN (nested-selected)')
    for name, score in ((gnn_label, oof),
                        ('baseline: #ROIs/patient', pnroi),
                        ('baseline: stain batch', pstain.astype(float)),
                        ('baseline: mean cells/ROI', pncell)):
        m = mask & ~np.isnan(score)
        if len(np.unique(py[m])) < 2:
            continue
        raw = roc_auc_score(py[m], score[m])
        # Baselines are direction-free confounders, so orient them upward. A model
        # is NOT direction-free: an AUC below 0.5 is a real result and its interval
        # must be computed on the same orientation, not on the flipped score.
        oriented = score[m]
        if name.startswith('baseline') and raw < 0.5:
            oriented = -score[m]
        auc = roc_auc_score(py[m], oriented)
        lo, hi = bootstrap_ci(py[m], oriented)
        rows[name] = (auc, lo, hi, int(m.sum()))

    print('\n' + '=' * 74)
    print(f'{args.cv.upper()} CV, {args.rois_per_patient} ROIs/patient, '
          f'{"all" if args.keep_all_batches else "mixed-class"} batches')
    print('=' * 74)
    print(f'{"model":30s} {"OOF AUC":>9s} {"95% CI":>18s} {"n":>5s}')
    for name, (auc, lo, hi, n) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
        print(f'{name:30s} {auc:>9.3f} {f"({lo:.3f}-{hi:.3f})":>18s} {n:>5d}')

    picks = collections.Counter(
        f'{s["gconv"]}/L{s["num_layers"]}/{s["gmode"]}/bal{int(s["balanced"])}/e{s["epochs"]}'
        for s in chosen)
    print('\nconfigs chosen by inner selection:')
    for name, n in picks.most_common():
        print(f'  {n:3d}x {name}')

    if args.out:
        with open(args.out, 'w') as handle:
            json.dump({'args': vars(args), 'n_patients': int(mask.sum()),
                       'results': {k: dict(auc=v[0], ci_low=v[1], ci_high=v[2], n=v[3])
                                   for k, v in rows.items()},
                       'chosen': dict(picks),
                       'oof': oof.tolist(), 'labels': py.tolist(),
                       'roi_predictions': roi_records}, handle, indent=2)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
