"""Repeated stratified grouped cross-validation for the ROI-level GNN pipeline.

The single 70/30 split evaluates 18 patients, where a random classifier scores
AUC 0.499 +/- 0.144 -- wide enough that no observed difference was interpretable,
and 45 of 63 patients were never evaluated at all. This harness instead puts
every patient in a held-out fold exactly once per repeat, so the reported AUC is
computed over all 63 (or 50, when single-class batches are dropped) patients.

Folds are grouped by patient and stratified by label, so no patient spans train
and test. Trivial baselines are scored on exactly the same folds, because a model
that cannot beat "count the ROIs" has not learned biology.
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from mainutils.utils import load_config

BASE_CACHE = ('/n/data1/hms/cellbio/sander/asif/KCLData/data/newdata3/'
              'updated_cell_table_CleanCohort_Metadata_v2_qc_trainfit.pkl')
LABELS = {'pCR': 1, 'Responder': 1, 'Non-Responder': 0}


def subset(data, indices):
    out = {}
    for key, values in data.items():
        if values is None or key in ('markers', 'celltypes'):
            out[key] = values
        else:
            out[key] = [values[i] for i in indices]
    out['labels'] = np.asarray(out['labels'])
    return out


def patient_frame(data):
    """Patient-level label / batch / ROI-count table, aligned to unique patients."""
    per = {}
    for pid, lab, st, coords in zip(data['patient'], np.asarray(data['labels']),
                                    np.asarray(data['stain']), data['coords']):
        e = per.setdefault(pid, {'labels': [], 'stains': [], 'ncells': []})
        e['labels'].append(int(lab))
        e['stains'].append(int(st))
        e['ncells'].append(len(coords))
    pids = sorted(per)
    y = np.array([collections.Counter(per[p]['labels']).most_common(1)[0][0] for p in pids])
    st = np.array([collections.Counter(per[p]['stains']).most_common(1)[0][0] for p in pids])
    nroi = np.array([len(per[p]['labels']) for p in pids], dtype=float)
    ncell = np.array([np.mean(per[p]['ncells']) for p in pids], dtype=float)
    return pids, y, st, nroi, ncell


def roi_to_patient(data, roi_probs):
    """Mean ROI probability per patient, with the modal ROI label."""
    per = {}
    for pid, lab, prob in zip(data['patient'], np.asarray(data['labels']), roi_probs):
        e = per.setdefault(pid, {'labels': [], 'probs': []})
        e['labels'].append(int(lab))
        e['probs'].append(float(prob))
    pids = sorted(per)
    y = np.array([collections.Counter(per[p]['labels']).most_common(1)[0][0] for p in pids])
    p = np.array([np.mean(per[p]['probs']) for p in pids])
    return pids, y, p


def bootstrap_auc_ci(y, score, n_boot=5000, seed=0):
    """Patient-level bootstrap CI; resamples patients, not ROIs."""
    y = np.asarray(y)
    score = np.asarray(score)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(roc_auc_score(y[idx], score[idx]))
    if not stats:
        return float('nan'), float('nan')
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def mean_marker_matrix(data):
    """Per-ROI mean marker vector -- the representation a 1-layer GCN collapses to."""
    return np.array([np.asarray(e, dtype=np.float64).mean(0) for e in data['expressions']])


def fit_gnn_fold(model_config, train, test, seed, epochs, logger):
    from models.trainer import ModelTrainer
    cfg = copy.deepcopy(model_config)
    cfg['eval'] = 'split'
    cfg['nm_epochs'] = epochs
    cfg['val_ratio'] = 0
    cfg['epoch_curve'] = False
    trainer = ModelTrainer(cfg, logger=logger, logfile=None, seed=seed)
    for _ in range(epochs):
        trainer.fit(train)
    return np.asarray(trainer.predict_proba(test)).reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--cache', default=BASE_CACHE)
    parser.add_argument('--gconv', default='deepgcn')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--drop-single-class-batches', action='store_true')
    parser.add_argument('--baselines-only', action='store_true')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    with open(args.cache, 'rb') as handle:
        data = pickle.load(handle)
    data['labels'] = np.asarray([LABELS[l] for l in data['labels']])

    if args.drop_single_class_batches:
        stain = np.asarray(data['stain'])
        keep = [b for b in np.unique(stain)
                if len(np.unique(np.asarray(data['labels'])[stain == b])) > 1]
        idx = np.flatnonzero(np.isin(stain, keep))
        data = subset(data, idx)
        print(f'kept stain batches {sorted(int(b) for b in keep)}: '
              f'{len(idx)} ROIs, {len(set(data["patient"]))} patients')

    pids, py, pstain, pnroi, pncell = patient_frame(data)
    print(f'cohort for CV: {len(data["labels"])} ROIs, {len(pids)} patients, '
          f'labels {dict(collections.Counter(py.tolist()))}')

    roi_patient = np.asarray(data['patient'])
    roi_label = np.asarray(data['labels'])
    pindex = {p: i for i, p in enumerate(pids)}

    results = collections.defaultdict(lambda: collections.defaultdict(list))

    logger = None
    if not args.baselines_only:
        os.environ.setdefault('WANDB_MODE', 'disabled')
        import wandb
        logger = wandb.init(mode='disabled')

    for repeat in range(args.repeats):
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                        random_state=1000 + repeat)
        oof = {name: np.full(len(pids), np.nan) for name in
               ('n_rois', 'stain', 'mean_marker_logreg', f'gnn_{args.gconv}')}

        for fold, (tr_idx, te_idx) in enumerate(
                splitter.split(np.zeros(len(roi_label)), roi_label, groups=roi_patient)):
            train, test = subset(data, tr_idx), subset(data, te_idx)
            te_pids = sorted(set(test['patient']))
            slot = [pindex[p] for p in te_pids]

            # --- trivial baselines, same folds -------------------------------
            _, tr_y, tr_stain, tr_nroi, _ = patient_frame(train)
            te_pid_list, te_y, te_stain, te_nroi, _ = patient_frame(test)
            oof['n_rois'][slot] = te_nroi
            oof['stain'][slot] = te_stain

            scaler = StandardScaler().fit(mean_marker_matrix(train))
            logreg = LogisticRegression(max_iter=2000, class_weight='balanced')
            logreg.fit(scaler.transform(mean_marker_matrix(train)), train['labels'])
            roi_prob = logreg.predict_proba(scaler.transform(mean_marker_matrix(test)))[:, 1]
            _, _, pat_prob = roi_to_patient(test, roi_prob)
            oof['mean_marker_logreg'][slot] = pat_prob

            # --- the GNN -----------------------------------------------------
            if not args.baselines_only:
                mcfg = copy.deepcopy(config['model'])
                mcfg['gnn']['gconv'] = args.gconv
                mcfg['gnn']['device'] = 'cpu'
                mcfg['gnn']['logger'] = logger
                mcfg['gnn']['fnorm'] = mcfg['fnorm']
                node_f = mcfg['gnn']['node_f']
                dim = {'expressions': len(data['markers']),
                       'celltypes': len(data['celltypes']),
                       'expressions_celltypes': len(data['markers']) + len(data['celltypes'])}[node_f]
                mcfg['feature_dim'] = dim
                mcfg['gnn'][args.gconv]['input_dim'] = dim
                probs = fit_gnn_fold(mcfg, train, test, 1000 + repeat, args.epochs, logger)
                _, _, pat_prob = roi_to_patient(test, probs)
                oof[f'gnn_{args.gconv}'][slot] = pat_prob
            print(f'  repeat {repeat} fold {fold}: {len(tr_idx)} train ROIs, '
                  f'{len(te_idx)} test ROIs, {len(te_pids)} test patients')

        for name, scores in oof.items():
            mask = ~np.isnan(scores)
            if mask.sum() < len(pids):
                print(f'  ! {name}: only {mask.sum()}/{len(pids)} patients scored')
            if len(np.unique(py[mask])) < 2:
                continue
            auc = roc_auc_score(py[mask], scores[mask])
            # Baselines are direction-free confounders; orient them upward so a
            # strong negative association is not read as "no signal".
            if name in ('n_rois', 'stain'):
                auc = max(auc, 1 - auc)
            results[name]['auc'].append(auc)
            results[name]['scores'].append(scores.copy())

    print('\n' + '=' * 78)
    print(f'Repeated stratified grouped {args.folds}-fold CV x {args.repeats} repeats')
    print(f'every one of {len(pids)} patients evaluated out-of-fold each repeat')
    print('=' * 78)
    print(f"{'model':24s} {'OOF AUC (mean+-sd over repeats)':>32s} {'95% patient bootstrap CI':>26s}")
    table = {}
    for name, payload in sorted(results.items(), key=lambda kv: -np.mean(kv[1]['auc'])):
        aucs = np.array(payload['auc'])
        pooled = np.nanmean(np.array(payload['scores']), axis=0)
        mask = ~np.isnan(pooled)
        lo, hi = bootstrap_auc_ci(py[mask], pooled[mask])
        if name in ('n_rois', 'stain') and roc_auc_score(py[mask], pooled[mask]) < 0.5:
            lo, hi = 1 - hi, 1 - lo
        print(f'{name:24s} {aucs.mean():>20.3f} +- {aucs.std():.3f} {f"({lo:.3f}-{hi:.3f})":>26s}')
        table[name] = {'auc_mean': float(aucs.mean()), 'auc_sd': float(aucs.std()),
                       'ci_low': lo, 'ci_high': hi, 'per_repeat': aucs.tolist()}

    if args.out:
        with open(args.out, 'w') as handle:
            json.dump({'config': vars(args), 'n_patients': len(pids),
                       'results': table}, handle, indent=2)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
