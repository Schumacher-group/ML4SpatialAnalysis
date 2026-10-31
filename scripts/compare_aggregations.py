"""Score every ROI->patient aggregation from one run's saved ROI-level predictions.

Aggregation choice is not free: some pooling rules grow with the number of ROIs a
patient contributed, and in this cohort ROI count is itself correlated with
response (AUC 0.66-0.74). An aggregation that is count-sensitive therefore
imports the confounder no matter how well the model was trained. This reports,
for each rule, the plain AUC, the correlation between the patient score and ROI
count, and the count-stratified AUC -- so a rule can be judged on the signal it
extracts rather than the artefact it inherits.
"""

import argparse
import collections
import json

import numpy as np
from sklearn.metrics import roc_auc_score

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_gnn import aggregate_probs  # noqa: E402

# Rules whose expected value does not move with the number of ROIs. Measured, not
# assumed: each rule was applied to the same score distribution at n=4 and n=38 and
# the drift recorded. Sample quantiles are biased at small n, so q75/q90/top25pct
# are NOT neutral despite being scale-free in form.
COUNT_NEUTRAL = {'mean', 'conf_weighted', 'median', 'logit_mean', 'majority_frac',
                 'trimmed'}
COUNT_DRIFT = {'mean': -0.000, 'conf_weighted': 0.000, 'median': 0.001,
               'logit_mean': 0.002, 'majority_frac': 0.005, 'trimmed': 0.006,
               'majority_conf': -0.040, 'top25pct': 0.043, 'q75': 0.052,
               'noisy_or': 0.062, 'q90': 0.105, 'max': 0.178, 'top2': 0.241}
ALL_MODES = ['mean', 'median', 'logit_mean', 'trimmed', 'conf_weighted',
             'majority_frac', 'majority_conf', 'q75', 'q90', 'top25pct',
             'top2', 'noisy_or', 'max']


def stratified_auc(y, score, strata):
    num = den = 0.0
    for s in np.unique(strata):
        k = strata == s
        if len(np.unique(y[k])) < 2:
            continue
        num += roc_auc_score(y[k], score[k]) * k.sum()
        den += k.sum()
    return num / den if den else float('nan')


def bootstrap(y, score, strata, n_boot, seed, stratify):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        v = (stratified_auc(y[i], score[i], strata[i]) if stratify
             else roc_auc_score(y[i], score[i]))
        if not np.isnan(v):
            out.append(v)
    return np.array(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_json')
    parser.add_argument('--n-boot', type=int, default=3000)
    args = parser.parse_args()

    with open(args.run_json) as handle:
        data = json.load(handle)
    records = data.get('roi_predictions')
    if not records:
        raise SystemExit(f'{args.run_json} has no roi_predictions; rerun with the '
                         'updated search_gnn.py to save them')

    per = collections.defaultdict(lambda: {'labels': [], 'probs': []})
    for r in records:
        per[r['patient']]['labels'].append(int(r['label']))
        per[r['patient']]['probs'].append(float(r['prob']))
    pids = sorted(per)
    y = np.array([collections.Counter(per[p]['labels']).most_common(1)[0][0] for p in pids])
    nroi = np.array([len(per[p]['probs']) for p in pids], dtype=float)
    tertile = np.digitize(nroi, np.percentile(nroi, [33.3, 66.7]))

    print(f'{len(pids)} patients, {len(records)} ROI predictions, '
          f'labels {dict(collections.Counter(y.tolist()))}')
    print(f'ROI count vs label AUC = '
          f'{max(roc_auc_score(y, nroi), 1 - roc_auc_score(y, nroi)):.3f}  '
          f'(the confounder this table must survive)\n')

    header = ('%-15s %6s %7s %16s %8s %9s %7s' %
              ('aggregation', 'AUC', 'r(#ROI)', '95% CI', 'stratAUC', 'CI_low', 'drift'))
    print(header)
    print('-' * len(header))

    rows = []
    for mode in ALL_MODES:
        score = np.array([aggregate_probs(per[p]['probs'], mode) for p in pids])
        if np.allclose(score, score[0]):
            continue
        auc = roc_auc_score(y, score)
        r = float(np.corrcoef(nroi, score)[0, 1])
        bs = bootstrap(y, score, tertile, args.n_boot, 0, stratify=False)
        sa = stratified_auc(y, score, tertile)
        sbs = bootstrap(y, score, tertile, args.n_boot, 0, stratify=True)
        rows.append((mode, auc, r, np.percentile(bs, 2.5), np.percentile(bs, 97.5),
                     sa, np.percentile(sbs, 2.5), mode in COUNT_NEUTRAL))

    for mode, auc, r, lo, hi, sa, slo, neutral in sorted(rows, key=lambda z: -z[5]):
        print('%-15s %6.3f %+7.3f %16s %8.3f %9.3f %+7.3f%s' %
              (mode, auc, r, f'({lo:.3f}-{hi:.3f})', sa, slo,
               COUNT_DRIFT.get(mode, float('nan')), '' if neutral else '  <-not neutral'))

    print('\nRanked by count-stratified AUC, which is the number that cannot be '
          'inflated by ROI count.\nA rule marked NO under "neutral" grows with the '
          'number of ROIs and should not be\nreported without the stratified column.')


if __name__ == '__main__':
    main()
