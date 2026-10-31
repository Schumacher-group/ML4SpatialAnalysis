"""Quantify how far Stain_Batch predicts the response label, cohort-wide.

The GNN comparison found that batch identity alone reaches patient AUC ~0.84 on a
held-out split, which is far above anything the models achieve. Before any
modelling change is worth making, the cohort-level association has to be known:
if response is largely determined by which batch a sample was stained in, an
apparently successful model cannot be distinguished from a batch detector.

Reads the base (pre-split) cache so the numbers describe the whole cohort rather
than one draw.
"""

import argparse
import collections
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

BASE_CACHE = ('/n/data1/hms/cellbio/sander/asif/KCLData/data/newdata3/'
              'updated_cell_table_CleanCohort_Metadata_v2_qc_trainfit.pkl')


def cramers_v(table):
    """Bias-corrected Cramer's V for a contingency table."""
    table = np.asarray(table, dtype=float)
    n = table.sum()
    if n == 0:
        return float('nan')
    row = table.sum(1, keepdims=True)
    col = table.sum(0, keepdims=True)
    expected = row @ col / n
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum(np.where(expected > 0, (table - expected) ** 2 / expected, 0.0))
    phi2 = chi2 / n
    r, k = table.shape
    phi2 = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r = r - (r - 1) ** 2 / (n - 1)
    k = k - (k - 1) ** 2 / (n - 1)
    denom = min(k - 1, r - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else float('nan')


def one_sided_auc(y, score):
    """AUC oriented so it is >= 0.5; a confounder is informative either way."""
    auc = roc_auc_score(y, score)
    return max(auc, 1 - auc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', default=BASE_CACHE)
    args = parser.parse_args()

    with open(args.cache, 'rb') as handle:
        data = pickle.load(handle)

    labels = np.asarray([1 if l in ('pCR', 'Responder', 1) else 0 for l in data['labels']])
    stain = np.asarray(data['stain'], dtype=int)
    patients = np.asarray(data['patient'])
    ncells = np.asarray([len(c) for c in data['coords']], dtype=float)

    print(f'cohort: {len(labels)} ROIs, {len(set(patients))} patients, '
          f'{len(set(stain))} stain batches')
    print(f'ROI labels: {dict(collections.Counter(labels.tolist()))}\n')

    # ---- ROI level -------------------------------------------------------
    batches = sorted(set(stain.tolist()))
    table = np.zeros((len(batches), 2))
    print('ROI-level composition by stain batch')
    print(f"{'batch':>6s} {'n':>5s} {'neg':>5s} {'pos':>5s} {'%pos':>6s}  purity")
    for i, b in enumerate(batches):
        m = stain == b
        neg = int((labels[m] == 0).sum())
        pos = int((labels[m] == 1).sum())
        table[i] = [neg, pos]
        pct = 100 * pos / max(1, neg + pos)
        flag = '  <-- SINGLE-CLASS' if neg == 0 or pos == 0 else ''
        print(f'{b:>6d} {neg + pos:>5d} {neg:>5d} {pos:>5d} {pct:>5.1f}%{flag}')

    print(f"\nROI-level: Cramer's V(stain, label) = {cramers_v(table):.3f}")
    print(f'ROI-level: AUC(stain -> label)       = {one_sided_auc(labels, stain):.3f}')
    print(f'ROI-level: AUC(cell count -> label)  = {one_sided_auc(labels, ncells):.3f}')

    # ---- patient level ---------------------------------------------------
    per_patient = {}
    for pid, lab, st in zip(patients, labels, stain):
        entry = per_patient.setdefault(pid, {'labels': [], 'stains': []})
        entry['labels'].append(int(lab))
        entry['stains'].append(int(st))

    pids = sorted(per_patient)
    plabels = np.array([collections.Counter(per_patient[p]['labels']).most_common(1)[0][0]
                        for p in pids])
    pstain = np.array([collections.Counter(per_patient[p]['stains']).most_common(1)[0][0]
                       for p in pids])
    proi = np.array([len(per_patient[p]['labels']) for p in pids], dtype=float)

    multi = sum(1 for p in pids if len(set(per_patient[p]['stains'])) > 1)
    print(f'\npatients: {len(pids)}  labels {dict(collections.Counter(plabels.tolist()))}')
    print(f'patients whose ROIs span >1 stain batch: {multi}/{len(pids)}')

    ptable = np.zeros((len(batches), 2))
    print('\npatient-level composition by (modal) stain batch')
    print(f"{'batch':>6s} {'n':>5s} {'neg':>5s} {'pos':>5s} {'%pos':>6s}  purity")
    for i, b in enumerate(batches):
        m = pstain == b
        neg = int((plabels[m] == 0).sum())
        pos = int((plabels[m] == 1).sum())
        ptable[i] = [neg, pos]
        if neg + pos == 0:
            print(f'{b:>6d} {0:>5d} {0:>5d} {0:>5d}      -')
            continue
        pct = 100 * pos / (neg + pos)
        flag = '  <-- SINGLE-CLASS' if neg == 0 or pos == 0 else ''
        print(f'{b:>6d} {neg + pos:>5d} {neg:>5d} {pos:>5d} {pct:>5.1f}%{flag}')

    print(f"\npatient-level: Cramer's V(stain, label) = {cramers_v(ptable):.3f}")
    print(f'patient-level: AUC(stain -> label)      = {one_sided_auc(plabels, pstain):.3f}')
    print(f'patient-level: AUC(#ROIs -> label)      = {one_sided_auc(plabels, proi):.3f}')

    usable = [b for i, b in enumerate(batches) if ptable[i].min() > 0]
    n_usable = int(sum(ptable[i].sum() for i, b in enumerate(batches) if b in usable))
    print(f'\nbatches with BOTH classes at patient level: {usable} '
          f'({n_usable}/{len(pids)} patients)')
    print('Restricting to those batches is the only way to estimate a '
          'batch-free effect without a correction step.')


if __name__ == '__main__':
    main()
