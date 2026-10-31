"""Aggregate the per-run JSON metric sinks into a GNN comparison table.

Each run writes {LOG_PATH}/metrics/{logname}.json via AbstractModel.record_metrics.
The seed is not stored as its own field, but main.py embeds it in the logname
(``..._seed_42_balanced_train_...``), so it is recovered by regex.

Test metrics are logged once, at the final epoch (models/trainer.py), so the last
entry for a Test mode is the reported value. Runs are grouped by
gconv x gmethod x k x node_f, so one metrics dir can hold a graph or k sweep.
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict

SEED_RE = re.compile(r'_seed_(\d+)_')

ROI_TEST = 'ROILevelTest'
PATIENT_TEST = 'Patient Level Test'
CELL = ('gconv', 'gmethod', 'k', 'node_f')


def cell_key(cfg):
    return tuple(cfg.get(field) for field in CELL)


def load_records(metrics_dir, gconvs=None, gmethod=None):
    records = []
    for path in sorted(glob.glob(os.path.join(metrics_dir, '*.json'))):
        try:
            with open(path) as handle:
                record = json.load(handle)
        except (ValueError, OSError):
            print(f'  ! skipping unreadable {os.path.basename(path)}')
            continue
        cfg = record.get('config', {})
        if cfg.get('model') != 'gnn':
            continue
        if gconvs and cfg.get('gconv') not in gconvs:
            continue
        if gmethod and cfg.get('gmethod') != gmethod:
            continue
        match = SEED_RE.search(cfg.get('logfile') or '')
        cfg['seed'] = int(match.group(1)) if match else None
        record['path'] = path
        records.append(record)
    return dedupe(records)


def completeness(record):
    """Rank records so a preempted partial run loses to its completed rerun."""
    metrics = record.get('metrics', {})
    has_test = bool(metrics.get(ROI_TEST)) and bool(metrics.get(PATIENT_TEST))
    return (has_test, sum(len(v) for v in metrics.values()))


def dedupe(records):
    """Keep one record per grid cell and seed.

    gpu_requeue preempts and requeues jobs, so a single grid cell can leave a
    truncated JSON from the killed attempt next to the full one from the rerun.
    Averaging both would double-count the cell and drag in a run that never
    reached test evaluation.
    """
    best = {}
    for record in records:
        cfg = record['config']
        key = cell_key(cfg) + (cfg.get('seed'),)
        if key not in best or completeness(record) > completeness(best[key]):
            best[key] = record
    dropped = len(records) - len(best)
    if dropped:
        print(f'  deduped {dropped} superseded/partial run(s) by cell and seed')
    return list(best.values())


def final(record, mode, needle):
    """Last logged value for the metric whose key contains `needle`."""
    entries = record.get('metrics', {}).get(mode) or []
    if not entries:
        return None
    last = entries[-1]
    for key, value in last.items():
        if needle in key:
            return value
    return None


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    if len(values) == 1:
        return values[0], 0.0, 1
    return statistics.mean(values), statistics.stdev(values), len(values)


def fmt(mean, sd, n):
    if mean is None:
        return 'n/a'
    return f'{mean:.3f} +/- {sd:.3f} (n={n})'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics-dir',
                        default='/n/data1/hms/cellbio/sander/asif/ML4SA_Results/metrics')
    parser.add_argument('--gconvs', nargs='+', default=None,
                        help='restrict to these convolutions; default keeps every run found')
    parser.add_argument('--gmethod', default=None,
                        help='restrict to one graph construction; default keeps all')
    parser.add_argument('--out-prefix', default=None)
    args = parser.parse_args()

    records = load_records(args.metrics_dir, set(args.gconvs) if args.gconvs else None, args.gmethod)
    print(f'loaded {len(records)} run(s) from {args.metrics_dir}')

    grouped = defaultdict(list)
    for record in records:
        grouped[cell_key(record['config'])].append(record)

    rows = []
    for key, runs in grouped.items():
        seeds = sorted(r['config']['seed'] for r in runs if r['config']['seed'] is not None)
        cell = dict(zip(CELL, key))
        cell['runs'] = len(runs)
        cell['seeds'] = ','.join(str(s) for s in seeds) or '-'
        for label, mode, needle in (
            ('patient_auc', PATIENT_TEST, 'AUC'),
            ('patient_bacc', PATIENT_TEST, 'Balanced Accuracy'),
            ('roi_auc', ROI_TEST, 'AUC'),
            ('roi_bacc', ROI_TEST, 'Balanced Accuracy'),
        ):
            mean, sd, n = mean_sd([final(r, mode, needle) for r in runs])
            cell[f'{label}_mean'] = mean
            cell[f'{label}_sd'] = sd
            cell[f'{label}_n'] = n
        rows.append(cell)

    rows.sort(key=lambda r: (r['patient_auc_mean'] is None,
                             -(r['patient_auc_mean'] or 0)))

    header = ('| Variant | Graph | k | Node features | Runs | Seeds | Patient AUC | '
              'Patient Bal.Acc | ROI AUC | ROI Bal.Acc |')
    lines = [
        '# GNN variant comparison',
        '',
        'Test set evaluated once at the final epoch. Mean +/- SD across seeds.',
        '',
        header,
        '|---|---|---:|---|---:|---|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append(
            f"| {r['gconv']} | {r['gmethod']} | {r['k']} | {r['node_f']} | {r['runs']} | {r['seeds']} | "
            f"{fmt(r['patient_auc_mean'], r['patient_auc_sd'], r['patient_auc_n'])} | "
            f"{fmt(r['patient_bacc_mean'], r['patient_bacc_sd'], r['patient_bacc_n'])} | "
            f"{fmt(r['roi_auc_mean'], r['roi_auc_sd'], r['roi_auc_n'])} | "
            f"{fmt(r['roi_bacc_mean'], r['roi_bacc_sd'], r['roi_bacc_n'])} |"
        )
    lines += [
        '',
        'Splits are patient-grouped and label-stratified (mainutils.utils.train_test_split '
        'splits unique patients, then assigns ROIs by membership), so no patient contributes '
        'ROIs to both sides. ROI AUC is therefore a clean estimate; the patient-level columns '
        'are still the clinically meaningful ones but rest on only ~18 test patients.',
    ]
    table = '\n'.join(lines)
    print()
    print(table)

    if args.out_prefix and rows:
        with open(f'{args.out_prefix}.md', 'w') as handle:
            handle.write(table + '\n')
        with open(f'{args.out_prefix}.csv', 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nwrote {args.out_prefix}.md and {args.out_prefix}.csv')

    # Per-run detail, so a single bad seed is visible rather than averaged away.
    print('\nper-run detail:')
    print(f"{'gconv':8s} {'graph':9s} {'k':>2s} {'node_f':22s} {'seed':>5s} {'patAUC':>8s} {'patBACC':>8s} {'roiAUC':>8s}")
    for record in sorted(records, key=lambda r: (str(cell_key(r['config'])), r['config']['seed'] or 0)):
        cfg = record['config']
        pa = final(record, PATIENT_TEST, 'AUC')
        pb = final(record, PATIENT_TEST, 'Balanced Accuracy')
        ra = final(record, ROI_TEST, 'AUC')
        f = lambda v: f'{v:8.3f}' if v is not None else '     n/a'
        print(f"{str(cfg['gconv']):8s} {str(cfg['gmethod']):9s} {str(cfg['k']):>2s} {str(cfg['node_f']):22s} "
              f"{str(cfg['seed']):>5s} {f(pa)} {f(pb)} {f(ra)}")


if __name__ == '__main__':
    main()
