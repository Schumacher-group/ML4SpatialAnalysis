"""Collect the rerun outputs into one markdown table for the revised Table 2.

Rows are listed in a fixed order (baselines first), not sorted by AUC, so the
table reads as a pre-specified comparison rather than a leaderboard. Every model
that was run is reported, whatever its score.
"""

import argparse
import csv
import json
from pathlib import Path

ORDER = [
    ("prevalence", "Prevalence (no information)"),
    ("roi_count", "ROI count only (confound)"),
    ("stain_batch", "Stain batch only (confound)"),
    ("summary_logistic", "Logistic regression, patient summaries (non-spatial)"),
    ("summary_random_forest", "Random forest, patient summaries (non-spatial)"),
    ("summary_xgboost", "XGBoost, patient summaries (non-spatial)"),
    ("spatial_logistic", "Logistic regression with cell-type contacts (spatial)"),
    ("deepsets", "DeepSets: GNN encoder without edges"),
    ("gnn", "Patient-level GNN"),
    ("gnn_edge_shuffle", "Patient-level GNN, edges shuffled (control)"),
]
MODES = {"none": "-", "expressions": "markers", "celltypes": "cell types", "combined": "markers + cell types"}


def read_rows(directory):
    path = Path(directory) / "comparison.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        return {(row["model"], row["feature_mode"]): row for row in csv.DictReader(handle)}


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fmt(row):
    if not row:
        return "not run"
    return (f"{float(row['repeat_auc_mean']):.2f} "
            f"({float(row['oof_auc_ci_low']):.2f} to {float(row['oof_auc_ci_high']):.2f})")


def cohort_line(report, label):
    if not report:
        return f"- {label}: not run\n"
    patients = report.get("patients_by_label", {})
    manifest = report.get("cache_manifest", {})
    line = (f"- {label}: {sum(patients.values())} patients "
            f"({patients.get('responder', 0)} R / {patients.get('non_responder', 0)} NR), "
            f"{manifest.get('retained_rois', '?')} ROIs")
    batch = report.get("single_class_batch_filter")
    if batch:
        line += (f"; after dropping single-class stain batches {batch['dropped_batches']}: "
                 f"{batch['patients_after']} patients ({batch['responders_after']} R / "
                 f"{batch['non_responders_after']} NR), {batch['rois_after']} ROIs")
    return line + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-dir", default="artifacts/rerun")
    parser.add_argument("--out", default="artifacts/rerun/table2.md")
    args = parser.parse_args()
    root = Path(args.rerun_dir)

    full = read_rows(root / "patient_benchmark")
    balanced = read_rows(root / "patient_benchmark_balanced")
    lines = ["# Revised Table 2: patient-level prediction of chemotherapy response\n\n",
             "Patient-level ROC AUC, mean over 5 repeats of stratified 5-fold cross-validation "
             "(every patient scored out-of-fold in each repeat), with the 95% patient-bootstrap "
             "interval averaged over repeats.\n\n"]
    lines.append(cohort_line(read_json(root / "patient_benchmark" / "cohort_report.json"), "Full cohort"))
    lines.append(cohort_line(read_json(root / "patient_benchmark_balanced" / "cohort_report.json"),
                             "Sensitivity cohort"))
    lines += ["\n| Model | Input | AUC, full cohort | AUC, mixed-response batches only |\n",
              "|---|---|---:|---:|\n"]
    seen = set()
    for model, label in ORDER:
        for mode in ("none", "expressions", "celltypes", "combined"):
            key = (model, mode)
            if key in full or key in balanced:
                seen.add(key)
                lines.append(f"| {label} | {MODES[mode]} | {fmt(full.get(key))} | {fmt(balanced.get(key))} |\n")
    for key in sorted(set(full) | set(balanced)):
        if key not in seen:
            lines.append(f"| {key[0]} | {MODES.get(key[1], key[1])} | {fmt(full.get(key))} | {fmt(balanced.get(key))} |\n")

    comparison = read_json(root / "patient_benchmark" / "comparison.json")
    if comparison and comparison.get("paired_auc_differences_vs_gnn_combined"):
        lines += ["\n## Paired AUC difference, patient-level GNN (markers + cell types) minus comparator\n\n",
                  "Mean over repeats of the observed difference; 95% paired patient-bootstrap interval averaged over repeats.\n\n",
                  "| Comparator | Input | Difference (95% CI) |\n|---|---|---:|\n"]
        for name, repeats in sorted(comparison["paired_auc_differences_vs_gnn_combined"].items()):
            if not repeats:
                continue
            model, mode = name.split(":")
            mean = sum(r["auc_difference"] for r in repeats) / len(repeats)
            low = sum(r["ci"][0] for r in repeats) / len(repeats)
            high = sum(r["ci"][1] for r in repeats) / len(repeats)
            lines.append(f"| {model} | {MODES.get(mode, mode)} | {mean:+.2f} ({low:+.2f} to {high:+.2f}) |\n")

    lines.append("\n## Original ROI-level GCN (manuscript architecture), patient-grouped 5-fold CV x 5 repeats\n\n")
    lines.append("ROI count and stain batch here are oriented using the test labels, so they are upper bounds; "
                 "use the train-fitted confound baselines above for the comparison.\n\n")
    lines.append("| Model | AUC, full cohort | AUC, mixed-response batches only |\n|---|---:|---:|\n")
    gcn_full = (read_json(root / "gcn_grouped_cv.json") or {}).get("results", {})
    gcn_bal = (read_json(root / "gcn_grouped_cv_balanced.json") or {}).get("results", {})

    def fmt_cv(entry):
        if not entry:
            return "not run"
        return f"{entry['auc_mean']:.2f} ± {entry['auc_sd']:.2f} ({entry['ci_low']:.2f} to {entry['ci_high']:.2f})"

    for name in sorted(set(gcn_full) | set(gcn_bal)):
        lines.append(f"| {name} | {fmt_cv(gcn_full.get(name))} | {fmt_cv(gcn_bal.get(name))} |\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(lines), encoding="utf-8")
    print("".join(lines))


if __name__ == "__main__":
    main()
