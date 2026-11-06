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
    ("count_summary_logistic", "ROI count + patient summaries, logistic regression"),
    ("count_summary_random_forest", "ROI count + patient summaries, random forest"),
    ("count_summary_xgboost", "ROI count + patient summaries, XGBoost"),
    ("count_spatial_logistic", "ROI count + cell-type contacts, logistic regression"),
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


def fmt_stratified(row):
    if not row or not row.get("count_stratified_auc_mean"):
        return "not run"
    return (f"{float(row['count_stratified_auc_mean']):.2f} "
            f"({float(row['count_stratified_ci_low']):.2f} to "
            f"{float(row['count_stratified_ci_high']):.2f})")


def paired_rows(differences, keep=lambda model: True):
    lines = []
    for name, repeats in sorted(differences.items()):
        model, mode = name.split(":")
        if not repeats or not keep(model):
            continue
        mean = sum(r["auc_difference"] for r in repeats) / len(repeats)
        low = sum(r["ci"][0] for r in repeats) / len(repeats)
        high = sum(r["ci"][1] for r in repeats) / len(repeats)
        lines.append(f"| {model} | {MODES.get(mode, mode)} | {mean:+.2f} ({low:+.2f} to {high:+.2f}) |\n")
    return lines


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
             "interval averaged over repeats. The stratified columns average the AUCs within "
             "ROI-count tertiles (weighted by tertile size), so that a score tracking ROI count "
             "gains less; counts still vary within a tertile, so the ROI-count row shows how much "
             "of the confound is left.\n\n"]
    lines.append(cohort_line(read_json(root / "patient_benchmark" / "cohort_report.json"), "Full cohort"))
    lines.append(cohort_line(read_json(root / "patient_benchmark_balanced" / "cohort_report.json"),
                             "Sensitivity cohort"))
    lines += ["\n| Model | Input | AUC, full cohort | Count-stratified, full | "
              "AUC, mixed-response batches only | Count-stratified, mixed |\n",
              "|---|---|---:|---:|---:|---:|\n"]
    seen = set()
    for model, label in ORDER:
        for mode in ("none", "expressions", "celltypes", "combined"):
            key = (model, mode)
            if key in full or key in balanced:
                seen.add(key)
                lines.append(f"| {label} | {MODES[mode]} | {fmt(full.get(key))} | "
                             f"{fmt_stratified(full.get(key))} | {fmt(balanced.get(key))} | "
                             f"{fmt_stratified(balanced.get(key))} |\n")
    for key in sorted(set(full) | set(balanced)):
        if key not in seen:
            lines.append(f"| {key[0]} | {MODES.get(key[1], key[1])} | {fmt(full.get(key))} | "
                         f"{fmt_stratified(full.get(key))} | {fmt(balanced.get(key))} | "
                         f"{fmt_stratified(balanced.get(key))} |\n")

    comparison = read_json(root / "patient_benchmark" / "comparison.json")
    if comparison and comparison.get("paired_auc_differences_vs_gnn_combined"):
        lines += ["\n## Paired AUC difference, patient-level GNN (markers + cell types) minus comparator\n\n",
                  "Mean over repeats of the observed difference; 95% paired patient-bootstrap interval averaged over repeats.\n\n",
                  "| Comparator | Input | Difference (95% CI) |\n|---|---|---:|\n"]
        lines += paired_rows(comparison["paired_auc_differences_vs_gnn_combined"])

    comparison_balanced = read_json(root / "patient_benchmark_balanced" / "comparison.json")
    for title, data in (("full cohort", comparison), ("mixed-response batches only", comparison_balanced)):
        differences = (data or {}).get("paired_auc_differences_vs_roi_count")
        if not differences:
            continue
        lines += [f"\n## Does tissue add to ROI count? ROI count only minus comparator, {title}\n\n",
                  "Negative values mean the comparator beats the count-only baseline. The count_* "
                  "models get the same tissue features plus log(1 + ROI count), on the same folds.\n\n",
                  "| Comparator | Input | Difference (95% CI) |\n|---|---|---:|\n"]
        lines += paired_rows(differences, keep=lambda model: model.startswith("count_"))

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
