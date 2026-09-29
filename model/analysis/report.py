"""Generate thesis summaries, tables and figures with one command.

Examples:
    python model/analysis/report.py model/data/thesis_experiments/final_main_v7
    python model/analysis/report.py SUITE --sections summary --output-dir SUITE
    python model/analysis/report.py SUITE --sections comparison friedman --exclude-dataset new_thyroid

Default outputs go into SUITE/report, leaving archived thesis evidence untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/thesis-matplotlib")
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import analysis as a
from analysis import ABLATION_METRICS, TRANSITION_METRICS, METHOD_LABELS, COMPONENT_LABELS

DISPLAY_NAMES = {
    "phase_accuracy__transition": "Transition accuracy",
    "phase_accuracy__early_recovery": "Early-recovery accuracy",
    "phase_accuracy__stable_post_change": "Stable accuracy",
    "phase_gmean__stable_post_change": "Stable G-Mean",
    "phase_pr_auc_macro__stable_post_change": "Stable macro PR-AUC",
    "phase_rec_min__stable_post_change": "Stable minority recall",
    "phase_f1_min__stable_post_change": "Stable minority F1",
    "phase_pr_auc_min__stable_post_change": "Stable minority PR-AUC",
    "accuracy_recovery_time": "Recovery observations",
    "inference_latency_mean_seconds": "Inference latency (ms)",
    "training_update_latency_mean_seconds": "Update latency (ms)",
    "process_rss_peak_bytes": "Peak RSS (MB)",
    "persistent_model_bytes": "Persistent size (KB)",
    "prototype_vector_bytes_at_end": "Prototype vectors (KB)",
    "s2_active_parameter_count": "Active S2 parameters",
}


def format_mean_sd(mean: float, std: float, decimals: int = 3) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def write_table(table: pd.DataFrame, output_dir: Path, stem: str, manifest: list, source: Path) -> None:
    csv_path = output_dir / f"{stem}.csv"
    tex_path = output_dir / f"{stem}.tex"
    table.to_csv(csv_path, index=False)
    tex_path.write_text(
        table.to_latex(index=False, escape=True, na_rep="NA"), encoding="utf-8"
    )
    manifest.append(
        {
            "table": stem,
            "csv": str(csv_path.resolve()),
            "latex": str(tex_path.resolve()),
            "source": str(source.resolve()),
        }
    )

def findings_text(suite: Path, runs: pd.DataFrame, paired: pd.DataFrame, audited: bool) -> str:
    datasets = list(dict.fromkeys(runs.dataset))
    methods = list(dict.fromkeys(runs.method))
    lines = [
        "# Feature-Transition Adaptation Analysis",
        "",
        "## Evidence status",
        "",
        f"- Suite: `{suite.name}`.",
        f"- Audit status: **{'passed' if audited else 'incomplete/provisional'}**.",
        "- The audit checks run artifacts and protocol fields; dataset-label semantics require separate review.",
        f"- Available archives: {runs.source_npz.nunique()}; datasets: {len(datasets)}; methods/conditions: {len(methods)}.",
        "- Results are centred on the declared feature-space transition, not averaged over the entire stream.",
        "- Recovery is the first point at which the rolling metric reaches 95% of its pre-transition level and remains there for max(10, 10% of the evaluation window) observations.",
        "- An unrecovered metric is retained as missing recovery time and separately counted by the recovery rate.",
        "- The two-window deficit area is reported for every run, allowing fair comparison when recovery is not observed.",
        "",
        "## Thesis interpretation",
        "",
    ]
    if not audited:
        lines.append("This suite is not complete and must not be used for final claims. The generated results are a pipeline check only.")
        return "\n".join(lines) + "\n"
    key = paired[paired.measure.isin(["transition_loss", "recovery_time_95", "deficit_area_2w", "stable_level"])] if not paired.empty else paired
    if key.empty:
        lines.append("No paired proposed-model comparison was available.")
    else:
        for dataset in datasets:
            subset = key[key.dataset == dataset]
            wins = int((subset.candidate_advantage < 0).sum())  # proposed better
            losses = int((subset.candidate_advantage > 0).sum())
            lines.append(f"- `{dataset}`: across the reported candidate comparisons, the proposed method had the favorable direction in {wins} cells and the unfavorable direction in {losses}; inspect effect sizes and intervals in `paired_comparisons.csv` before claiming an advantage.")
    lines.extend([
        "",
        "Do not convert directional counts into a superiority claim. The defensible story should identify which methods and metrics show smaller loss, faster maintained recovery, smaller deficit area, or a higher stable level, and connect those outcomes to the matching ablation evidence.",
    ])
    return "\n".join(lines) + "\n"

def findings_markdown(rows: pd.DataFrame, audit_path: Path) -> str:
    lines = [
        "# Cross-Drift Ablation Findings",
        "",
        "## Evidence status",
        "",
        f"- Audit source: `{audit_path}`.",
        f"- Included comparisons: {rows.dataset.nunique()} datasets and {rows.condition.nunique()} component removals. Paired seed counts are recorded per comparison.",
        "- Positive contribution means the full model performed better than its ablation. For recovery time, positive means the ablation needed more observations to recover.",
        "- Holm adjustment is applied across every dataset × component × reported-metric paired test.",
        "",
        "## Component summary across datasets",
        "",
        "The following statements describe cross-dataset patterns; dataset-specific values remain in `paired_component_effects.csv`.",
        "",
    ]
    for condition, component in COMPONENT_LABELS.items():
        subset = rows[rows.condition == condition]
        if subset.empty:
            continue
        stable = subset[subset.metric == "stable_accuracy"]
        transition = subset[subset.metric == "transition_accuracy"]
        stable_help = int((stable.contribution_ci95_low > 0).sum())
        stable_harm = int((stable.contribution_ci95_high < 0).sum())
        transition_help = int((transition.contribution_ci95_low > 0).sum())
        transition_harm = int((transition.contribution_ci95_high < 0).sum())
        strongest = transition.loc[transition.component_contribution.idxmax()]
        weakest = transition.loc[transition.component_contribution.idxmin()]
        lines.extend(
            [
                f"### {component}",
                "",
                f"Across {subset.dataset.nunique()} streams, the transition-accuracy interval favored retaining this component in **{transition_help}** datasets and favored its removal in **{transition_harm}**. For stable accuracy, the corresponding counts were **{stable_help}** and **{stable_harm}**.",
                f"Its largest transition contribution was {strongest.component_contribution:+.4f} on `{strongest.dataset}`; its smallest was {weakest.component_contribution:+.4f} on `{weakest.dataset}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation rules for the thesis",
            "",
            "- Do not call a component universally useful unless its direction is consistent across drift types.",
            "- A transition benefit with no stable benefit means the component supplies a jump-start rather than a better final plateau.",
            "- A negative contribution is retained as evidence that the component can be unnecessary or harmful in that stream.",
            "- Small paired-seed samples provide limited power; discuss effect sizes and confidence intervals, not only adjusted p-values.",
        ]
    )
    return "\n".join(lines) + "\n"


# Tables consume the same seed aggregates used by plots.

PREDICTIVE_METRICS = (
    "phase_accuracy__transition", "phase_accuracy__early_recovery",
    "phase_accuracy__stable_post_change", "phase_gmean__stable_post_change",
    "phase_pr_auc_macro__stable_post_change", "phase_rec_min__stable_post_change",
    "phase_f1_min__stable_post_change", "phase_pr_auc_min__stable_post_change",
    "accuracy_recovery_time",
)
RESOURCE_METRICS = (
    "inference_latency_mean_seconds", "training_update_latency_mean_seconds",
    "process_rss_peak_bytes", "persistent_model_bytes",
    "prototype_vector_bytes_at_end", "s2_active_parameter_count",
)
DETECTOR_METRICS = ["detector_matched_changes", "detector_missed_changes", "detector_false_alarms", "detector_mean_delay"]


def result_table(aggregate, metrics):
    rows = []
    for experiment, subset in aggregate.groupby("experiment", sort=False):
        row = {"Method": experiment}
        subset = subset.set_index("metric")
        for metric in metrics:
            if metric not in subset.index:
                continue
            mean, sd = subset.loc[metric, ["mean", "std"]]
            scale = 1000 if metric.endswith("latency_mean_seconds") else 1
            if metric == "process_rss_peak_bytes":
                scale = 1 / 1024**2
            elif metric in ("persistent_model_bytes", "prototype_vector_bytes_at_end"):
                scale = 1 / 1024
            row[DISPLAY_NAMES.get(metric, metric)] = format_mean_sd(mean*scale, sd*scale)
        rows.append(row)
    return pd.DataFrame(rows)


def export_tables(aggregate, paired, output, source):
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for metrics, name in ((PREDICTIVE_METRICS, "main_predictive_results"), (RESOURCE_METRICS, "computational_results")):
        write_table(result_table(aggregate, metrics), output, name, manifest, source)
    if not paired.empty:
        selected = paired[paired.metric.isin(PREDICTIVE_METRICS)].rename(columns={"metric": "Outcome", "candidate_experiment": "Comparison"})
        write_table(selected, output, "paired_statistical_tests", manifest, source)
    detectors = aggregate[aggregate.metric.isin(DETECTOR_METRICS)]
    if not detectors.empty:
        write_table(result_table(detectors, DETECTOR_METRICS), output, "detector_results", manifest, source)
    write_json(output / "table_manifest.json", {"tables": manifest, "format": "mean ± sample SD across finite run seeds"})


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def save_frames(output, frames):
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)


def export_transition(suite, output, runs, recovery_fraction, audited, exclusions, figures):
    run_metrics = a.transition_runs(runs, recovery_fraction)
    aggregate = a.aggregate_runs(run_metrics)
    paired = a.paired_method_comparisons(run_metrics)
    save_frames(output / "tables", {"run_level_metrics": run_metrics, "aggregate_metrics": aggregate, "paired_comparisons": paired})
    for measure in ("transition_loss", "recovery_time_95", "deficit_area_2w", "stable_level"):
        subset = aggregate[aggregate.measure == measure].copy()
        subset["value"] = subset.apply(lambda row: f"{row['mean']:.3f} [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]", axis=1)
        table = subset.pivot_table(index=["dataset", "metric"], columns="method", values="value", aggfunc="first")
        table.to_csv(output / "tables" / f"{measure}.csv")
        (output / "tables" / f"{measure}.tex").write_text(table.to_latex(escape=True, na_rep="--"))
    if figures:
        import plots
        plots.transition_figures(a.transition_curves(runs), output / "figures", [])
    notes = findings_text(suite, run_metrics, paired, audited)
    notes = notes.replace("95%", f"{100*recovery_fraction:g}%")
    if exclusions:
        notes += "\nExcluded datasets: " + ", ".join(exclusions) + ".\n"
    (output / "findings.md").write_text(notes)


def export_ablation(suite, output, summary, figures):
    rows = pd.DataFrame(a.ablation_comparisons(summary))
    if rows.empty:
        raise ValueError("No paired component-removal comparisons found")
    save_frames(output / "tables", {"paired_component_effects": rows})
    (output / "figures").mkdir(exist_ok=True)
    for metric, (_, label, _) in ABLATION_METRICS.items():
        subset = rows[rows.metric == metric]
        pivot = subset.pivot(index="component_tested", columns="dataset", values="component_contribution")
        stem = output / "tables" / f"component_contribution__{metric}"
        pivot.to_csv(stem.with_suffix(".csv"))
        stem.with_suffix(".tex").write_text(pivot.to_latex(float_format="%.4f"))
        if figures:
            import plots
            plots.heatmap(pivot, f"Component contribution by stream: {label}", output / "figures" / f"heatmap__{metric}")
            plots.bar_panels(subset, f"Ablation contribution by stream: {label}", output / "figures" / f"bars__{metric}")
    (output / "cross_drift_findings.md").write_text(findings_markdown(rows, suite / "audit_report.json"))


def export_comparison(frame, curves, output, figures):
    save_frames(output / "tables", {"all_metrics_mean_sd": frame})
    primary = frame[["dataset", "method"]].copy()
    for metric in ("accuracy", "gmean", "rec_min", "f1_min", "pr_auc"):
        primary[metric] = frame.apply(
            lambda row: f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_sd']:.3f}", axis=1)
    primary.to_csv(output / "tables" / "overall_prequential_performance.csv", index=False)
    (output / "tables" / "overall_prequential_performance.tex").write_text(
        primary.to_latex(index=False, escape=True,
                         caption="Overall prequential performance (mean and sample standard deviation across seeds)."))
    for metric in a.COMPARISON_METRICS:
        display = frame.copy()
        display["value"] = display.apply(lambda row: f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_sd']:.3f}", axis=1)
        wide = display.pivot(index="dataset", columns="method", values="value")
        wide = wide.reindex(index=list(dict.fromkeys(frame.dataset)), columns=list(dict.fromkeys(frame.method)))
        wide.to_csv(output / "tables" / f"{metric}_comparison.csv")
        (output / "tables" / f"{metric}_comparison.tex").write_text(wide.to_latex(escape=True, na_rep="--"))
    if figures:
        import plots
        plots.dataset_figures(curves, output / "figures", [])


def export_friedman(frame, output, figures=True):
    omnibus, ranks, posthoc = a.friedman_comparisons(frame)
    save_frames(output, {"friedman_omnibus": omnibus, "average_ranks": ranks, "proposed_vs_baselines_wilcoxon_holm": posthoc})
    critical, pairs = a.nemenyi_comparisons(omnibus, ranks)
    save_frames(output, {"nemenyi_critical_differences": critical, "nemenyi_all_pairs": pairs})
    if figures:
        import plots
        plots.configure_style()
        plots.nemenyi_diagrams(ranks, critical, output / "figures")
    significant = posthoc[posthoc.significantly_better_at_0_05]
    notes = [
        "# Friedman comparison", "",
        f"Evidence: {frame.dataset.nunique()} datasets, {frame.method.nunique()} methods. For multi-location runs, locations are averaged within each seed first. Seeds are then averaged within each dataset–method pair; locations are not additional dataset blocks.", "",
        "ACR is minimized; the other metrics are maximized. Post-hoc comparisons use one-sided Wilcoxon tests with Pratt zeros and Holm correction within each metric. Superiority requires a significant Friedman omnibus test, adjusted p < 0.05, and a positive median oriented difference.", "",
        "Average rank 1 is best. Missing dataset–method pairs are rejected. A non-significant result does not establish equality.", "",
        f"Comparisons meeting the full superiority rule: {len(significant)}. See the CSV tables for all estimates, effect sizes and p-values.", "",
        "Nemenyi diagrams compare all pairs using CD = q_alpha sqrt(k(k+1)/(6N)), alpha=0.05. The Studentized-range quantile is divided by sqrt(2). Bars connect maximal groups whose rank differences do not exceed CD; they do not establish equivalence. Pairwise significance claims also require the Friedman gate. Diagrams remain descriptive when that gate fails. Nemenyi p-values already control all-pairs multiplicity within each metric; no additional Holm correction is applied to them.", "",
        "Related INSECTS variants are not independent replications of unrelated datasets. Interpret cross-variant p-values cautiously; seeds are averaged and are not extra dataset blocks. The tests are asymptotic, with limited power on small dataset collections.", "",
        "Reference: https://jmlr.org/papers/v7/demsar06a.html (Section 3.2.2, Table 5a).", "",
        "These results apply to the included datasets and do not establish superiority on all streaming problems.",
    ]
    (output / "FRIEDMAN_TEST_THESIS_NOTES.md").write_text("\n".join(notes) + "\n")


def export_figures(runs, aggregate, output, summary_path, experiment=None, seed=None, explainability_dir=None):
    import plots
    plots.configure_style()
    selected = [run for run in runs if (experiment is None or run.experiment == experiment) and (seed is None or run.seed == seed)]
    if not selected:
        raise ValueError(f"No run matches experiment={experiment!r}, seed={seed!r}")
    run = selected[0]
    manifest = []
    plots.time_series_figures(run.data, run.info, output, manifest, run.path, a.diagnostic_accuracy(run))
    plots.router_figure(run.data, run.info, output, manifest, run.path)
    plots.prototype_figure(run.data, run.info, output, manifest, run.path)
    plots.comparison_figures(aggregate, output, manifest, summary_path)
    plots.sensitivity_figure(aggregate, output, manifest, summary_path)
    if explainability_dir is not None:
        source = explainability_dir / "shap_global_importance_all_snapshots.csv"
        pivot = a.shap_importance_table(pd.read_csv(source))
        plots.explainability_figure(pivot, output, manifest, source)
    write_json(output / "figure_manifest.json", {"experiment": run.experiment, "seed": run.seed, "figures": manifest})


def default_sections(runs):
    sections = ["summary", "tables", "figures", "transition"]
    conditions = {a.split_experiment(run.experiment)[1] for run in runs}
    if all("__" in run.experiment for run in runs):
        if conditions & set(a.COMPONENT_LABELS):
            sections.append("ablation")
        else:
            sections.append("comparison")
            if len({a.split_experiment(run.experiment)[0] for run in runs}) >= 3 and len(conditions) >= 3 and "full_model" in conditions:
                sections.append("friedman")
    return sections


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Defaults to SUITE/report")
    parser.add_argument("--sections", nargs="+", choices=("summary", "tables", "figures", "transition", "ablation", "comparison", "friedman", "audit"))
    parser.add_argument("--exclude-dataset", action="append", default=[])
    parser.add_argument("--reference")
    parser.add_argument("--experiment", help="Run used for diagnostic curves; defaults to full_model or the first run")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--explainability-dir", type=Path)
    parser.add_argument("--config", type=Path, help="Frozen configuration to audit; otherwise use the suite manifest")
    parser.add_argument("--partial", action="store_true", help="Write only summary_partial.csv")
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow provisional transition/ablation outputs without a passing audit")
    parser.add_argument("--recovery-fraction", type=float, default=.95)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.recovery_fraction <= 1:
        parser.error("--recovery-fraction must be in (0, 1]")
    suite = args.suite_dir.resolve()
    output = (args.output_dir or suite / "report").resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = suite / "manifest.json"
    config = json.loads(manifest_path.read_text()).get("config", {}) if manifest_path.exists() else {}
    sections = args.sections
    if sections and "audit" in sections:
        config_path = args.config
        if config_path is None:
            if not config:
                parser.error("Audit requires --config or a suite manifest containing config")
            config_path = output / "audit_config.json"
            write_json(config_path, config)
        audit = a.audit_suite(config_path, suite)
        write_json(output / "audit_report.json", audit)
        print(f"Audit passed: {audit['complete_and_valid']}")
        if not audit["complete_and_valid"]:
            return 1
        sections = [section for section in sections if section != "audit"]
        if not sections:
            return 0
    runs = a.load_runs(suite, args.exclude_dataset)
    sections = ["summary"] if args.partial else sections or default_sections(runs)
    audit_path = suite / "audit_report.json"
    if (output / "audit_report.json").exists():
        audit_path = output / "audit_report.json"
    audited = audit_path.exists() and bool(json.loads(audit_path.read_text()).get("complete_and_valid"))
    if {"transition", "ablation", "comparison", "friedman"} & set(sections) and not audited and not args.allow_incomplete:
        parser.error("Final comparison/transition/ablation reports require a passing audit; run --sections audit or use --allow-incomplete for provisional output")
    reference = args.reference or config.get("reference_experiment")
    experiments = {run.experiment for run in runs}
    if reference not in experiments:
        if args.reference:
            parser.error(f"Reference {reference!r} is absent from included runs")
        reference = "full_model" if "full_model" in experiments else runs[0].experiment
    needs_summary = bool(set(sections) & {"summary", "tables", "figures", "ablation"})
    if needs_summary:
        summary = a.summarize_runs(runs)
        if args.partial:
            summary.to_csv(output / "summary_partial.csv", index=False)
            print(f"Summarized {len(runs)} completed runs (partial)")
            return 0
        outcomes = config.get("analysis_outcomes", a.DEFAULT_OUTCOMES)
        aggregate_outcomes = list(dict.fromkeys([*outcomes, *DISPLAY_NAMES, *DETECTOR_METRICS]))
        aggregate, paired = a.aggregate_summary(summary, reference, outcomes, aggregate_outcomes)
        save_frames(output, {"summary": summary, "aggregate_statistics": aggregate, "paired_tests": paired})
    if "tables" in sections:
        export_tables(aggregate, paired, output / "tables", output / "summary.csv")
    if "figures" in sections and not args.no_figures:
        experiment = args.experiment or (reference if reference in set(summary.experiment) else None)
        export_figures(runs, aggregate, output / "figures", output / "summary.csv", experiment, args.seed, args.explainability_dir)
    if "transition" in sections:
        export_transition(suite, output / "feature_transition_analysis", runs, args.recovery_fraction, audited, args.exclude_dataset, not args.no_figures)
    if "ablation" in sections:
        export_ablation(suite, output / "cross_drift_analysis", summary, not args.no_figures)
    if {"comparison", "friedman"} & set(sections):
        comparison, curves = a.dataset_comparison(runs)
        if "comparison" in sections:
            export_comparison(comparison, curves, output / "comparison_package", not args.no_figures)
        if "friedman" in sections:
            dataset_frame = a.dataset_comparison_across_locations(runs)
            save_frames(output / "friedman_analysis", {"dataset_level_comparison": dataset_frame})
            export_friedman(dataset_frame, output / "friedman_analysis", not args.no_figures)
    artifacts = []
    if needs_summary:
        artifacts.extend(output / name for name in ("summary.csv", "aggregate_statistics.csv", "paired_tests.csv"))
    folders = {"tables": "tables", "figures": "figures", "transition": "feature_transition_analysis", "ablation": "cross_drift_analysis", "comparison": "comparison_package", "friedman": "friedman_analysis"}
    for section in sections:
        if section in folders:
            artifacts.extend(p for p in (output / folders[section]).rglob("*") if p.is_file())
    write_json(output / "report_manifest.json", {
        "suite": str(suite), "sections": sections, "reference": reference,
        "excluded_datasets": args.exclude_dataset, "audit_passed": audited,
        "recovery_fraction": args.recovery_fraction,
        "sources": [{"path": str(r.path), "sha256": hashlib.sha256(r.path.read_bytes()).hexdigest()} for r in runs],
        "definitions": {"phase_metrics": "exact predictions within non-overlapping phases", "transition": "saved rolling curves; sustained recovery; unreached recovery remains NaN", "dataset_comparison": "whole-stream metrics from recorded predictions per run, then mean/sample SD across paired seeds; ACR remains a supporting diagnostic", "paired_tests": "candidate minus reference; paired t and two-sided Wilcoxon; Holm across reported tests", "friedman": "dataset blocks; directional control comparisons; Holm within metric"},
        "artifacts": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts],
    })
    print(f"Generated {', '.join(sections)} for {len(runs)} runs in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
