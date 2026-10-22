"""Build and stage all Act 8 thesis figures and tables from verified outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SYSTEM_ORDER = ["stardist_saved", "cellpose_auto", "cellpose_fixed"]
DISPLAY_NAMES = {
    "stardist_saved": "StarDist",
    "cellpose_auto": "Cellpose automatic",
    "cellpose_fixed": "Cellpose fixed",
}
COLORS = {
    "stardist_saved": "#0072B2",
    "cellpose_auto": "#009E73",
    "cellpose_fixed": "#E69F00",
}
HATCHES = {
    "stardist_saved": "",
    "cellpose_auto": "///",
    "cellpose_fixed": "xx",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def verify_indexed_sources(results_root: Path, required: Sequence[Path]) -> None:
    index = read_json(results_root / "results_index.json")
    indexed = {
        (results_root / item["relative_path"]).resolve(): item["sha256"]
        for item in index["sources"]
    }
    for path in required:
        resolved = path.resolve()
        expected = indexed.get(resolved)
        if expected is None:
            raise ValueError(f"Source is not indexed: {resolved}")
        actual = sha256_file(resolved)
        if actual != expected:
            raise ValueError(f"Indexed hash mismatch for {resolved}: {actual} != {expected}")


def verify_manifest_output(manifest_path: Path, source: Path) -> None:
    manifest = read_json(manifest_path)
    candidates = {Path(item["path"]).resolve(): item["sha256"] for item in manifest["outputs"]}
    expected = candidates.get(source.resolve())
    if expected is None:
        raise ValueError(f"{source} is not recorded in {manifest_path}")
    actual = sha256_file(source)
    if actual != expected:
        raise ValueError(f"Figure hash mismatch for {source}: {actual} != {expected}")


def save_figure(fig: mpl.figure.Figure, base: Path) -> list[Path]:
    pdf = base.with_suffix(".pdf")
    png = base.with_suffix(".png")
    metadata = {
        "Creator": "scripts_figures/build_thesis_assets.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf, bbox_inches="tight", metadata=metadata)
    fig.savefig(png, dpi=300, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return [pdf, png]


def label_bars(axis: mpl.axes.Axes, bars: Any, fmt: str = ".3f", offset: float = 0.015) -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            format(value, fmt),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_detection(comparison: pd.DataFrame, figures_dir: Path) -> list[Path]:
    data = comparison.set_index("system").loc[SYSTEM_ORDER]
    x = np.arange(len(SYSTEM_ORDER))
    width = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)

    for offset, (column, label, color, hatch) in zip(
        (-width, 0, width),
        (
            ("pooled_precision", "Precision", "#4C78A8", ""),
            ("pooled_recall", "Recall", "#F58518", "///"),
            ("pooled_f1", "F1", "#54A24B", "xx"),
        ),
    ):
        bars = axes[0].bar(
            x + offset,
            data[column],
            width,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.5,
        )
        label_bars(axes[0], bars)
    axes[0].set_title("Detection coverage")
    axes[0].set_ylabel("Score")
    axes[0].set_xticks(
        x,
        [DISPLAY_NAMES[item] for item in SYSTEM_ORDER],
        rotation=15,
        ha="right",
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, ncols=3, fontsize=8, loc="upper center")

    for offset, (column, label, color, hatch) in zip(
        (-width / 2, width / 2),
        (
            ("pooled_pq", "Panoptic quality", "#B279A2", "///"),
            ("pooled_matched_iou", "Matched-instance IoU", "#72B7B2", ""),
        ),
    ):
        bars = axes[1].bar(
            x + offset,
            data[column],
            width,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.5,
        )
        label_bars(axes[1], bars)
    axes[1].set_title("Combined quality vs conditional overlap")
    axes[1].set_xticks(
        x,
        [DISPLAY_NAMES[item] for item in SYSTEM_ORDER],
        rotation=15,
        ha="right",
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(frameon=False, fontsize=8, loc="upper center")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    fig.suptitle(r"MI05 instance-segmentation performance at IoU $\geq$ 0.5", fontsize=12)
    return save_figure(fig, figures_dir / "mi05_detection_comparison")


def plot_classification(comparison: pd.DataFrame, figures_dir: Path) -> list[Path]:
    data = comparison.set_index("system").loc[SYSTEM_ORDER]
    x = np.arange(len(SYSTEM_ORDER))
    width = 0.28
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)

    for offset, (column, label, color, hatch) in zip(
        (-width / 2, width / 2),
        (
            ("classification_only_macro_f1", "Matched detections only", "#4C78A8", ""),
            ("end_to_end_macro_f1", "End-to-end", "#E45756", "///"),
        ),
    ):
        bars = axes[0].bar(
            x + offset,
            data[column],
            width,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.5,
        )
        label_bars(axes[0], bars)
    axes[0].set_title("Macro F1")
    axes[0].set_ylabel("F1 score")
    axes[0].set_xticks(
        x,
        [DISPLAY_NAMES[item] for item in SYSTEM_ORDER],
        rotation=15,
        ha="right",
    )
    axes[0].legend(frameon=False, fontsize=8)

    for offset, (column, label, color, hatch) in zip(
        (-width / 2, width / 2),
        (
            ("end_to_end_interphase_f1", "Interphase", "#56B4E9", ""),
            ("end_to_end_mitotic_f1", "Mitotic", "#CC79A7", "///"),
        ),
    ):
        bars = axes[1].bar(
            x + offset,
            data[column],
            width,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.5,
        )
        label_bars(axes[1], bars)
    axes[1].set_title("End-to-end annotation-class F1")
    axes[1].set_xticks(
        x,
        [DISPLAY_NAMES[item] for item in SYSTEM_ORDER],
        rotation=15,
        ha="right",
    )
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    fig.suptitle(
        "MI05 downstream class recovery using the shared locked Random Forest",
        fontsize=12,
    )
    return save_figure(fig, figures_dir / "mi05_classification_comparison")


def format_score(value: float) -> str:
    return f"{value:.3f}"


def build_tables(
    comparison: pd.DataFrame,
    classifier: pd.DataFrame,
    dataset: dict[str, Any],
    bootstrap: dict[str, Any],
    agreement: pd.DataFrame,
    tables_dir: Path,
) -> list[Path]:
    outputs: list[Path] = []
    roles = {
        "MI01": "Classifier training",
        "MI02": "Classifier training",
        "MI03": "Classifier training",
        "MI04": "Classifier validation and lock",
        "MI05": "Retrospective final comparison",
    }
    rows = [
        f"{sequence} & {dataset['counts'][sequence]['image']} & "
        f"{dataset['counts'][sequence]['instance_mask']} & "
        f"{dataset['counts'][sequence]['aligned_core']} & "
        f"{roles[sequence]} \\\\"
        for sequence in dataset["sequences"]
    ]
    text = "\n".join(
        [
            r"\begin{tabularx}{\textwidth}{l S[table-format=3.0] S[table-format=3.0] S[table-format=3.0] X}",
            r"\toprule",
            r"Sequence & \multicolumn{1}{c}{Images} & \multicolumn{1}{c}{Instance masks} & \multicolumn{1}{c}{Aligned} & Role \\",
            r"\midrule",
            *rows,
            r"\midrule",
            f"Total & 343 & 345 & {dataset['total_aligned_images']} & "
            r"Two MI01 masks are annotation-only and excluded from aligned analyses. \\",
            r"\bottomrule",
            r"\end{tabularx}",
        ]
    )
    path = tables_dir / "dataset_split_table.tex"
    write_text(path, text)
    outputs.append(path)

    item = classifier.iloc[0]
    text = "\n".join(
        [
            r"\begin{tabular}{l S[table-format=4.3]}",
            r"\toprule",
            r"Quantity & \multicolumn{1}{c}{Value} \\",
            r"\midrule",
            f"Training instances (MI01--MI03) & {int(item['training_rows'])} \\\\",
            f"Validation instances (MI04) & {int(item['validation_rows'])} \\\\",
            f"Accuracy & {format_score(item['accuracy'])} \\\\",
            f"Balanced accuracy & {format_score(item['balanced_accuracy'])} \\\\",
            f"Macro F1 & {format_score(item['macro_f1'])} \\\\",
            f"Interphase F1 & {format_score(item['interphase_f1'])} \\\\",
            f"Mitotic F1 & {format_score(item['mitotic_f1'])} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    path = tables_dir / "classifier_validation_table.tex"
    write_text(path, text)
    outputs.append(path)

    by_system = comparison.set_index("system").loc[SYSTEM_ORDER]
    rows = []
    for system, row in by_system.iterrows():
        rows.append(
            f"{DISPLAY_NAMES[system]} & {format_score(row['pooled_precision'])} & "
            f"{format_score(row['pooled_recall'])} & {format_score(row['pooled_f1'])} & "
            f"{format_score(row['pooled_pq'])} & {format_score(row['pooled_matched_iou'])} \\\\"
        )
    text = "\n".join(
        [
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l *{5}{S[table-format=1.3]}}",
            r"\toprule",
            r"System & \multicolumn{1}{c}{Precision} & \multicolumn{1}{c}{Recall} & \multicolumn{1}{c}{F1} & \multicolumn{1}{c}{PQ} & \multicolumn{1}{c}{Matched IoU} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular*}",
        ]
    )
    path = tables_dir / "mi05_detection_table.tex"
    write_text(path, text)
    outputs.append(path)

    rows = []
    for system, row in by_system.iterrows():
        rows.append(
            f"{DISPLAY_NAMES[system]} & {format_score(row['classification_only_macro_f1'])} & "
            f"{format_score(row['end_to_end_macro_f1'])} & "
            f"{format_score(row['end_to_end_interphase_f1'])} & "
            f"{format_score(row['end_to_end_mitotic_f1'])} \\\\"
        )
    text = "\n".join(
        [
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l *{4}{S[table-format=1.3]}}",
            r"\toprule",
            r"System & \multicolumn{1}{c}{\shortstack{Matched-only\\macro F1}} & \multicolumn{1}{c}{\shortstack{End-to-end\\macro F1}} & \multicolumn{1}{c}{\shortstack{End-to-end\\interphase F1}} & \multicolumn{1}{c}{\shortstack{End-to-end\\mitotic F1}} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular*}",
        ]
    )
    path = tables_dir / "mi05_classification_table.tex"
    write_text(path, text)
    outputs.append(path)

    agreement_by_system = agreement.set_index("system").loc[SYSTEM_ORDER]
    analytics_display_names = {
        "stardist_saved": "StarDist",
        "cellpose_auto": r"\shortstack[l]{Cellpose\\automatic}",
        "cellpose_fixed": r"\shortstack[l]{Cellpose\\fixed}",
    }
    rows = []
    for system, row in agreement_by_system.iterrows():
        rows.append(
            f"{analytics_display_names[system]} & {row['count_bias']:.2f} & "
            f"{row['count_mae']:.2f} & {100 * row['pooled_ratio_error']:+.2f} & "
            f"{100 * row['ratio_bias']:+.2f} & {100 * row['ratio_mae']:.2f} & "
            f"{100 * row['ratio_error_cancellation_fraction']:.1f} \\\\"
        )
    text = "\n".join(
        [
            r"\begingroup",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l S[table-format=-1.2] S[table-format=1.2] S[table-format=+2.2] S[table-format=+2.2] S[table-format=2.2] S[table-format=2.1]}",
            r"\toprule",
            r"System & \multicolumn{1}{c}{\shortstack{Count\\bias}} & \multicolumn{1}{c}{\shortstack{Count\\MAE}} & \multicolumn{1}{c}{\shortstack{Pooled error\\(pp)}} & \multicolumn{1}{c}{\shortstack{Frame bias\\(pp)}} & \multicolumn{1}{c}{\shortstack{Frame MAE\\(pp)}} & \multicolumn{1}{c}{\shortstack{Cancellation\\(\%)}} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular*}",
            r"\endgroup",
        ]
    )
    path = tables_dir / "mi05_analytics_table.tex"
    write_text(path, text)
    outputs.append(path)

    rows = []
    for system, row in agreement_by_system.iterrows():
        rows.append(
            f"{DISPLAY_NAMES[system]} & {int(row['ratio_close_frames'])} & "
            f"{int(row['count_wrong_ratio_close_frames'])} & "
            f"{int(row['final_class_wrong_ratio_close_frames'])} & "
            f"{int(row['gt_shape_class_wrong_ratio_close_frames'])} \\\\"
        )
    text = "\n".join(
        [
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l *{4}{S[table-format=2.0]}}",
            r"\toprule",
            r"System & \multicolumn{1}{c}{\shortstack{Ratio-close\\frames}} & \multicolumn{1}{c}{\shortstack{Wrong\\count}} & \multicolumn{1}{c}{\shortstack{Final class\\error}} & \multicolumn{1}{c}{\shortstack{GT-shape\\class error}} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular*}",
        ]
    )
    path = tables_dir / "mi05_cancellation_cases_table.tex"
    write_text(path, text)
    outputs.append(path)

    rows = []
    for system in SYSTEM_ORDER:
        boot = bootstrap["bootstrap"]["chosen_f1_intervals"][system]
        rows.append(
            f"{DISPLAY_NAMES[system]} & {format_score(boot['point_estimate'])} & "
            f"[{format_score(boot['percentile_2_5'])}, "
            f"{format_score(boot['percentile_97_5'])}] \\\\"
        )
    text = "\n".join(
        [
            r"\begin{tabular}{l S[table-format=1.3] c}",
            r"\toprule",
            r"System & \multicolumn{1}{c}{Pooled F1} & \shortstack{Moving-block 95\%\\percentile interval} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    path = tables_dir / "bootstrap_f1_table.tex"
    write_text(path, text)
    outputs.append(path)
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--thesis-root", required=True)
    args = parser.parse_args(argv)
    results_root = Path(args.results_root).resolve()
    thesis_root = Path(args.thesis_root).resolve()
    figures_dir = thesis_root / "figures"
    tables_dir = thesis_root / "tables"
    generated_dir = thesis_root / "generated"
    for directory in (figures_dir, tables_dir, generated_dir):
        directory.mkdir(parents=True, exist_ok=True)

    comparison_path = results_root / "comparison_summary.csv"
    classifier_path = results_root / "classifier_validation_summary.csv"
    agreement_path = results_root / "counts_ratio_validation" / "agreement_summary.csv"
    bootstrap_path = results_root / "temporal_bootstrap" / "analysis_summary.json"
    verify_indexed_sources(
        results_root,
        (comparison_path, classifier_path, agreement_path, bootstrap_path),
    )

    comparison = pd.read_csv(comparison_path)
    classifier = pd.read_csv(classifier_path)
    if set(comparison["system"]) != set(SYSTEM_ORDER):
        raise ValueError("Comparison summary does not contain exactly the three approved systems")
    if len(classifier) != 1:
        raise ValueError("Classifier validation summary must contain exactly one row")
    dataset_path = results_root / "dataset_audit" / "dataset_manifest.json"
    dataset = read_json(dataset_path)
    agreement = pd.read_csv(agreement_path)
    if set(agreement["system"]) != set(SYSTEM_ORDER):
        raise ValueError("Agreement summary does not contain exactly the three approved systems")
    bootstrap = read_json(bootstrap_path)

    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    outputs: list[Path] = []
    outputs.extend(
        build_tables(comparison, classifier, dataset, bootstrap, agreement, tables_dir)
    )

    staged = [
        (
            results_root / "figures" / "methods" / "figure_manifest.json",
            results_root / "figures" / "methods" / "evaluation_pipeline.pdf",
        ),
        (
            results_root / "mask_conversion_audit" / "figures" / "figure_manifest.json",
            results_root / "mask_conversion_audit" / "figures" / "mask_conversion_audit.pdf",
        ),
        *[
            (
                results_root / "figures" / "step19" / "figure_manifest.json",
                results_root / "figures" / "step19" / filename,
            )
            for filename in (
                "classifier_confusion_matrices.pdf",
                "permutation_importance_stability_and_groups.pdf",
                "mi05_detection_comparison.pdf",
                "mi05_classification_comparison.pdf",
                "mi05_temporal_segmentation_metrics.pdf",
                "mi05_cell_count_validation.pdf",
                "mi05_mitotic_ratio_validation.pdf",
            )
        ],
        (
            results_root / "figures" / "step20" / "figure_manifest.json",
            results_root / "figures" / "step20" / "mi05_failure_case_comparison.pdf",
        ),
    ]
    staged_sources: list[dict[str, str]] = []
    for manifest_path, source in staged:
        verify_manifest_output(manifest_path, source)
        destination = figures_dir / source.name
        shutil.copyfile(source, destination)
        outputs.append(destination)
        staged_sources.append(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "source": str(source),
                "source_sha256": sha256_file(source),
                "destination": str(destination),
            }
        )

    manifest_path = generated_dir / "act8_assets_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "verified_inputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (
                comparison_path,
                classifier_path,
                agreement_path,
                bootstrap_path,
                dataset_path,
            )
        ],
        "staged_sources": staged_sources,
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(outputs, key=lambda item: str(item))
        ],
        "notes": (
            "All Act 8 figures and tables derive from verified outputs; copied figures "
            "were checked against their source manifests before staging."
        ),
    }
    write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"outputs": len(outputs), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
