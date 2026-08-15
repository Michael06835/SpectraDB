from pathlib import Path
import json
import csv
from statistics import mean, stdev

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise RuntimeError(
        "matplotlib is not installed.\n"
        "Install it with:\n"
        "conda install -c conda-forge matplotlib"
    ) from exc


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

SEED_RUNS = {
    "seed42": RUNS / "EXP-RAMAN-001-FULL",
    "seed123": RUNS / "EXP-RAMAN-001-SEED123",
    "seed2026": RUNS / "EXP-RAMAN-001-SEED2026",
}

WEIGHTED_RUN = RUNS / "EXP-RAMAN-002-FULL"

# Canonical checkpoint:
# This remains the pre-specified seed-42 baseline.
# We do NOT select a checkpoint according to test performance.
CANONICAL_RUN = SEED_RUNS["seed42"]

OUTPUT_DIR = RUNS / "Raman-v1-FINAL"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Files
# ============================================================

SEED_SUMMARY_CSV = OUTPUT_DIR / "raman_seed_summary.csv"
SEED_SUMMARY_JSON = OUTPUT_DIR / "raman_seed_summary.json"

PER_LABEL_CSV = OUTPUT_DIR / "raman_per_label_3seed_summary.csv"

ABLATION_CSV = OUTPUT_DIR / "raman_ablation_summary.csv"

FINAL_JSON = OUTPUT_DIR / "raman_final_summary.json"
FINAL_MD = OUTPUT_DIR / "RAMAN_FINAL_SUMMARY.md"

LOSS_FIG = OUTPUT_DIR / "training_loss_seed42.png"
VAL_METRICS_FIG = OUTPUT_DIR / "validation_metrics_seed42.png"
PER_LABEL_F1_FIG = OUTPUT_DIR / "per_label_f1_3seed.png"
ABLATION_FIG = OUTPUT_DIR / "bce_weighting_comparison.png"


# ============================================================
# Helpers
# ============================================================

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def safe_float(value):
    if value is None:
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()

    if value == "":
        return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan


def mean_sd(values):
    values = [
        float(x)
        for x in values
        if not np.isnan(float(x))
    ]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return values[0], np.nan

    return mean(values), stdev(values)


# ============================================================
# Check required files
# ============================================================

def validate_inputs():

    required = []

    for run_dir in SEED_RUNS.values():

        required.extend([
            run_dir / "test_metrics.json",
            run_dir / "training_log.csv",
            run_dir / "test_per_label_metrics.csv",
            run_dir / "best_model.pt",
            run_dir / "config.json",
        ])

    required.extend([
        WEIGHTED_RUN / "test_metrics.json",
        WEIGHTED_RUN / "test_per_label_metrics.csv",
    ])

    missing = [
        p for p in required
        if not p.exists()
    ]

    if missing:

        msg = "\n".join(
            str(p)
            for p in missing
        )

        raise FileNotFoundError(
            "The following required files are missing:\n"
            + msg
        )


# ============================================================
# 1. Three-seed overall summary
# ============================================================

def build_seed_summary():

    metric_keys = [
        "test_loss",
        "test_micro_f1",
        "test_macro_f1",
        "test_micro_precision",
        "test_micro_recall",
        "test_mAP",
        "test_macro_AUROC",
    ]

    seed_results = {}

    for seed_name, run_dir in SEED_RUNS.items():

        metrics = read_json(
            run_dir / "test_metrics.json"
        )

        seed_results[seed_name] = metrics

    summary = {}

    for metric in metric_keys:

        values = [
            seed_results[seed][metric]
            for seed in SEED_RUNS
        ]

        avg, sd = mean_sd(values)

        summary[metric] = {
            "mean": avg,
            "sample_sd": sd,
            "values": {
                seed: float(
                    seed_results[seed][metric]
                )
                for seed in SEED_RUNS
            },
        }

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    with open(
        SEED_SUMMARY_CSV,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "metric",
            "seed42",
            "seed123",
            "seed2026",
            "mean",
            "sample_sd",
        ])

        for metric in metric_keys:

            row = summary[metric]

            writer.writerow([
                metric,
                row["values"]["seed42"],
                row["values"]["seed123"],
                row["values"]["seed2026"],
                row["mean"],
                row["sample_sd"],
            ])

    write_json(
        SEED_SUMMARY_JSON,
        summary,
    )

    return summary, seed_results


# ============================================================
# 2. Three-seed per-label summary
# ============================================================

def build_per_label_summary():

    seed_tables = {}

    for seed_name, run_dir in SEED_RUNS.items():

        rows = read_csv(
            run_dir / "test_per_label_metrics.csv"
        )

        seed_tables[seed_name] = {
            row["label"]: row
            for row in rows
        }

    label_names = list(
        seed_tables["seed42"].keys()
    )

    metric_columns = [
        "precision",
        "recall",
        "f1",
        "average_precision",
        "auroc",
    ]

    output_rows = []

    for label in label_names:

        row_out = {
            "label": label,
            "positive_test_samples_seed42":
                int(
                    float(
                        seed_tables["seed42"][label][
                            "positive_test_samples"
                        ]
                    )
                ),
        }

        for metric in metric_columns:

            values = []

            for seed_name in SEED_RUNS:

                value = safe_float(
                    seed_tables[
                        seed_name
                    ][label][metric]
                )

                values.append(value)

                row_out[
                    f"{metric}_{seed_name}"
                ] = value

            avg, sd = mean_sd(values)

            row_out[
                f"{metric}_mean"
            ] = avg

            row_out[
                f"{metric}_sd"
            ] = sd

        output_rows.append(
            row_out
        )

    # Sort by mean F1 descending
    output_rows.sort(
        key=lambda x: x["f1_mean"],
        reverse=True,
    )

    fieldnames = [
        "label",
        "positive_test_samples_seed42",

        "precision_seed42",
        "precision_seed123",
        "precision_seed2026",
        "precision_mean",
        "precision_sd",

        "recall_seed42",
        "recall_seed123",
        "recall_seed2026",
        "recall_mean",
        "recall_sd",

        "f1_seed42",
        "f1_seed123",
        "f1_seed2026",
        "f1_mean",
        "f1_sd",

        "average_precision_seed42",
        "average_precision_seed123",
        "average_precision_seed2026",
        "average_precision_mean",
        "average_precision_sd",

        "auroc_seed42",
        "auroc_seed123",
        "auroc_seed2026",
        "auroc_mean",
        "auroc_sd",
    ]

    with open(
        PER_LABEL_CSV,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(output_rows)

    return output_rows


# ============================================================
# 3. Seed-42 training curves
# ============================================================

def plot_seed42_training():

    log_rows = read_csv(
        CANONICAL_RUN / "training_log.csv"
    )

    epochs = np.array([
        int(row["epoch"])
        for row in log_rows
    ])

    train_loss = np.array([
        float(row["train_loss"])
        for row in log_rows
    ])

    val_loss = np.array([
        float(row["val_loss"])
        for row in log_rows
    ])

    val_micro_f1 = np.array([
        float(row["val_micro_f1"])
        for row in log_rows
    ])

    val_macro_f1 = np.array([
        float(row["val_macro_f1"])
        for row in log_rows
    ])

    val_map = np.array([
        float(row["val_mAP"])
        for row in log_rows
    ])

    # --------------------------------------------------------
    # Loss curve
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        train_loss,
        label="Train loss",
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title(
        "Raman 1D-CNN Training and Validation Loss (Seed 42)"
    )

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        LOSS_FIG,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Validation metrics
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        val_micro_f1,
        label="Micro-F1",
    )

    plt.plot(
        epochs,
        val_macro_f1,
        label="Macro-F1",
    )

    plt.plot(
        epochs,
        val_map,
        label="mAP",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.ylim(0.0, 1.02)

    plt.title(
        "Raman 1D-CNN Validation Metrics (Seed 42)"
    )

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        VAL_METRICS_FIG,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 4. Per-label 3-seed F1 figure
# ============================================================

def plot_per_label_f1(per_label_rows):

    # Plot lowest-to-highest for readability
    rows = sorted(
        per_label_rows,
        key=lambda x: x["f1_mean"],
    )

    labels = [
        row["label"]
        for row in rows
    ]

    means = np.array([
        row["f1_mean"]
        for row in rows
    ])

    sds = np.array([
        row["f1_sd"]
        for row in rows
    ])

    y = np.arange(
        len(labels)
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        y,
        means,
        xerr=sds,
        capsize=3,
    )

    plt.yticks(
        y,
        labels,
    )

    plt.xlabel(
        "F1 score (mean ± sample SD, 3 seeds)"
    )

    plt.xlim(
        0.0,
        1.02,
    )

    plt.title(
        "Raman Functional-Group Performance Across 3 Seeds"
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        PER_LABEL_F1_FIG,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 5. BCE vs weighted BCE comparison
# ============================================================

def build_ablation_summary():

    baseline = read_json(
        CANONICAL_RUN / "test_metrics.json"
    )

    weighted = read_json(
        WEIGHTED_RUN / "test_metrics.json"
    )

    threshold_summary_path = (
        CANONICAL_RUN
        / "threshold_calibration_summary.json"
    )

    threshold_summary = None

    if threshold_summary_path.exists():

        threshold_summary = read_json(
            threshold_summary_path
        )

    rows = [
        {
            "experiment":
                "EXP-RAMAN-001-FULL",
            "description":
                "Ordinary BCE, threshold=0.5",
            "micro_f1":
                baseline["test_micro_f1"],
            "macro_f1":
                baseline["test_macro_f1"],
            "precision":
                baseline["test_micro_precision"],
            "recall":
                baseline["test_micro_recall"],
            "mAP":
                baseline["test_mAP"],
            "macro_AUROC":
                baseline["test_macro_AUROC"],
        },
        {
            "experiment":
                "EXP-RAMAN-002-FULL",
            "description":
                "sqrt(pos_weight), cap=20",
            "micro_f1":
                weighted["test_micro_f1"],
            "macro_f1":
                weighted["test_macro_f1"],
            "precision":
                weighted["test_micro_precision"],
            "recall":
                weighted["test_micro_recall"],
            "mAP":
                weighted["test_mAP"],
            "macro_AUROC":
                weighted["test_macro_AUROC"],
        },
    ]

    if threshold_summary is not None:

        # Raman threshold calibration summary format:
        # threshold_summary["test_default"]
        # threshold_summary["test_calibrated"]
        test_cal = (
            threshold_summary["test_calibrated"]
        )

        rows.append({
            "experiment":
                "EXP-RAMAN-001-CALIBRATED",
            "description":
                "Seed42 BCE + validation-calibrated thresholds",
            "micro_f1":
                test_cal["micro_f1"],
            "macro_f1":
                test_cal["macro_f1"],
            "precision":
                test_cal["micro_precision"],
            "recall":
                test_cal["micro_recall"],
            # Threshold calibration does not change
            # threshold-free model rankings.
            "mAP":
                baseline["test_mAP"],
            "macro_AUROC":
                baseline[
                    "test_macro_AUROC"
                ],
        })

    fieldnames = [
        "experiment",
        "description",
        "micro_f1",
        "macro_f1",
        "precision",
        "recall",
        "mAP",
        "macro_AUROC",
    ]

    with open(
        ABLATION_CSV,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return rows


def plot_weighting_comparison(ablation_rows):

    selected = [
        row
        for row in ablation_rows
        if row["experiment"]
        in {
            "EXP-RAMAN-001-FULL",
            "EXP-RAMAN-002-FULL",
        }
    ]

    metrics = [
        "micro_f1",
        "macro_f1",
        "precision",
        "recall",
        "mAP",
        "macro_AUROC",
    ]

    labels = [
        "Micro-F1",
        "Macro-F1",
        "Precision",
        "Recall",
        "mAP",
        "AUROC",
    ]

    x = np.arange(
        len(metrics)
    )

    width = 0.36

    baseline_values = [
        selected[0][m]
        for m in metrics
    ]

    weighted_values = [
        selected[1][m]
        for m in metrics
    ]

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        x - width / 2,
        baseline_values,
        width,
        label="Ordinary BCE",
    )

    plt.bar(
        x + width / 2,
        weighted_values,
        width,
        label="Weighted BCE",
    )

    plt.xticks(
        x,
        labels,
        rotation=20,
    )

    plt.ylabel("Score")
    plt.ylim(0.80, 1.01)

    plt.title(
        "Raman Baseline vs Weighted BCE"
    )

    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        ABLATION_FIG,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 6. Final JSON + Markdown report
# ============================================================

def build_final_report(
    seed_summary,
    seed_results,
    per_label_rows,
    ablation_rows,
):

    canonical_metrics = read_json(
        CANONICAL_RUN / "test_metrics.json"
    )

    canonical_config = read_json(
        CANONICAL_RUN / "config.json"
    )

    threshold_path = (
        CANONICAL_RUN
        / "threshold_calibration_summary.json"
    )

    threshold_summary = (
        read_json(threshold_path)
        if threshold_path.exists()
        else None
    )

    # Lowest-F1 structural labels across 3 seeds
    lowest_labels = sorted(
        per_label_rows,
        key=lambda x: x["f1_mean"],
    )[:5]

    final_json = {

        "model_version":
            "Raman-v1",

        "status":
            "FROZEN",

        "canonical_checkpoint":
            str(
                CANONICAL_RUN
                / "best_model.pt"
            ),

        "canonical_checkpoint_policy":
            (
                "Seed 42 remains the pre-specified "
                "canonical checkpoint. Test-set "
                "performance was not used to select "
                "between random seeds."
            ),

        "architecture":
            {
                "backbone":
                    "1D-CNN / residual 1D CNN",
                "embedding_dim":
                    canonical_config.get(
                        "embedding_dim",
                        256,
                    ),
                "normalization":
                    canonical_config.get(
                        "normalization",
                        "max",
                    ),
                "loss":
                    "BCEWithLogitsLoss",
                "threshold":
                    0.5,
            },

        "three_seed_summary":
            seed_summary,

        "seed_runs":
            seed_results,

        "canonical_seed42_test":
            canonical_metrics,

        "threshold_calibration_conclusion":
            (
                "Validation-optimized per-label thresholds "
                "increased test Recall and Macro-F1 but "
                "reduced Precision and Micro-F1. "
                "The default threshold of 0.5 is retained."
            ),

        "class_weighting_conclusion":
            (
                "sqrt(pos_weight) with cap=20 increased "
                "Recall and Macro-F1 and slightly improved mAP, "
                "but reduced Precision and Micro-F1. "
                "Ordinary BCE is retained."
            ),

        "lowest_mean_f1_labels":
            [
                {
                    "label":
                        row["label"],
                    "f1_mean":
                        row["f1_mean"],
                    "f1_sd":
                        row["f1_sd"],
                }
                for row in lowest_labels
            ],
    }

    write_json(
        FINAL_JSON,
        final_json,
    )

    # --------------------------------------------------------
    # Markdown
    # --------------------------------------------------------

    micro = seed_summary[
        "test_micro_f1"
    ]

    macro = seed_summary[
        "test_macro_f1"
    ]

    map_result = seed_summary[
        "test_mAP"
    ]

    auroc = seed_summary[
        "test_macro_AUROC"
    ]

    precision = seed_summary[
        "test_micro_precision"
    ]

    recall = seed_summary[
        "test_micro_recall"
    ]

    lines = []

    lines.append(
        "# Raman 单模态模型 v1 最终实验总结"
    )

    lines.append("")

    lines.append(
        "状态：**FROZEN**"
    )

    lines.append("")

    lines.append(
        "## 1. 最终模型设置"
    )

    lines.append("")

    lines.append(
        "- Backbone：1D-CNN / residual 1D CNN"
    )

    lines.append(
        "- 输入：QM9S Raman，3501 points"
    )

    lines.append(
        "- 数据划分：scaffold split"
    )

    lines.append(
        "- Loss：BCEWithLogitsLoss"
    )

    lines.append(
        "- 统一判定阈值：0.5"
    )

    lines.append(
        "- Normalization：per-spectrum max normalization"
    )

    lines.append(
        "- Canonical checkpoint：Seed 42"
    )

    lines.append("")

    lines.append(
        "注意：Seed 42 为预先固定的 canonical checkpoint；"
        "不依据 test set 在多个随机种子中挑选最佳 checkpoint。"
    )

    lines.append("")

    lines.append(
        "## 2. 三随机种子最终结果"
    )

    lines.append("")

    lines.append(
        "| Metric | Mean ± sample SD |"
    )

    lines.append(
        "|---|---:|"
    )

    lines.append(
        f"| Micro-F1 | "
        f"{micro['mean']:.4f} ± "
        f"{micro['sample_sd']:.4f} |"
    )

    lines.append(
        f"| Macro-F1 | "
        f"{macro['mean']:.4f} ± "
        f"{macro['sample_sd']:.4f} |"
    )

    lines.append(
        f"| Precision | "
        f"{precision['mean']:.4f} ± "
        f"{precision['sample_sd']:.4f} |"
    )

    lines.append(
        f"| Recall | "
        f"{recall['mean']:.4f} ± "
        f"{recall['sample_sd']:.4f} |"
    )

    lines.append(
        f"| mAP | "
        f"{map_result['mean']:.4f} ± "
        f"{map_result['sample_sd']:.4f} |"
    )

    lines.append(
        f"| Macro-AUROC | "
        f"{auroc['mean']:.4f} ± "
        f"{auroc['sample_sd']:.4f} |"
    )

    lines.append("")

    lines.append(
        "## 3. 阈值校准"
    )

    lines.append("")

    lines.append(
        "逐标签 validation-based threshold calibration "
        "可提升独立 scaffold test 上的 Recall 和 Macro-F1，"
        "但 Precision 与 Micro-F1 下降，因此 Raman-v1 "
        "保留统一阈值 0.5。"
    )

    lines.append("")

    lines.append(
        "## 4. 类别不平衡实验"
    )

    lines.append("")

    lines.append(
        "采用 sqrt(pos_weight) 并将权重截断至 20 后，"
        "模型 Recall、Macro-F1 和 mAP 略有提高，"
        "但 Precision 和 Micro-F1 下降。因此 Raman-v1 "
        "不采用 class weighting。"
    )

    lines.append("")

    lines.append(
        "## 5. 三随机种子表现相对较弱的类别"
    )

    lines.append("")

    lines.append(
        "| Label | F1 mean | F1 SD |"
    )

    lines.append(
        "|---|---:|---:|"
    )

    for row in lowest_labels:

        lines.append(
            f"| {row['label']} | "
            f"{row['f1_mean']:.4f} | "
            f"{row['f1_sd']:.4f} |"
        )

    lines.append("")

    lines.append(
        "## 6. 输出文件"
    )

    lines.append("")

    lines.append(
        "- `raman_seed_summary.csv`"
    )

    lines.append(
        "- `raman_seed_summary.json`"
    )

    lines.append(
        "- `raman_per_label_3seed_summary.csv`"
    )

    lines.append(
        "- `raman_ablation_summary.csv`"
    )

    lines.append(
        "- `training_loss_seed42.png`"
    )

    lines.append(
        "- `validation_metrics_seed42.png`"
    )

    lines.append(
        "- `per_label_f1_3seed.png`"
    )

    lines.append(
        "- `bce_weighting_comparison.png`"
    )

    lines.append(
        "- `raman_final_summary.json`"
    )

    with open(
        FINAL_MD,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(lines)
        )

    return final_json


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 80)
    print("Raman-v1 FINALIZATION")
    print("=" * 80)

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    print()
    print(
        "[1/6] Validating input files..."
    )

    validate_inputs()

    print(
        "      OK"
    )

    # --------------------------------------------------------
    # Seed summary
    # --------------------------------------------------------

    print()
    print(
        "[2/6] Building 3-seed summary..."
    )

    (
        seed_summary,
        seed_results,
    ) = build_seed_summary()

    print(
        "      OK"
    )

    # --------------------------------------------------------
    # Per-label
    # --------------------------------------------------------

    print()
    print(
        "[3/6] Building per-label 3-seed summary..."
    )

    per_label_rows = (
        build_per_label_summary()
    )

    print(
        "      OK"
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    print()
    print(
        "[4/6] Generating figures..."
    )

    plot_seed42_training()

    plot_per_label_f1(
        per_label_rows
    )

    print(
        "      OK"
    )

    # --------------------------------------------------------
    # Ablation
    # --------------------------------------------------------

    print()
    print(
        "[5/6] Building ablation summary..."
    )

    ablation_rows = (
        build_ablation_summary()
    )

    plot_weighting_comparison(
        ablation_rows
    )

    print(
        "      OK"
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print(
        "[6/6] Writing final report..."
    )

    build_final_report(
        seed_summary,
        seed_results,
        per_label_rows,
        ablation_rows,
    )

    print(
        "      OK"
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Raman-v1 FINAL RESULTS")
    print("=" * 80)

    display_metrics = [
        (
            "Micro-F1",
            "test_micro_f1"
        ),
        (
            "Macro-F1",
            "test_macro_f1"
        ),
        (
            "Precision",
            "test_micro_precision"
        ),
        (
            "Recall",
            "test_micro_recall"
        ),
        (
            "mAP",
            "test_mAP"
        ),
        (
            "Macro-AUROC",
            "test_macro_AUROC"
        ),
    ]

    print()

    for display_name, key in display_metrics:

        result = seed_summary[key]

        print(
            f"{display_name:12s}: "
            f"{result['mean']:.4f} "
            f"± "
            f"{result['sample_sd']:.4f}"
        )

    print()
    print(
        "Canonical checkpoint:"
    )

    print(
        CANONICAL_RUN
        / "best_model.pt"
    )

    print()
    print(
        "Raman-v1 status: FROZEN"
    )

    print()
    print(
        "Saved outputs:"
    )

    for path in [
        SEED_SUMMARY_CSV,
        SEED_SUMMARY_JSON,
        PER_LABEL_CSV,
        ABLATION_CSV,
        LOSS_FIG,
        VAL_METRICS_FIG,
        PER_LABEL_F1_FIG,
        ABLATION_FIG,
        FINAL_JSON,
        FINAL_MD,
    ]:

        print(
            " ",
            path
        )


if __name__ == "__main__":
    main()
