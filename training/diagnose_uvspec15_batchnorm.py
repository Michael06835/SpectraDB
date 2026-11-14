"""Diagnose UVSPEC15 BatchNorm behavior without changing frozen checkpoints.

This combines and corrects the two earlier UV-Vis BatchNorm diagnostics:
  - Quick: compare train/eval-mode outputs on the *same fixed validation samples*.
  - Full: recompute BatchNorm running stats on TRAIN ONLY, on a copy of the
          model, and compare validation/test inference before and after.

Neither operation trains or saves model weights. No model is selected by test
performance. Existing 003/004 runs are read-only. All diagnostic JSON files
are saved under runs/UVSPEC15-BN-DIAGNOSTICS/.

Usage (from the SpectraDB root in the spectra-db environment):
  python training/diagnose_uvspec15_batchnorm.py --mode quick --run both
  python training/diagnose_uvspec15_batchnorm.py --mode full --run both
  python training/diagnose_uvspec15_batchnorm.py --mode both --run 004
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.datasets.qm9s_spectrum_dataset import QM9SSpectrumDataset
from training.models.spectrum_cnn1d import SpectrumStructureModel

PREPARED = ROOT / "processed" / "qm9s" / "prepared"
UV_PATH = PREPARED / "uvvis_float32.npy"
LABEL_PATH = PREPARED / "uvspec15_labels.npy"
LABEL_NAMES_PATH = PREPARED / "uvspec15_label_names.json"
SPLIT_PATH = PREPARED / "uvvis_scaffold_split_valid.npz"
OUTPUT_ROOT = ROOT / "runs" / "UVSPEC15-BN-DIAGNOSTICS"

RUN_IDS = {
    "003": "EXP-UVVIS-003-UVSPEC15-FULL",
    "004": "EXP-UVVIS-004-UVSPEC15-WBCE",
}

# Preserve the published UVSPEC15 experiments' loader configuration.
BATCH_SIZE = 512
NUM_WORKERS = 4
THRESHOLD = 0.5
SEED = 42


def require_inputs(run_dir: Path) -> None:
    paths = [
        UV_PATH,
        LABEL_PATH,
        LABEL_NAMES_PATH,
        SPLIT_PATH,
        run_dir / "best_model.pt",
        run_dir / "test_metrics.json",
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")


def make_loader(indices: np.ndarray, device: torch.device) -> DataLoader:
    dataset = QM9SSpectrumDataset(
        spectra_path=UV_PATH,
        labels_path=LABEL_PATH,
        indices=indices,
        normalization="max",
    )
    # Same batch size, num_workers, persistent_workers and prefetch factor
    # as the official baseline training/evaluation code.
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        prefetch_factor=2,
    )


def make_model(checkpoint: dict, n_labels: int, device: torch.device) -> torch.nn.Module:
    config = checkpoint.get("config", {})
    model = SpectrumStructureModel(
        num_labels=n_labels,
        embedding_dim=int(config.get("embedding_dim", 256)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def collect(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Run the model in its current mode (eval or train), without gradients."""
    logits_list, labels_list, probs_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["spectrum"].to(device, non_blocking=True)
            y = batch["labels"]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(x)["logits"]
            # Match the official trainer: sigmoid runs on the original
            # logits dtype (FP16 under CUDA AMP) before conversion to NumPy.
            probabilities = torch.sigmoid(logits)
            logits_list.append(logits.detach().float().cpu().numpy())
            probs_list.append(probabilities.detach().float().cpu().numpy())
            labels_list.append(y.detach().cpu().numpy())
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0).astype(np.int32)
    probs = np.concatenate(probs_list, axis=0)
    if not np.isfinite(logits).all() or not np.isfinite(probs).all():
        raise RuntimeError("Non-finite logits/probabilities; investigate model numerics.")
    return labels, logits, probs


def get_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= THRESHOLD).astype(np.int32)
    ap_list, auroc_list = [], []
    for j in range(labels.shape[1]):
        true, score = labels[:, j], probabilities[:, j]
        pos = int(true.sum())
        neg = int(len(true) - pos)
        if pos and neg:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ap_list.append(float(average_precision_score(true, score)))
                auroc_list.append(float(roc_auc_score(true, score)))
        else:
            ap_list.append(None)
            auroc_list.append(None)
    def mean_valid(vals):
        valid = [v for v in vals if v is not None]
        return float(np.mean(valid)) if valid else None
    return {
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(labels, predictions, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(labels, predictions, average="micro", zero_division=0)),
        "mAP": mean_valid(ap_list),
        "macro_AUROC": mean_valid(auroc_list),
    }


def summarize_bn(model: torch.nn.Module) -> list[dict]:
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            mean = module.running_mean.detach().float().cpu().numpy()
            var = module.running_var.detach().float().cpu().numpy()
            rows.append({
                "name": name,
                "channels": int(len(mean)),
                "num_batches_tracked": int(module.num_batches_tracked.item()),
                "mean_abs_mean": float(np.mean(np.abs(mean))),
                "min_running_var": float(np.min(var)),
                "max_running_var": float(np.max(var)),
                "nonfinite_mean": int((~np.isfinite(mean)).sum()),
                "nonfinite_var": int((~np.isfinite(var)).sum()),
                "nonpositive_var": int((var <= 0).sum()),
            })
    return rows


def metrics_line(prefix: str, m: dict) -> None:
    print(
        f"{prefix:27s} | Micro-F1={m['micro_f1']:.4f}"
        f" Macro-F1={m['macro_f1']:.4f}"
        f" Precision={m['micro_precision']:.4f}"
        f" Recall={m['micro_recall']:.4f}"
        f" mAP={m['mAP']:.4f}"
        f" AUROC={m['macro_AUROC']:.4f}"
    )


def quick_diagnostic(base_model: torch.nn.Module, val_idx: np.ndarray,
                     device: torch.device, n_samples: int) -> dict:
    """Compare running stats vs per-batch stats; TRAIN MODE IS NOT VALID EVALUATION."""
    if not 1 <= n_samples <= len(val_idx):
        raise ValueError(f"--samples must be 1..{len(val_idx)}")
    rng = np.random.default_rng(SEED)
    subset = np.sort(rng.choice(val_idx, size=n_samples, replace=False))
    loader = make_loader(subset, device)

    # Original model is evaluated and never enters train mode.
    base_model.eval()
    labels_eval, logits_eval, probs_eval = collect(base_model, loader, device)
    metrics_eval = get_metrics(labels_eval, probs_eval)

    # Separate, disposable copy: train() updates its BN running buffers.
    mode_copy = copy.deepcopy(base_model)
    mode_copy.train()
    labels_train, logits_train, probs_train = collect(mode_copy, loader, device)
    if not np.array_equal(labels_train, labels_eval):
        raise AssertionError("Sample/label mismatch between train and eval passes")
    metrics_train = get_metrics(labels_train, probs_train)

    delta = np.abs(logits_train - logits_eval)
    decision_disagreement = float(np.mean(
        (probs_train >= THRESHOLD) != (probs_eval >= THRESHOLD)
    ))
    result = {
        "scope": "Fixed validation subset; diagnostic only, not valid train-mode performance",
        "samples": int(len(subset)),
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "threshold": THRESHOLD,
        "eval_mode": metrics_eval,
        "train_mode_batch_stats": metrics_train,
        "logit_difference_mean_absolute": float(delta.mean()),
        "logit_difference_p95_absolute": float(np.quantile(delta, 0.95)),
        "threshold_0_5_decision_disagreement_fraction": decision_disagreement,
    }
    metrics_line("Eval (checkpoint BN)", metrics_eval)
    metrics_line("Train (batch BN; diagnostic)", metrics_train)
    print(f"  Mean |logit difference| = {result['logit_difference_mean_absolute']:.4f}; "
          f"decision disagreement = {decision_disagreement:.2%}")
    del mode_copy
    return result


def reset_recompute_bn(model: torch.nn.Module, loader: DataLoader,
                       device: torch.device) -> dict:
    """Reestimate *only* BN buffers on train data; no gradient or weight update."""
    bn_modules = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    if not bn_modules:
        raise RuntimeError("No BatchNorm1d layers found")
    model.eval()
    moments = []
    for bn in bn_modules:
        moments.append(bn.momentum)
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average of batch statistics
        bn.train()
    num_batches, num_samples = 0, 0
    try:
        with torch.no_grad():
            for batch in loader:
                x = batch["spectrum"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    _ = model(x)
                num_batches += 1
                num_samples += int(x.shape[0])
    finally:
        for bn, momentum in zip(bn_modules, moments):
            bn.momentum = momentum
        model.eval()
    return {
        "batchnorm_layers": len(bn_modules),
        "train_batches_used": num_batches,
        "train_samples_used": num_samples,
        "recalibration": "train split only; reset running stats; cumulative batch momentum; no optimizer",
    }


def full_diagnostic(base_model: torch.nn.Module, train_idx: np.ndarray,
                    val_idx: np.ndarray, test_idx: np.ndarray,
                    device: torch.device, run_dir: Path) -> dict:
    val_loader = make_loader(val_idx, device)
    test_loader = make_loader(test_idx, device)
    train_loader = make_loader(train_idx, device)

    base_model.eval()
    vy, _, vp = collect(base_model, val_loader, device)
    ty, _, tp = collect(base_model, test_loader, device)
    before_val = get_metrics(vy, vp)
    before_test = get_metrics(ty, tp)
    metrics_line("BEFORE - Validation", before_val)
    metrics_line("BEFORE - Test", before_test)

    # Verification of published 0.5 baseline BEFORE any BN modifications.
    with open(run_dir / "test_metrics.json", "r", encoding="utf-8") as handle:
        recorded = json.load(handle)
    recorded_keys = {
        "micro_f1": "test_micro_f1",
        "macro_f1": "test_macro_f1",
        "mAP": "test_mAP",
        "macro_AUROC": "test_macro_AUROC",
    }
    discrepancies = {}
    for our_key, saved_key in recorded_keys.items():
        if saved_key in recorded and abs(before_test[our_key] - float(recorded[saved_key])) > 0.002:
            discrepancies[our_key] = {
                "computed": before_test[our_key],
                "recorded": float(recorded[saved_key]),
            }
    if discrepancies:
        raise RuntimeError(f"Original Test metrics failed reproduction: {discrepancies}")
    print("PASS: frozen checkpoint baseline reproduced (threshold 0.5)")

    # Recalibration is on a copy, strictly avoiding frozen checkpoint modification.
    bn_copy = copy.deepcopy(base_model)
    params_before = {k: v.detach().clone() for k, v in bn_copy.named_parameters()}
    bn_before = summarize_bn(bn_copy)
    information = reset_recompute_bn(bn_copy, train_loader, device)
    bn_after = summarize_bn(bn_copy)
    if not all(torch.equal(params_before[k], v) for k, v in bn_copy.named_parameters()):
        raise AssertionError("Model parameter changed during BN-only recalibration")
    print(f"PASS: model parameters unchanged; recomputed "
          f"{information['batchnorm_layers']} BatchNorm layers using "
          f"{information['train_samples_used']} TRAIN samples")

    _, _, vp_after = collect(bn_copy, val_loader, device)
    _, _, tp_after = collect(bn_copy, test_loader, device)
    after_val = get_metrics(vy, vp_after)
    after_test = get_metrics(ty, tp_after)
    metrics_line("AFTER - Validation", after_val)
    metrics_line("AFTER - Test", after_test)

    metric_keys = list(before_val)
    result = {
        "note": "DIAGNOSTIC ONLY. Recomputed BN on a COPY from TRAIN. "
                "Do not choose/publish a new checkpoint based on Test result.",
        "baseline_test_reproduced": True,
        "model_weights_unchanged": True,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "recalibration": information,
        "batchnorm_before": bn_before,
        "batchnorm_after": bn_after,
        "before": {"validation": before_val, "test": before_test},
        "after": {"validation": after_val, "test": after_test},
        "delta": {
            "validation": {k: after_val[k] - before_val[k] for k in metric_keys},
            "test": {k: after_test[k] - before_test[k] for k in metric_keys},
        },
    }
    del bn_copy
    return result


def run_one(run_key: str, args: argparse.Namespace, device: torch.device,
            label_names: list[str], split: dict) -> None:
    run_name = RUN_IDS[run_key]
    run_dir = ROOT / "runs" / run_name
    require_inputs(run_dir)
    checkpoint = torch.load(
        run_dir / "best_model.pt", map_location=device, weights_only=False
    )
    config = checkpoint.get("config", {})
    if config.get("experiment") != run_name:
        raise RuntimeError(f"Wrong checkpoint experiment: {config.get('experiment')}")
    if config.get("label_names") != label_names or checkpoint.get("label_names") != label_names:
        raise RuntimeError("Checkpoint label names/order != UVSPEC15 labels")
    if int(config.get("batch_size", -1)) != BATCH_SIZE or int(config.get("num_workers", -1)) != NUM_WORKERS:
        raise RuntimeError("Source run does not match batch_size=512/num_workers=4")
    if config.get("normalization") != "max" or float(config.get("threshold", -1)) != THRESHOLD:
        raise RuntimeError("Source run preprocessing/threshold differs from baseline")
    if bool(config.get("use_pos_weight")) != (run_key == "004"):
        raise RuntimeError("003/004 ordinary vs weighted BCE setup does not match")
    if run_key == "004" and (
        config.get("pos_weight_transform") != "sqrt"
        or float(config.get("pos_weight_cap", -1)) != 20.0
    ):
        raise RuntimeError("004 weighted-loss transformation/cap mismatch")
    model = make_model(checkpoint, len(label_names), device)
    initial_bn = summarize_bn(model)
    nonfinite = sum(r["nonfinite_mean"] + r["nonfinite_var"] for r in initial_bn)
    if nonfinite:
        raise RuntimeError(f"Checkpoint has {nonfinite} nonfinite BatchNorm stats")

    print("\n" + "=" * 90)
    print(f"{run_name} | {args.mode.upper()} BatchNorm diagnostics")
    print("=" * 90)
    print(f"Checkpoint best epoch: {checkpoint.get('epoch')} | "
          f"Best val mAP: {checkpoint.get('best_val_mAP')}")
    print(f"Device: {device} | Batch size: {BATCH_SIZE} | Workers: {NUM_WORKERS}")
    print(f"BN layers: {len(initial_bn)} | nonfinite running stats: {nonfinite}")
    out_dir = OUTPUT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("quick", "both"):
        print("\nQUICK: train/eval-mode comparison on fixed Validation subset")
        result = quick_diagnostic(model, split["val"], device, args.samples)
        result.update({
            "experiment": run_name,
            "source_checkpoint": str(run_dir / "best_model.pt"),
            "batchnorm_checkpoint": initial_bn,
        })
        dest = out_dir / "batchnorm_quick_diagnostic.json"
        dest.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Saved:", dest)

    if args.mode in ("full", "both"):
        print("\nFULL: recompute BN stats from TRAIN on disposable model copy")
        result = full_diagnostic(
            model, split["train"], split["val"], split["test"], device, run_dir
        )
        result.update({
            "experiment": run_name,
            "source_checkpoint": str(run_dir / "best_model.pt"),
        })
        dest = out_dir / "batchnorm_full_diagnostic.json"
        dest.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Saved:", dest)
    print("Original checkpoint untouched:", run_dir / "best_model.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose UVSPEC15 BatchNorm behavior (003/004).")
    parser.add_argument("--mode", choices=["quick", "full", "both"], default="quick")
    parser.add_argument("--run", choices=["003", "004", "both"], default="both")
    parser.add_argument("--samples", type=int, default=1024,
                        help="validation samples for quick diagnosis; default 1024")
    args = parser.parse_args()

    for path in (UV_PATH, LABEL_PATH, LABEL_NAMES_PATH, SPLIT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required data not found: {path}")
    with open(LABEL_NAMES_PATH, "r", encoding="utf-8") as handle:
        label_names = json.load(handle)
    if len(label_names) != 15:
        raise RuntimeError(f"Expected 15 UVSPEC labels; got {len(label_names)}")
    spectra = np.load(UV_PATH, mmap_mode="r")
    labels = np.load(LABEL_PATH, mmap_mode="r")
    if spectra.shape != (129817, 701) or labels.shape != (129817, 15):
        raise RuntimeError(f"Unexpected UV/label shapes: {spectra.shape} / {labels.shape}")

    with np.load(SPLIT_PATH) as loaded:
        split = {k: np.asarray(loaded[k], dtype=np.int64) for k in ("train", "val", "test")}
    if tuple(len(split[k]) for k in ("train", "val", "test")) != (103853, 12981, 12981):
        raise RuntimeError("Unexpected UV scaffold split sizes")
    all_indices = np.concatenate([split["train"], split["val"], split["test"]])
    if len(np.unique(all_indices)) != len(all_indices):
        raise RuntimeError("Train/validation/test indices overlap")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    for key in (["003", "004"] if args.run == "both" else [args.run]):
        run_one(key, args, device, label_names, split)
    print("\nALL REQUESTED DIAGNOSTICS COMPLETED. Frozen checkpoints were never written.")


if __name__ == "__main__":
    main()
