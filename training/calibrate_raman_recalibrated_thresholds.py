"""Raman recalibrated-checkpoint threshold calibration (UVSPEC15 v1 pipeline).

Sibling of training/calibrate_raman_original_thresholds.py. Same UVSPEC15 v1
original-checkpoint calibration algorithm, but the input is a train-only
BN-recalibrated checkpoint and thresholds are refitted from scratch on the
recalibrated probabilities (Original-BN thresholds are never reused).

No training, no BN recalibration, no change to model weights, label schema or
dataset split. Class thresholds are optimized ONLY on validation and then
frozen; Test is evaluated afterwards and never used to fit or select anything.

The recalibrated checkpoint's provenance is verified against its
bn_recalibration_manifest.json (source model, seed, loss, source SHA256,
recalibrated SHA256, label order) and its 0.5 baseline is cross-checked against
the approved Full BN Diagnostic record.

Outputs go to a NEW directory (runs/RAMAN-BN-THRESHOLD-CALIBRATION/<run_name>/);
Original and Recalibrated checkpoints are never modified.

Usage (from the SpectraDB root in the spectra-db environment), one model per run:
  python training/calibrate_raman_recalibrated_thresholds.py --loss bce  --seed 42
  python training/calibrate_raman_recalibrated_thresholds.py --loss bce  --seed 123
  python training/calibrate_raman_recalibrated_thresholds.py --loss bce  --seed 2026
  python training/calibrate_raman_recalibrated_thresholds.py --loss wbce --seed 42
  python training/calibrate_raman_recalibrated_thresholds.py --loss wbce --seed 123
  python training/calibrate_raman_recalibrated_thresholds.py --loss wbce --seed 2026
"""

from pathlib import Path
import argparse
import hashlib
import sys, json, csv, warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score,
    precision_recall_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.datasets.qm9s_spectrum_dataset import QM9SSpectrumDataset
from training.models.spectrum_cnn1d import SpectrumStructureModel
from training import raman_numeric_audit as numeric_audit

PREPARED = ROOT / "processed" / "qm9s" / "prepared"
RAMAN_PATH = PREPARED / "raman_float32.npy"
LABEL_PATH = PREPARED / "functional_group_labels.npy"
LABEL_NAMES_PATH = PREPARED / "functional_group_label_names.json"
SPLIT_PATH = PREPARED / "raman_scaffold_split_valid.npz"

RECALIBRATED_ROOT = ROOT / "runs" / "RAMAN-BN-RECALIBRATED"
DIAGNOSTIC_ROOT = ROOT / "runs" / "RAMAN-BN-DIAGNOSTICS"
OUTPUT_ROOT = ROOT / "runs" / "RAMAN-BN-THRESHOLD-CALIBRATION"

RUN_IDS = {
    "bce": {
        42: "EXP-RAMAN-001-FULL",
        123: "EXP-RAMAN-001-SEED123",
        2026: "EXP-RAMAN-001-SEED2026",
    },
    "wbce": {
        42: "EXP-RAMAN-002-FULL",
        123: "EXP-RAMAN-002-SEED123",
        2026: "EXP-RAMAN-002-SEED2026",
    },
}

NUM_WORKERS = 4  # Deliberately identical to the original Raman training runs.
BATCH_SIZE = 512
EMBEDDING_DIM = 256
NORMALIZATION = "max"
DEFAULT_THRESHOLD = 0.5

EXPECTED_N_LABELS = 14
EXPECTED_SPECTRA_SHAPE = (129817, 3501)
EXPECTED_LABEL_SHAPE = (129817, 14)
EXPECTED_SPLIT_SIZES = (103855, 12981, 12981)

EXPECTED_BN_LAYERS = 12
EXPECTED_BN_BUFFERS = 36

# --------------------------------------------------------------------------- #
# Diagnostic reproduction tolerance (gate policy raman-bn-gate-v3).
#
# A single 1e-3 tolerance applies to ALL metrics on BOTH splits, identical to IR v1 and
# UVSPEC15 v1. There is no mAP-specific exception and no "recorded variation" state: if
# the official FP16 numbers do not reproduce, this script reports a FAILURE.
#
# Context (not a justification for relaxation): the Phase 2 record was produced with
# cudnn.benchmark=True and cudnn.deterministic unset, so its FP16 mAP is one sample from a
# nondeterministic ensemble (measured spread 1.086e-3 on the same file). That is a
# property of the reference configuration and is handled at the methodology level
# (see training/verify_raman_bn_reproducibility.py), never by widening the tolerance.
# --------------------------------------------------------------------------- #
BASELINE_ATOL = numeric_audit.TOLERANCE  # single source of truth (1e-3, IR v1 parity)
GATE_POLICY_VERSION = numeric_audit.GATE_POLICY_VERSION  # single source of truth


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(model, loader, device, use_amp):
    model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["spectrum"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(x)["logits"]

            ys.append(y.cpu().numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())

    return (
        np.concatenate(ys).astype(np.int32),
        np.concatenate(ps).astype(np.float32),
    )


def overall_metrics(y_true, probs, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float32)

    if thresholds.ndim == 0:
        thresholds = np.full(
            y_true.shape[1],
            float(thresholds),
            dtype=np.float32,
        )

    pred = (probs >= thresholds[None, :]).astype(np.int32)

    return {
        "micro_f1": float(
            f1_score(y_true, pred, average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, pred, average="macro", zero_division=0)
        ),
        "micro_precision": float(
            precision_score(y_true, pred, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, pred, average="micro", zero_division=0)
        ),
    }


def threshold_free_metrics(y_true, probs):
    aps, aucs = [], []

    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        ps = probs[:, j]

        n_pos = int(yt.sum())
        n_neg = int(len(yt) - n_pos)

        if n_pos == 0:
            ap = np.nan
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ap = average_precision_score(yt, ps)

        if n_pos == 0 or n_neg == 0:
            auc = np.nan
        else:
            auc = roc_auc_score(yt, ps)

        aps.append(ap)
        aucs.append(auc)

    return {
        "mAP": float(np.nanmean(aps)),
        "macro_AUROC": float(np.nanmean(aucs)),
    }


def optimize_threshold(y_true, probs, fallback=0.5):
    n_pos = int(y_true.sum())

    if n_pos == 0 or n_pos == len(y_true):
        return float(fallback), np.nan

    p, r, thresholds = precision_recall_curve(y_true, probs)

    if len(thresholds) == 0:
        return float(fallback), np.nan

    p = p[:-1]
    r = r[:-1]

    denom = p + r
    f1 = np.divide(
        2 * p * r,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    )

    i = int(np.nanargmax(f1))
    return float(thresholds[i]), float(f1[i])


def per_label_metrics(y_true, probs, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float32)

    if thresholds.ndim == 0:
        thresholds = np.full(
            y_true.shape[1],
            float(thresholds),
            dtype=np.float32,
        )

    rows = []

    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        pred = (probs[:, j] >= thresholds[j]).astype(np.int32)

        rows.append({
            "positive_samples": int(yt.sum()),
            "negative_samples": int(len(yt) - yt.sum()),
            "threshold": float(thresholds[j]),
            "precision": float(
                precision_score(yt, pred, zero_division=0)
            ),
            "recall": float(
                recall_score(yt, pred, zero_division=0)
            ),
            "f1": float(
                f1_score(yt, pred, zero_division=0)
            ),
        })

    return rows


def verify_manifest(
    manifest,
    run_name,
    loss_key,
    seed,
    original_ckpt,
    recalibrated_ckpt,
    label_names,
    original_sha256,
    recalibrated_sha256,
):
    """The recalibrated artifact must be traceable to the requested run."""
    problems = []

    if manifest.get("run_directory") != run_name:
        problems.append(
            f"manifest.run_directory={manifest.get('run_directory')!r} != {run_name!r}"
        )
    if manifest.get("loss") != loss_key:
        problems.append(f"manifest.loss={manifest.get('loss')!r} != {loss_key!r}")
    if int(manifest.get("seed", -1)) != seed:
        problems.append(f"manifest.seed={manifest.get('seed')} != {seed}")
    if int(manifest.get("checkpoint_config_seed", -1)) != seed:
        problems.append(
            f"manifest.checkpoint_config_seed={manifest.get('checkpoint_config_seed')} != {seed}"
        )
    if Path(str(manifest.get("source_checkpoint", ""))).resolve() != original_ckpt.resolve():
        problems.append(
            f"manifest.source_checkpoint={manifest.get('source_checkpoint')!r} "
            f"!= {str(original_ckpt)!r}"
        )
    if manifest.get("source_sha256") != original_sha256:
        problems.append("manifest.source_sha256 does not match the original checkpoint")
    if Path(str(manifest.get("saved_checkpoint", ""))).resolve() != recalibrated_ckpt.resolve():
        problems.append(
            f"manifest.saved_checkpoint={manifest.get('saved_checkpoint')!r} "
            f"!= {str(recalibrated_ckpt)!r}"
        )
    if manifest.get("saved_sha256") != recalibrated_sha256:
        problems.append("manifest.saved_sha256 does not match the recalibrated checkpoint")
    changed = int(manifest.get("changed_bn_buffer_count", -1))
    if not 1 <= changed <= EXPECTED_BN_BUFFERS:
        problems.append(
            f"manifest.changed_bn_buffer_count={changed} outside 1..{EXPECTED_BN_BUFFERS}"
        )
    if int(manifest.get("batchnorm_layers", -1)) != EXPECTED_BN_LAYERS:
        problems.append(
            f"manifest.batchnorm_layers={manifest.get('batchnorm_layers')} != {EXPECTED_BN_LAYERS}"
        )

    config_summary = manifest.get("config_summary", {})
    if int(config_summary.get("num_labels", -1)) != len(label_names):
        problems.append("manifest config_summary.num_labels != 14")
    if int(config_summary.get("batch_size", -1)) != BATCH_SIZE:
        problems.append("manifest config_summary.batch_size != 512")
    if int(config_summary.get("num_workers", -1)) != NUM_WORKERS:
        problems.append("manifest config_summary.num_workers != 4")
    if config_summary.get("normalization") != NORMALIZATION:
        problems.append("manifest config_summary.normalization != 'max'")
    if float(config_summary.get("threshold", -1)) != DEFAULT_THRESHOLD:
        problems.append("manifest config_summary.threshold != 0.5")

    data_scale = manifest.get("data_scale", {})
    if int(data_scale.get("train_samples", -1)) != EXPECTED_SPLIT_SIZES[0]:
        problems.append("manifest data_scale.train_samples != 103855")
    if int(data_scale.get("num_labels", -1)) != len(label_names):
        problems.append("manifest data_scale.num_labels != 14")
    if int(data_scale.get("spectra_length", -1)) != EXPECTED_SPECTRA_SHAPE[1]:
        problems.append("manifest data_scale.spectra_length != 3501")

    if problems:
        raise RuntimeError(
            "Recalibration manifest verification failed:\n  " + "\n  ".join(problems)
        )


def verify_checkpoint(checkpoint, config, run_name, loss_key, seed, label_names):
    """Refuse to calibrate a checkpoint that does not match the requested run."""
    problems = []

    if len(label_names) != EXPECTED_N_LABELS or len(set(label_names)) != EXPECTED_N_LABELS:
        problems.append(f"expected {EXPECTED_N_LABELS} unique Raman labels, got {len(label_names)}")
    if checkpoint.get("label_names") != label_names:
        problems.append("checkpoint label_names do not match the Raman label order")
    if config.get("label_names") != label_names:
        problems.append("checkpoint config label_names mismatch")
    if config.get("experiment") != run_name:
        problems.append(
            f"checkpoint experiment {config.get('experiment')!r} != {run_name!r}"
        )
    if int(config.get("seed", -1)) != seed:
        problems.append(f"checkpoint seed {config.get('seed')} != {seed}")
    if bool(config.get("use_pos_weight")) != (loss_key == "wbce"):
        problems.append("bce/wbce ordinary vs weighted BCE setup does not match")
    if loss_key == "wbce":
        if config.get("pos_weight_transform") != "sqrt":
            problems.append("wbce checkpoint is missing pos_weight_transform='sqrt'")
        if float(config.get("pos_weight_cap", -1)) != 20.0:
            problems.append("wbce checkpoint is missing pos_weight_cap=20.0")
    if int(config.get("num_workers", -1)) != NUM_WORKERS:
        problems.append(f"num_workers {config.get('num_workers')} != {NUM_WORKERS}")
    if int(config.get("batch_size", -1)) != BATCH_SIZE:
        problems.append(f"batch_size {config.get('batch_size')} != {BATCH_SIZE}")
    if int(config.get("embedding_dim", -1)) != EMBEDDING_DIM:
        problems.append(f"embedding_dim {config.get('embedding_dim')} != {EMBEDDING_DIM}")
    if config.get("normalization") != NORMALIZATION:
        problems.append(f"normalization {config.get('normalization')!r} != {NORMALIZATION!r}")
    if float(config.get("threshold", -1)) != DEFAULT_THRESHOLD:
        problems.append(f"default threshold {config.get('threshold')} != {DEFAULT_THRESHOLD}")

    bn_meta = checkpoint.get("bn_recalibration", {})
    if not bn_meta:
        problems.append("checkpoint has no bn_recalibration provenance block")
    else:
        if int(bn_meta.get("seed", -1)) != seed:
            problems.append(f"checkpoint bn_recalibration.seed={bn_meta.get('seed')} != {seed}")
        if bn_meta.get("run_directory") != run_name:
            problems.append("checkpoint bn_recalibration.run_directory mismatch")

    if problems:
        raise RuntimeError(
            "Checkpoint verification failed:\n  " + "\n  ".join(problems)
        )


def verify_against_diagnostic(diagnostic, val_default, test_default, test_free, run_name):
    """Audit the threshold-0.5 baseline against the approved Full BN Diagnostic.

    Artifact correctness is NOT decided here (the manifest hash chain and the Phase 3
    artifact gate do that). This returns the three-class audit verdict
    (PASS / MAP_ONLY_NUMERIC_MISMATCH / OTHER_ANOMALY) and does not raise; the caller
    applies the manual-approval policy.
    """
    observed = {
        "validation.micro_f1": val_default["micro_f1"],
        "validation.macro_f1": val_default["macro_f1"],
        "validation.micro_precision": val_default["micro_precision"],
        "validation.micro_recall": val_default["micro_recall"],
        "test.micro_f1": test_default["micro_f1"],
        "test.macro_f1": test_default["macro_f1"],
        "test.micro_precision": test_default["micro_precision"],
        "test.micro_recall": test_default["micro_recall"],
        "test.mAP": test_free["mAP"],
        "test.macro_AUROC": test_free["macro_AUROC"],
    }
    after = diagnostic.get("after", {})
    recorded = {}
    for key in observed:
        split, name = key.split(".", 1)
        recorded[key] = after.get(split, {}).get(name)

    audit = numeric_audit.classify(
        observed, recorded,
        context=f"Phase 5 {run_name}: Recalibrated BN, 0.5 baseline vs Full BN Diagnostic",
    )
    audit["diagnostic_path"] = str(
        DIAGNOSTIC_ROOT / run_name / "batchnorm_full_diagnostic.json")
    audit["max_abs_diff"] = max(
        (row["abs_diff"] for row in audit["metrics"].values()), default=0.0)
    audit["checked"] = sorted(audit["metrics"])
    return audit


def main():
    parser = argparse.ArgumentParser(
        description="Raman recalibrated-checkpoint threshold calibration (UVSPEC15 v1 flow)."
    )
    parser.add_argument("--loss", choices=["bce", "wbce"], default="bce")
    parser.add_argument("--seed", type=int, choices=[42, 123, 2026], default=42)
    parser.add_argument("--manual-approval-note", default=None,
                        help="resume authorisation only: required to resume an interrupted "
                             "calibration whose thresholds are already frozen. It is NOT an "
                             "anomaly waiver - the checkpoint release lives in the manifest "
                             "plus manual_approval.json, both bound to the checkpoint sha256.")
    args = parser.parse_args()

    if args.seed not in RUN_IDS[args.loss]:
        parser.error(
            f"loss={args.loss} has no configured seed {args.seed} "
            f"(available: {sorted(RUN_IDS[args.loss])})"
        )

    run_name = RUN_IDS[args.loss][args.seed]
    run_dir = ROOT / "runs" / run_name
    original_ckpt = run_dir / "best_model.pt"
    recal_dir = RECALIBRATED_ROOT / run_name
    recalibrated_ckpt = recal_dir / "best_model_bn_recalibrated.pt"
    manifest_path = recal_dir / "bn_recalibration_manifest.json"
    diagnostic_path = DIAGNOSTIC_ROOT / run_name / "batchnorm_full_diagnostic.json"
    out_dir = OUTPUT_ROOT / run_name

    # Never overwrite a completed calibration. An INTERRUPTED directory (thresholds frozen,
    # summary absent) may only be resumed with an explicit manual-approval note; the frozen
    # thresholds are then reused instead of being refitted.
    dir_state = numeric_audit.calibration_dir_state(
        out_dir, "optimal_thresholds.json", "threshold_calibration_summary.json")
    resume = False
    if dir_state == "complete":
        raise FileExistsError(
            f"Calibration output already exists and is complete: {out_dir}. "
            "Nothing was overwritten. Inspect or relocate it manually."
        )
    if dir_state == "interrupted":
        if not args.manual_approval_note:
            raise SystemExit(
                f"STOP: an interrupted calibration exists at {out_dir} "
                "(thresholds frozen, no summary). Re-run with "
                "--manual-approval-note \"...\" to resume; the frozen thresholds will be "
                "reused and will NOT be refitted."
            )
        resume = True
        print(f"RESUME: reusing the frozen thresholds in {out_dir}")
    elif dir_state == "unknown":
        raise FileExistsError(
            f"Directory {out_dir} exists but contains neither a frozen threshold file nor a "
            "summary. Refusing to touch it; inspect manually."
        )

    required = [
        RAMAN_PATH,
        LABEL_PATH,
        LABEL_NAMES_PATH,
        SPLIT_PATH,
        original_ckpt,
        recalibrated_ckpt,
        manifest_path,
        diagnostic_path,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file:\n{path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print()
    print("=" * 80)
    print(f"{run_name} | recalibrated-checkpoint threshold calibration")
    print("=" * 80)
    print()
    print("Device :", device)
    if device.type == "cuda":
        print("GPU    :", torch.cuda.get_device_name(0))
        print("CUDA   :", torch.version.cuda)
    print("PyTorch:", torch.__version__)
    print("AMP    :", use_amp)
    print("Workers:", NUM_WORKERS)

    with open(LABEL_NAMES_PATH, "r", encoding="utf-8") as f:
        label_names = json.load(f)

    split = np.load(SPLIT_PATH)
    train_idx = np.asarray(split["train"], dtype=np.int64)
    val_idx = np.asarray(split["val"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)

    spectra_check = np.load(RAMAN_PATH, mmap_mode="r")
    label_check = np.load(LABEL_PATH, mmap_mode="r")
    if tuple(spectra_check.shape) != EXPECTED_SPECTRA_SHAPE:
        raise ValueError(f"Unexpected Raman matrix shape: {spectra_check.shape}")
    if tuple(label_check.shape) != EXPECTED_LABEL_SHAPE:
        raise ValueError(f"Unexpected Raman label matrix shape: {label_check.shape}")
    if (len(train_idx), len(val_idx), len(test_idx)) != EXPECTED_SPLIT_SIZES:
        raise ValueError(
            f"Unexpected Raman scaffold split sizes: "
            f"{(len(train_idx), len(val_idx), len(test_idx))}"
        )

    original_sha256 = sha256_file(original_ckpt)
    recalibrated_sha256 = sha256_file(recalibrated_ckpt)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest(
        manifest,
        run_name,
        args.loss,
        args.seed,
        original_ckpt,
        recalibrated_ckpt,
        label_names,
        original_sha256,
        recalibrated_sha256,
    )
    print()
    print("Manifest          :", manifest_path)
    print("Source checkpoint :", original_ckpt)
    print("Source sha256     :", original_sha256)
    print("Recal checkpoint  :", recalibrated_ckpt)
    print("Recal sha256      :", recalibrated_sha256)
    print(f"Changed BN buffers: {manifest['changed_bn_buffer_count']}/{EXPECTED_BN_BUFFERS}")
    print("PASS: recalibration manifest verified (source model, seed, loss, hashes, labels)")

    checkpoint = torch.load(
        recalibrated_ckpt,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint.get("config", {})
    verify_checkpoint(checkpoint, config, run_name, args.loss, args.seed, label_names)

    if checkpoint["bn_recalibration"].get("source_sha256") != original_sha256:
        raise RuntimeError(
            "Checkpoint bn_recalibration.source_sha256 does not match the current "
            "original checkpoint; the source model may have changed."
        )

    batch_size = int(config["batch_size"])
    embedding_dim = int(config["embedding_dim"])
    normalization = config["normalization"]
    default_threshold = float(config["threshold"])

    print()
    print("Validation samples:", len(val_idx))
    print("Test samples      :", len(test_idx))
    print("Loss / Seed       :", args.loss, "/", args.seed)
    print("Best epoch        :", checkpoint.get("epoch", "unknown"))
    print("Best val mAP      :", checkpoint.get("best_val_mAP", "unknown"))
    print("Batch size        :", batch_size)
    print("Normalization     :", normalization)
    print("Default threshold :", default_threshold)
    print("PASS: checkpoint source, seed, loss config and label order verified")

    # Resume: if the first run already froze its probability arrays, load them and perform
    # NO inference at all. A resumed run may never replace the first observation.
    frozen_probs = None
    if resume:
        try:
            frozen_probs = numeric_audit.load_frozen_probabilities(
                out_dir, recalibrated_sha256, label_names,
                {"validation": int(len(val_idx)), "test": int(len(test_idx))},
            )
            print()
            print("RESUME: loaded the FROZEN probability arrays; no inference is performed.")
            for _name in ("validation", "test"):
                _e = frozen_probs["manifest"]["arrays"][_name]
                print(f"  {_name:10s} {_e['file']}  sha256={_e['sha256'][:16]}  "
                      f"shape={tuple(_e['shape'])}")
        except numeric_audit.ProbabilityArtifactError as exc:
            print(f"RESUME: no usable frozen arrays ({exc}); a fresh inference will be used "
                  "and will become the first probability observation")

    if frozen_probs is not None:
        val_p = frozen_probs["validation"]
        test_p = frozen_probs["test"]
        val_y = np.asarray(label_check[val_idx], dtype=np.float32)
        test_y = np.asarray(label_check[test_idx], dtype=np.float32)
        probability_artifacts = frozen_probs["manifest"]
        probability_source = "resumed_from_frozen_arrays"
    else:
        val_ds = QM9SSpectrumDataset(
            spectra_path=RAMAN_PATH,
            labels_path=LABEL_PATH,
            indices=val_idx,
            normalization=normalization,
        )

        test_ds = QM9SSpectrumDataset(
            spectra_path=RAMAN_PATH,
            labels_path=LABEL_PATH,
            indices=test_idx,
            normalization=normalization,
        )

        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": NUM_WORKERS,
            "pin_memory": device.type == "cuda",
            "persistent_workers": NUM_WORKERS > 0,
            "prefetch_factor": 2,
        }

        val_loader = DataLoader(val_ds, **loader_kwargs)
        test_loader = DataLoader(test_ds, **loader_kwargs)

        model = SpectrumStructureModel(
            num_labels=len(label_names),
            embedding_dim=embedding_dim,
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        print()
        print("Collecting validation probabilities...")
        val_y, val_p = collect(model, val_loader, device, use_amp)

        print("Collecting test probabilities...")
        test_y, test_p = collect(model, test_loader, device, use_amp)

        probability_source = "first_run"

    thresholds = []
    val_best_f1 = []

    if resume:
        frozen_path = out_dir / "optimal_thresholds.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen.get("checkpoint_sha256") != recalibrated_sha256:
            raise SystemExit(
                "STOP: the frozen thresholds belong to a different recalibrated checkpoint "
                f"({frozen.get('checkpoint_sha256')} != {recalibrated_sha256})."
            )
        if frozen.get("bn_state") != "recalibrated":
            raise SystemExit("STOP: frozen thresholds are not for a recalibrated checkpoint")
        missing = [n for n in label_names if n not in frozen.get("thresholds", {})]
        if missing:
            raise SystemExit(f"STOP: frozen thresholds are missing labels: {missing}")
        thresholds = np.asarray([frozen["thresholds"][n] for n in label_names],
                                dtype=np.float32)
        val_best_f1 = [frozen.get("validation_best_f1", {}).get(n) for n in label_names]
        print()
        print("RESUME: thresholds reused from the interrupted run; NOT refitted.")
        print(" ", frozen_path)
    else:
        print()
        print("=" * 80)
        print("VALIDATION-OPTIMIZED PER-LABEL THRESHOLDS (refitted, recalibrated BN)")
        print("=" * 80)
        print()
        print(f"{'label':16s}{'positive':>10s}{'threshold':>12s}{'best_F1':>12s}")
        print("-" * 50)

        for j, name in enumerate(label_names):
            threshold, best_f1 = optimize_threshold(
                val_y[:, j],
                val_p[:, j],
                fallback=default_threshold,
            )

            thresholds.append(threshold)
            val_best_f1.append(best_f1)

            best_text = "nan" if np.isnan(best_f1) else f"{best_f1:.4f}"

            print(
                f"{name:16s}"
                f"{int(val_y[:, j].sum()):10d}"
                f"{threshold:12.4f}"
                f"{best_text:>12s}"
            )

        thresholds = np.asarray(thresholds, dtype=np.float32)

        # The thresholds are fitted strictly on Validation - from the recalibrated
        # probabilities, never from the Original-BN threshold files - and persisted
        # BEFORE any Test metric is computed.
        out_dir.mkdir(parents=True, exist_ok=False)

        thresholds_json = {
            "experiment": run_name,
            "loss": args.loss,
            "seed": args.seed,
            "bn_state": "recalibrated",
            "checkpoint": str(recalibrated_ckpt),
            "checkpoint_sha256": recalibrated_sha256,
            "original_checkpoint": str(original_ckpt),
            "original_sha256": original_sha256,
            "bn_recalibration_manifest": str(manifest_path),
            "source": "validation split only; refitted on recalibrated BN probabilities",
            "default_threshold": default_threshold,
            "thresholds": {
                name: float(thresholds[j])
                for j, name in enumerate(label_names)
            },
            "validation_best_f1": {
                name: (
                    None
                    if np.isnan(val_best_f1[j])
                    else float(val_best_f1[j])
                )
                for j, name in enumerate(label_names)
            },
            "frozen_before_test_metrics": True,
            "numeric_audit_policy": numeric_audit.GATE_POLICY_VERSION,
        }

        with open(out_dir / "optimal_thresholds.json", "w", encoding="utf-8") as f:
            json.dump(thresholds_json, f, ensure_ascii=False, indent=2)

        print()
        print("Thresholds frozen on Validation and saved before Test evaluation:")
        print(" ", out_dir / "optimal_thresholds.json")

    # One inference per split; both threshold conditions are derived from these arrays.
    if probability_source == "first_run":
        probability_artifacts = numeric_audit.persist_probabilities(
            out_dir,
            run_name,
            {"validation": val_p, "test": test_p},
            {
                "checkpoint": str(recalibrated_ckpt),
                "checkpoint_sha256": recalibrated_sha256,
                "original_checkpoint_sha256": original_sha256,
                "bn_state": "recalibrated",
                "splits": {"validation": int(len(val_idx)), "test": int(len(test_idx))},
                "dtype": "float32",
                "metric_basis": ("official FP16 sigmoid at threshold 0.5 and threshold-free; "
                                 "both threshold conditions reuse the same array"),
                "label_names": label_names,
                "batch_size": batch_size,
                "num_workers": NUM_WORKERS,
                "normalization": normalization,
                "amp": use_amp,
            },
        )

    val_default = overall_metrics(val_y, val_p, default_threshold)
    val_calibrated = overall_metrics(val_y, val_p, thresholds)
    test_default = overall_metrics(test_y, test_p, default_threshold)
    test_calibrated = overall_metrics(test_y, test_p, thresholds)
    test_free = threshold_free_metrics(test_y, test_p)

    # Independent reproducibility check against the approved Full BN Diagnostic.
    # Do NOT use the test scores to select or change calibration thresholds.
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))

    # ---- release check: artifact integrity AND the manifest's ablation_release --------
    approval = numeric_audit.load_approval(recal_dir, recalibrated_sha256,
                                           manifest.get("saved_sha256"))
    decision = numeric_audit.decide_phase5_release(manifest, approval)
    release_status = (manifest.get("ablation_release") or {}).get("status")
    print()
    print(f"RELEASE GATE: mode={decision['mode']} allowed={decision['allowed']}")
    print(f"  {decision['reason']}")
    if not decision["allowed"]:
        raise SystemExit(
            "STOP: this checkpoint is not released for the formal ablation. "
            f"{decision['reason']}\nRecord a sha-bound release with:\n"
            "  python training\\save_raman_bn_checkpoint.py --loss "
            f"{args.loss} --seed {args.seed} --approve-release "
            "--manual-approval-note \"<who / when / on what evidence>\" "
            "--approve-checkpoint-sha256 <checkpoint sha256>"
        )
    if decision["mode"] == "manual_approval":
        print(f"  approval record : {decision['approval_path']}")
        print(f"  approval note   : {decision['approval_note']}")

    audit = verify_against_diagnostic(
        diagnostic, val_default, test_default, test_free, run_name
    )
    authoritative, first_observation, record_file = numeric_audit.resolve_authoritative(
        run_name, "recalibrated_thresholds", audit
    )
    split = numeric_audit.split_release_and_anomaly(authoritative, "validation.", "test.")
    numeric_audit.print_audit(
        authoritative, record_file, first_observation,
        extra={"validation_release": split["validation_release"],
               "test_anomaly": split["test_anomaly"]})

    # The Validation side is the technical release condition; it has no approval path.
    if split["validation_release"]["status"] != numeric_audit.PASS:
        raise SystemExit(
            "STOP: the Validation-side release condition failed "
            f"(exceeding: {split['validation_release']['exceeding_metrics']}). A "
            "threshold-based metric moved, which no manual approval can cover."
        )
    # The Test side is a record only: it selects nothing and releases nothing.
    if split["test_anomaly"]["status"] != numeric_audit.PASS:
        print("  TEST-SIDE ANOMALY RECORDED ONLY. It selects no model, no threshold and no")
        print("  inference run; Test-side consistency is enforced by Phase 6's uniform")
        print("  0.0025 cross-check, not here.")

    print()
    print("=" * 80)
    print("VALIDATION COMPARISON")
    print("=" * 80)
    print()
    print(
        "Default 0.5 : "
        f"Micro-F1={val_default['micro_f1']:.4f} | "
        f"Macro-F1={val_default['macro_f1']:.4f} | "
        f"Precision={val_default['micro_precision']:.4f} | "
        f"Recall={val_default['micro_recall']:.4f}"
    )
    print(
        "Calibrated  : "
        f"Micro-F1={val_calibrated['micro_f1']:.4f} | "
        f"Macro-F1={val_calibrated['macro_f1']:.4f} | "
        f"Precision={val_calibrated['micro_precision']:.4f} | "
        f"Recall={val_calibrated['micro_recall']:.4f}"
    )

    print()
    print("=" * 80)
    print("TEST COMPARISON")
    print("=" * 80)
    print()
    print(
        "Default 0.5 : "
        f"Micro-F1={test_default['micro_f1']:.4f} | "
        f"Macro-F1={test_default['macro_f1']:.4f} | "
        f"Precision={test_default['micro_precision']:.4f} | "
        f"Recall={test_default['micro_recall']:.4f}"
    )
    print(
        "Calibrated  : "
        f"Micro-F1={test_calibrated['micro_f1']:.4f} | "
        f"Macro-F1={test_calibrated['macro_f1']:.4f} | "
        f"Precision={test_calibrated['micro_precision']:.4f} | "
        f"Recall={test_calibrated['micro_recall']:.4f}"
    )
    print(
        f"Threshold-free: mAP={test_free['mAP']:.4f} | "
        f"Macro-AUROC={test_free['macro_AUROC']:.4f}"
    )

    default_rows = per_label_metrics(test_y, test_p, default_threshold)
    calibrated_rows = per_label_metrics(test_y, test_p, thresholds)
    validation_default_rows = per_label_metrics(val_y, val_p, default_threshold)
    validation_calibrated_rows = per_label_metrics(val_y, val_p, thresholds)

    with open(
        out_dir / "validation_threshold_comparison.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "label_index", "label", "positive_val_samples",
            "optimal_threshold", "validation_best_f1",
            "default_precision", "default_recall", "default_f1",
            "calibrated_precision", "calibrated_recall", "calibrated_f1",
            "f1_change",
        ])
        for j, name in enumerate(label_names):
            d = validation_default_rows[j]
            c = validation_calibrated_rows[j]
            writer.writerow([
                j, name, d["positive_samples"], float(thresholds[j]),
                "" if np.isnan(val_best_f1[j]) else float(val_best_f1[j]),
                d["precision"], d["recall"], d["f1"],
                c["precision"], c["recall"], c["f1"], c["f1"] - d["f1"],
            ])

    with open(
        out_dir / "threshold_comparison.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "label_index",
            "label",
            "positive_val_samples",
            "optimal_threshold",
            "val_best_f1",
            "positive_test_samples",
            "default_precision",
            "default_recall",
            "default_f1",
            "calibrated_precision",
            "calibrated_recall",
            "calibrated_f1",
            "f1_change",
        ])
        for j, name in enumerate(label_names):
            d = default_rows[j]
            c = calibrated_rows[j]
            writer.writerow([
                j,
                name,
                int(val_y[:, j].sum()),
                float(thresholds[j]),
                "" if np.isnan(val_best_f1[j]) else float(val_best_f1[j]),
                d["positive_samples"],
                d["precision"],
                d["recall"],
                d["f1"],
                c["precision"],
                c["recall"],
                c["f1"],
                c["f1"] - d["f1"],
            ])

    with open(
        out_dir / "test_per_label_metrics_calibrated.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "label_index",
            "label",
            "positive_test_samples",
            "negative_test_samples",
            "threshold",
            "precision",
            "recall",
            "f1",
        ])
        for j, name in enumerate(label_names):
            row = calibrated_rows[j]
            writer.writerow([
                j,
                name,
                row["positive_samples"],
                row["negative_samples"],
                row["threshold"],
                row["precision"],
                row["recall"],
                row["f1"],
            ])

    with open(
        out_dir / "test_per_label_metrics_default.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "label_index",
            "label",
            "positive_test_samples",
            "negative_test_samples",
            "threshold",
            "precision",
            "recall",
            "f1",
        ])
        for j, name in enumerate(label_names):
            row = default_rows[j]
            writer.writerow([
                j,
                name,
                row["positive_samples"],
                row["negative_samples"],
                row["threshold"],
                row["precision"],
                row["recall"],
                row["f1"],
            ])

    summary = {
        "experiment": run_name,
        "loss": args.loss,
        "seed": args.seed,
        "bn_state": "recalibrated",
        "checkpoint": str(recalibrated_ckpt),
        "checkpoint_sha256": recalibrated_sha256,
        "original_checkpoint": str(original_ckpt),
        "original_sha256": original_sha256,
        "bn_recalibration_manifest": str(manifest_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "changed_bn_buffer_count": int(manifest["changed_bn_buffer_count"]),
        "label_set": "RAMAN_FG14",
        "num_labels": len(label_names),
        "batch_size": batch_size,
        "num_workers": NUM_WORKERS,
        "normalization": normalization,
        "calibration": (
            "per-label F1 optimization on validation split, refitted on the "
            "recalibrated BN probabilities"
        ),
        "default_threshold": default_threshold,
        "validation_default": val_default,
        "validation_calibrated": val_calibrated,
        "test_default": test_default,
        "test_calibrated": test_calibrated,
        "test_threshold_free": test_free,
        "diagnostic_reference": authoritative,
        "probability_artifacts": probability_artifacts,
        "probability_source": probability_source,
        "numeric_audit": authoritative,
        "validation_release": split["validation_release"],
        "test_anomaly": split["test_anomaly"],
        "manifest_release": {
            "status": release_status,
            "approval": approval,
            "manifest": str(manifest_path),
        },
        "resumed_from_frozen_thresholds": bool(resume),
        "resumed_from_frozen_probabilities":
            probability_source == "resumed_from_frozen_arrays",
        "note": (
            "Thresholds were selected on validation data only and frozen before any "
            "test metric was computed. mAP and AUROC are threshold-free and therefore "
            "unchanged by the threshold choice. Original-BN thresholds were not reused."
        ),
    }

    with open(
        out_dir / "threshold_calibration_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 80)
    print("Threshold calibration completed")
    print("=" * 80)
    print()
    print("Saved (new directory, nothing overwritten):")
    for name in (
        "optimal_thresholds.json",
        "validation_threshold_comparison.csv",
        "threshold_comparison.csv",
        "test_per_label_metrics_calibrated.csv",
        "test_per_label_metrics_default.csv",
        "threshold_calibration_summary.json",
    ):
        print(" ", out_dir / name)
    print("Original checkpoint untouched:", original_ckpt)
    print("Recalibrated checkpoint untouched:", recalibrated_ckpt)


if __name__ == "__main__":
    main()
