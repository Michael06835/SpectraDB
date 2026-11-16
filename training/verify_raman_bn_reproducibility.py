"""Auxiliary, NON-GATING reproducibility verification for Raman BN checkpoints.

Why this script exists
----------------------
The Phase 2 Full BN Diagnostic records were produced with
`torch.backends.cudnn.benchmark = True` and `cudnn.deterministic` unset. Under that
configuration cuDNN picks a convolution algorithm by timing, so two processes are not
guaranteed to select the same kernel. The T1/T2 probe (2026-11-16, EXP-RAMAN-001-FULL)
measured, for the SAME checkpoint file:

    official FP16 sigmoid mAP : 0.945208 and 0.946288   (spread 1.086e-3)
    FP32 sigmoid mAP          : 0.946809 and 0.946807   (spread 2e-6)

i.e. the recorded FP16 mAP is one draw from a nondeterministic ensemble, and the drift is
concentrated in `nitro` (11 Validation positives) because FP16 sigmoid quantisation
re-orders near-tied scores.

This script does NOT gate anything and does NOT relax any acceptance rule. It measures a
different, strictly separate question:

    "With the convolution algorithm pinned, does this checkpoint reproduce itself
     bit-for-bit across independent processes?"

Deterministic configuration applied here (recorded in every summary):
    CUBLAS_WORKSPACE_CONFIG=:4096:8 (set before CUDA initialisation)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

Two channels are reported and NEVER mixed:

    fp16_sigmoid_deterministic : sigmoid(logits) on the logits dtype, i.e. the same op as
                                 the historical protocol, but under the deterministic
                                 configuration, so its numbers are NOT comparable with the
                                 Phase 2 record and must never be substituted for it.
    fp32_sigmoid_deterministic : sigmoid(logits.float()), a quantisation-free view.

`compare` refuses to compare across channels: a cross-channel difference is a protocol
difference, not a reproducibility measurement.

Outputs (exclusive create, system TEMP only, never runs/):
    <temp>/raman-bn-reproducibility/<tag>/<tag>__rN__summary.json
    <temp>/raman-bn-reproducibility/<tag>/<tag>__rN__probs__<channel>.npy
    <temp>/raman-bn-reproducibility/<tag>/<tag>__rN__labels.npy

Usage (from the SpectraDB root, spectra-db environment), one process per invocation:
    python training\\verify_raman_bn_reproducibility.py run --checkpoint orphan --tag RA
    python training\\verify_raman_bn_reproducibility.py run --checkpoint orphan --tag RB
    python training\\verify_raman_bn_reproducibility.py run --checkpoint orphan --tag RC --repeat 2
    python training\\verify_raman_bn_reproducibility.py compare --tag-a RA --tag-b RB
    python training\\verify_raman_bn_reproducibility.py run --checkpoint orphan --tag RDRY --dry-run
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

# Must be set before any CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import diagnose_raman_batchnorm_full as diag  # noqa: E402
from training.datasets.qm9s_spectrum_dataset import QM9SSpectrumDataset  # noqa: E402

TEMP_ROOT = Path(tempfile.gettempdir()) / "raman-bn-reproducibility"

CHECKPOINTS = {
    "original": (
        ROOT / "runs" / "EXP-RAMAN-001-FULL" / "best_model.pt",
        "573ae7e2a3351252c5f1ad7e0ec060c77435468d3c7bddb37a641e1206cdfcd3",
    ),
    "orphan": (
        ROOT / "runs" / "RAMAN-BN-RECALIBRATED" / "EXP-RAMAN-001-FULL"
        / "best_model_bn_recalibrated.pt",
        "a9f9d55f1239abd7c91f52a4900f3ba5f23a9f4ced799b626802d956197022f5",
    ),
}

EXPECTED_VAL = 12981
CHANNELS = ("fp16_sigmoid_deterministic", "fp32_sigmoid_deterministic")
METRIC_KEYS = ("micro_f1", "macro_f1", "micro_precision", "micro_recall",
               "mAP", "macro_AUROC")

DETERMINISM_ACTIONS = [
    "CUBLAS_WORKSPACE_CONFIG=:4096:8 (before CUDA init)",
    "torch.backends.cudnn.benchmark = False",
    "torch.backends.cudnn.deterministic = True",
    "torch.use_deterministic_algorithms(True)",
    "torch.backends.cuda.matmul.allow_tf32 = False",
    "torch.backends.cudnn.allow_tf32 = False",
]


# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(a: np.ndarray) -> str:
    b = np.ascontiguousarray(np.asarray(a).astype("<f4", copy=False))
    return hashlib.sha256(b.tobytes(order="C")).hexdigest()


def guard_out_root(p: Path) -> Path:
    rp = p.resolve()
    root = ROOT.resolve()
    if rp == root or root in rp.parents:
        raise SystemExit(f"REFUSED: output root {rp} is inside the project tree")
    if "runs" in {part.lower() for part in rp.parts}:
        raise SystemExit(f"REFUSED: output root {rp} contains a 'runs' component")
    return rp


def next_run_id(out_dir: Path, tag: str) -> str:
    used = 0
    if out_dir.is_dir():
        for f in out_dir.glob(f"{tag}__r*"):
            mid = f.name[len(tag) + 2:].split("__")[0]
            if mid.startswith("r") and mid[1:].isdigit():
                used = max(used, int(mid[1:]))
    return f"r{used + 1}"


def write_new_binary(path: Path, writer) -> None:
    with path.open("xb") as fh:
        writer(fh)


def write_new_text(path: Path, writer) -> None:
    with path.open("x", encoding="utf-8") as fh:
        writer(fh)


def configure_determinism() -> dict:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "actions": DETERMINISM_ACTIONS,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


# --------------------------------------------------------------------------- #
def collect(model, loader, device):
    """One Validation pass; both channels come from the same logits."""
    y_l, c_fp16, c_fp32, logit_l = [], [], [], []
    logits_dtype = fp16_dtype = None
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
            p_fp16 = torch.sigmoid(logits)              # historical op, deterministic run
            p_fp32 = torch.sigmoid(logits.float())      # quantisation-free view
            if logits_dtype is None:
                logits_dtype = str(logits.dtype)
                fp16_dtype = str(p_fp16.dtype)
            y_l.append(y.detach().cpu().numpy())
            c_fp16.append(p_fp16.detach().float().cpu().numpy())
            c_fp32.append(p_fp32.detach().float().cpu().numpy())
            logit_l.append(logits.detach().float().cpu().numpy())

    labels = np.concatenate(y_l, axis=0).astype(np.int32)
    out = {
        CHANNELS[0]: np.concatenate(c_fp16, axis=0),
        CHANNELS[1]: np.concatenate(c_fp32, axis=0),
    }
    logits_np = np.concatenate(logit_l, axis=0)
    for name, arr in list(out.items()) + [("logits", logits_np)]:
        if not np.isfinite(arr).all():
            raise RuntimeError(f"non-finite {name}")
    return labels, logits_np, out, logits_dtype, fp16_dtype


def channel_metrics(labels, probs, label_names):
    metrics = diag.get_metrics(labels, probs)
    per_label = []
    for j, name in enumerate(label_names):
        y = labels[:, j].astype(np.int32)
        s = probs[:, j]
        pos = int(y.sum())
        ap = auroc = None
        if 0 < pos < y.size:
            ap = float(average_precision_score(y, s))
            auroc = float(roc_auc_score(y, s))
        per_label.append({
            "label": name,
            "positives": pos,
            "average_precision": ap,
            "auroc": auroc,
            "unique_probabilities": int(np.unique(s).size),
            "largest_tie_group": int(np.unique(s, return_counts=True)[1].max()),
        })
    return metrics, per_label


# --------------------------------------------------------------------------- #
def cmd_run(args) -> int:
    ckpt_path, expected = CHECKPOINTS[args.checkpoint]
    out_root = guard_out_root(Path(args.out_root) if args.out_root else TEMP_ROOT)
    out_dir = out_root / args.tag

    print("=" * 84)
    print("Auxiliary reproducibility verification (NON-GATING, deterministic configuration)")
    print(f"  tag        : {args.tag}")
    print(f"  checkpoint : {args.checkpoint}  ({ckpt_path})")
    print(f"  output     : {out_root}")
    print("  This script does not gate, does not compare with the Phase 2 record, and")
    print("  never mixes its two channels. The official acceptance path is")
    print("  training/save_raman_bn_checkpoint.py.")
    print("=" * 84)

    if not ckpt_path.is_file():
        raise SystemExit(f"STOP: checkpoint not found: {ckpt_path}")
    sha = sha256_file(ckpt_path)
    print(f"  sha256     : {sha}")
    if not args.allow_any_checkpoint and sha != expected:
        raise SystemExit("STOP: checkpoint sha256 differs from the audited value")
    print("  PASS: checkpoint sha256 matches the audited value")

    with np.load(diag.SPLIT_PATH) as sp:
        val_idx = np.asarray(sp["val"], dtype=np.int64)
    if val_idx.size != EXPECTED_VAL or len(np.unique(val_idx)) != val_idx.size:
        raise SystemExit("STOP: unexpected validation index set")
    print(f"  PASS: validation rows = {val_idx.size}; Test split never loaded")

    label_names = json.loads(diag.LABEL_NAMES_PATH.read_text(encoding="utf-8"))
    run_id = next_run_id(out_dir, args.tag)
    planned = [f"{args.tag}__{run_id}__summary.json", f"{args.tag}__{run_id}__labels.npy"]
    planned += [f"{args.tag}__{run_id}__probs__{c}.npy" for c in CHANNELS]
    if args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\n  DRY RUN: deterministic configuration that WOULD be applied:")
        for action in DETERMINISM_ACTIONS:
            print(f"    - {action}")
        print(f"  next run id: {run_id}")
        for name in planned:
            target = out_dir / name
            print(f"  will create: {target}   exists={target.exists()}")
        print("  All targets are exclusive-create ('xb'); nothing is overwritten.")
        return 0

    det = configure_determinism()
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("STOP: --device cuda requested but CUDA is unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("  WARNING: CPU run. This validates the plumbing only; it says nothing about")
        print("           the GPU fp16 path that the official gate uses.")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = diag.make_model(checkpoint, len(label_names), device)
    loader = diag.make_loader(val_idx, device)
    print(f"  device={device} | logits.dtype<expected fp16 on CUDA> | deterministic={det['deterministic_algorithms']}")

    passes = []
    for k in range(args.repeat):
        labels, logits, channels, logits_dtype, fp16_dtype = collect(model, loader, device)
        entry = {"repeat_index": k + 1, "logits_dtype": logits_dtype,
                 "fp16_sigmoid_dtype": fp16_dtype, "channels": {}}
        for channel, probs in channels.items():
            metrics, per_label = channel_metrics(labels, probs, label_names)
            entry["channels"][channel] = {
                "metrics": metrics,
                "per_label": per_label,
                "probs_sha256": array_sha256(probs),
                "logits_sha256": array_sha256(logits),
                "labels_sha256": array_sha256(labels.astype(np.float32)),
                "probs_mean": float(probs.mean()),
                "probs_max": float(probs.max()),
            }
        passes.append((entry, channels, labels, logits))
        print(f"\n  --- pass {k + 1}/{args.repeat} (logits {logits_dtype}, "
              f"fp16 sigmoid {fp16_dtype}) ---")
        for channel in CHANNELS:
            m = entry["channels"][channel]["metrics"]
            print(f"    {channel:26s} mAP={m['mAP']:.6f} microF1={m['micro_f1']:.6f} "
                  f"AUROC={m['macro_AUROC']:.6f} sha256={entry['channels'][channel]['probs_sha256'][:16]}")

    in_process = None
    if args.repeat > 1:
        a, b = passes[0], passes[1]
        in_process = {}
        for channel in CHANNELS:
            pa, pb = a[1][channel], b[1][channel]
            d = np.abs(pa - pb)
            in_process[channel] = {
                "probs_bitwise_equal": bool(np.array_equal(pa, pb)),
                "max_abs_prob_diff": float(d.max()),
                "elements_differing": int((d > 0).sum()),
                "map_pass1": a[0]["channels"][channel]["metrics"]["mAP"],
                "map_pass2": b[0]["channels"][channel]["metrics"]["mAP"],
            }
        print("\n  in-process repeat:")
        for channel in CHANNELS:
            r = in_process[channel]
            print(f"    {channel:26s} bitwise_equal={r['probs_bitwise_equal']} "
                  f"max|d|={r['max_abs_prob_diff']:.3e} "
                  f"dMAP={r['map_pass2'] - r['map_pass1']:+.6f}")

    if sha256_file(ckpt_path) != sha:
        raise SystemExit("CRITICAL: checkpoint changed during the run")
    print("\n  PASS: checkpoint unchanged after the run")

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"labels": str(out_dir / f"{args.tag}__{run_id}__labels.npy")}
    write_new_binary(Path(files["labels"]), lambda fh: np.save(fh, passes[0][2]))
    for k, (_entry, channels, _labels, _logits) in enumerate(passes):
        suffix = "" if k == 0 else f"__p{k + 1}"
        for channel in CHANNELS:
            path = out_dir / f"{args.tag}__{run_id}__probs__{channel}{suffix}.npy"
            write_new_binary(path, lambda fh, a=channels[channel]: np.save(fh, a))
            if k == 0:
                files[channel] = str(path)

    summary = {
        "tag": args.tag,
        "run_id": run_id,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": ("auxiliary, non-gating cross-process reproducibility measurement; "
                    "the official acceptance path is save_raman_bn_checkpoint.py"),
        "not_comparable_with_phase2_record": True,
        "reason": ("the Phase 2 record was produced with cudnn.benchmark=True and "
                   "cudnn.deterministic unset, so its FP16 metrics are not bit-reproducible"),
        "checkpoint_kind": args.checkpoint,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "determinism": det,
        "batch_size": diag.BATCH_SIZE,
        "num_workers": diag.NUM_WORKERS,
        "threshold": diag.THRESHOLD,
        "n_val": int(val_idx.size),
        "label_names": label_names,
        "channels": list(CHANNELS),
        "channel_policy": ("each channel is compared only with itself across processes; a "
                           "cross-channel difference is a protocol difference, not a "
                           "reproducibility measurement"),
        "passes": [p[0] for p in passes],
        "in_process_repeat": in_process,
        "files": files,
    }
    summary_path = out_dir / f"{args.tag}__{run_id}__summary.json"
    write_new_text(summary_path, lambda fh: json.dump(summary, fh, ensure_ascii=False, indent=2))
    print(f"\n  SUMMARY: {summary_path}")
    return 0


# --------------------------------------------------------------------------- #
def load_latest(tag: str, out_root: Path):
    out_dir = out_root / tag
    cands = list(out_dir.glob(f"{tag}__r*__summary.json"))
    if not cands:
        return None, None

    def rid(p: Path) -> int:
        mid = p.name[len(tag) + 2:].split("__")[0]
        return int(mid[1:]) if mid.startswith("r") and mid[1:].isdigit() else 0

    path = max(cands, key=rid)
    return path, json.loads(path.read_text(encoding="utf-8"))


def cmd_compare(args) -> int:
    out_root = guard_out_root(Path(args.out_root) if args.out_root else TEMP_ROOT)
    pa, a = load_latest(args.tag_a, out_root)
    pb, b = load_latest(args.tag_b, out_root)
    if a is None or b is None:
        raise SystemExit("STOP: missing result for one of the tags")
    if a["checkpoint_sha256"] != b["checkpoint_sha256"]:
        raise SystemExit("STOP: the two runs used different checkpoint files")

    print("=" * 96)
    print(f"AUXILIARY CROSS-PROCESS COMPARISON  {args.tag_a}/{a['run_id']} vs "
          f"{args.tag_b}/{b['run_id']}")
    print(f"  checkpoint sha256 (equal): {a['checkpoint_sha256']}")
    print(f"  deterministic config: {a['determinism']['deterministic_algorithms']} "
          f"benchmark={a['determinism']['cudnn_benchmark']} "
          f"deterministic={a['determinism']['cudnn_deterministic']}")
    print("  Channels are compared WITHIN themselves only. No cross-channel number is")
    print("  produced, and no number here is compared with the Phase 2 record.")
    print("=" * 96)

    for channel in CHANNELS:
        xa, xb = np.load(a["files"][channel]), np.load(b["files"][channel])
        d = np.abs(xa - xb)
        print(f"\n  [{channel}]")
        print(f"    probs shape {xa.shape} dtype {xa.dtype}")
        print(f"    bitwise identical      : {bool(np.array_equal(xa, xb))}")
        print(f"    sha256 equal           : "
              f"{a['passes'][0]['channels'][channel]['probs_sha256'] == b['passes'][0]['channels'][channel]['probs_sha256']}")
        print(f"    max |dprob|            : {d.max():.3e}")
        print(f"    elements differing     : {int((d > 0).sum()):,} / {d.size:,}")
        ma = a["passes"][0]["channels"][channel]["metrics"]
        mb = b["passes"][0]["channels"][channel]["metrics"]
        print(f"    {'metric':18s} {'A':>12s} {'B':>12s} {'delta':>12s}")
        for key in METRIC_KEYS:
            print(f"    {key:18s} {ma[key]:12.6f} {mb[key]:12.6f} {mb[key] - ma[key]:+12.6f}")
        na = next(r for r in a["passes"][0]["channels"][channel]["per_label"]
                  if r["label"] == "nitro")
        nb = next(r for r in b["passes"][0]["channels"][channel]["per_label"]
                  if r["label"] == "nitro")
        print(f"    nitro AP               : {na['average_precision']:.6f} -> "
              f"{nb['average_precision']:.6f} "
              f"(delta {nb['average_precision'] - na['average_precision']:+.6f})")
        print(f"    nitro unique probs     : {na['unique_probabilities']} vs "
              f"{nb['unique_probabilities']}  (largest tie {na['largest_tie_group']} vs "
              f"{nb['largest_tie_group']})")
        print("    per-label AP delta     : " + ", ".join(
            f"{ra['label']}={rb['average_precision'] - ra['average_precision']:+.6f}"
            for ra, rb in zip(a["passes"][0]["channels"][channel]["per_label"],
                              b["passes"][0]["channels"][channel]["per_label"])))

    print("\n  INTERPRETATION")
    print("    bitwise identical in both channels -> with the algorithm pinned, the")
    print("      artifact reproduces itself exactly; the official-FP16 disagreement is then")
    print("      a property of the reference recording configuration, not of the artifact.")
    print("    identical only in the fp32 channel -> the residual difference is fp16")
    print("      quantisation of the sigmoid, i.e. the recorded metric is discontinuous.")
    print("    different in both channels -> some nondeterminism survives even with pinned")
    print("      algorithms; report it and do not widen any official tolerance.")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify_raman_bn_reproducibility.py",
        description="Auxiliary NON-GATING reproducibility measurement under a deterministic "
                    "CUDA configuration. Writes only to the system TEMP directory.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="One deterministic Validation pass (two channels).")
    r.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    r.add_argument("--out-root", default=None)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--allow-any-checkpoint", action="store_true")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="Compare two runs, within each channel only.")
    c.add_argument("--tag-a", required=True)
    c.add_argument("--tag-b", required=True)
    c.add_argument("--out-root", default=None)
    c.set_defaults(func=cmd_compare)
    return p


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    if getattr(args, "repeat", 1) < 1:
        raise SystemExit("STOP: --repeat must be >= 1")
    return args.func(args)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
