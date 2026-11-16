#!/usr/bin/env python
"""Validation-only reproducibility probe for EXP-RAMAN-001-FULL (BCE seed 42).

Purpose
-------
Diagnose the Validation-mAP discrepancy between
  * Phase 2 Full BN Diagnostic  (runs/RAMAN-BN-DIAGNOSTICS/EXP-RAMAN-001-FULL/...)
  * Phase 3 BN-checkpoint save  (which failed its 1e-3 Validation-mAP cross-check)

This script is READ-ONLY with respect to the project tree:
  * no checkpoint, log, config or diagnostic JSON is written or modified;
  * no BN recalibration, no backward pass, no optimizer, no training;
  * Test split is never loaded;
  * every output file is written to the operating-system TEMP directory only.

What it measures
----------------
Official Raman evaluation protocol (identical model / loader / metrics as
training/diagnose_raman_batchnorm_full.py):
    probs = torch.sigmoid(logits)                # sigmoid on the logits dtype
    metrics = diag.get_metrics(labels, probs)    # threshold 0.5
plus a diagnostic-only variant
    metrics_sigmoid_fp32 = diag.get_metrics(labels, torch.sigmoid(logits.float()))
which is reported next to the official numbers and never replaces them.

Sub-commands
------------
  run      Validation-only forward pass(es) for one checkpoint.
  compare  Read-only diff of two `run` results (probabilities, mAP, nitro AP, per-label AP).
  report   Read-only consolidated text block of every result in the TEMP root.

Examples (Windows PowerShell, from the SpectraDB root, spectra-db environment)
------------------------------------------------------------------------------
  python training\\diagnose_raman_validation_repro.py run --checkpoint orphan   --tag T1A
  python training\\diagnose_raman_validation_repro.py run --checkpoint orphan   --tag T1B
  python training\\diagnose_raman_validation_repro.py run --checkpoint orphan   --tag T1C --repeat 2
  python training\\diagnose_raman_validation_repro.py run --checkpoint original --tag T2
  python training\\diagnose_raman_validation_repro.py compare --tag-a T1A --tag-b T1B
  python training\\diagnose_raman_validation_repro.py report
  python training\\diagnose_raman_validation_repro.py run --checkpoint orphan --tag DRYRUN --dry-run

Every `run` gets its own run id (r1, r2, ...) and refuses to overwrite an
existing file, so repeating a tag never clobbers an earlier result.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import diagnose_raman_batchnorm_full as diag  # noqa: E402

TEMP_ROOT = Path(tempfile.gettempdir()) / "raman-validation-repro"

DIAGNOSTIC_JSON = (
    ROOT / "runs" / "RAMAN-BN-DIAGNOSTICS" / "EXP-RAMAN-001-FULL"
    / "batchnorm_full_diagnostic.json"
)

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
EXPECTED_LABELS = 14
METRIC_KEYS = ("micro_f1", "macro_f1", "micro_precision", "micro_recall",
               "mAP", "macro_AUROC")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(a: np.ndarray, dtype: str = "<f4") -> str:
    """Canonical hash: contiguous little-endian bytes, C order."""
    b = np.ascontiguousarray(np.asarray(a).astype(dtype, copy=False))
    return hashlib.sha256(b.tobytes(order="C")).hexdigest()


def guard_out_root(p: Path) -> Path:
    """Refuse any output location inside the project tree (especially runs/)."""
    rp = p.resolve()
    root = ROOT.resolve()
    if rp == root or root in rp.parents:
        raise SystemExit(
            f"REFUSED: output root {rp} is inside the project tree {root}. "
            "Diagnostics must go to the system TEMP directory."
        )
    if "runs" in {part.lower() for part in rp.parts}:
        raise SystemExit(f"REFUSED: output root {rp} contains a 'runs' component.")
    return rp


def next_run_id(out_dir: Path, tag: str) -> str:
    """Never reuse an id already present on disk (even from an incomplete run)."""
    used = 0
    if out_dir.is_dir():
        for f in out_dir.glob(f"{tag}__r*"):
            mid = f.name[len(tag) + 2:].split("__")[0]
            if mid.startswith("r") and mid[1:].isdigit():
                used = max(used, int(mid[1:]))
    return f"r{used + 1}"


def write_new_binary(path: Path, writer) -> None:
    """Exclusive create in binary mode: fails if the file already exists."""
    with path.open("xb") as fh:
        writer(fh)


def write_new_text(path: Path, writer) -> None:
    """Exclusive create in UTF-8 text mode: fails if the file already exists."""
    with path.open("x", encoding="utf-8") as fh:
        writer(fh)


def tie_stats(values: np.ndarray) -> dict:
    _, counts = np.unique(values, return_counts=True)
    tied = int(counts[counts > 1].sum())
    return {
        "unique_values": int(counts.size),
        "largest_tie_group": int(counts.max()),
        "samples_in_ties": tied,
        "tied_fraction": float(tied / values.size),
    }


# --------------------------------------------------------------------------- #
# forward pass
# --------------------------------------------------------------------------- #
def collect(model, loader, device):
    """Mirror of diag.collect, plus the diagnostic-only FP32-sigmoid variant."""
    logits_l, p_off_l, p_f32_l, y_l = [], [], [], []
    logits_dtype = sigmoid_dtype = None
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
            # Official path: sigmoid on the logits dtype (as in diag.collect).
            p_off = torch.sigmoid(logits)
            # Diagnostic-only path: force FP32 logits before sigmoid.
            p_f32 = torch.sigmoid(logits.float())
            if logits_dtype is None:
                logits_dtype = str(logits.dtype)
                sigmoid_dtype = str(p_off.dtype)
            logits_l.append(logits.detach().float().cpu().numpy())
            p_off_l.append(p_off.detach().float().cpu().numpy())
            p_f32_l.append(p_f32.detach().float().cpu().numpy())
            y_l.append(y.detach().cpu().numpy())

    logits = np.concatenate(logits_l, axis=0)
    labels = np.concatenate(y_l, axis=0).astype(np.int32)
    p_off = np.concatenate(p_off_l, axis=0)
    p_f32 = np.concatenate(p_f32_l, axis=0)
    for name, arr in (("logits", logits), ("probs_official", p_off),
                      ("probs_sigmoid_fp32", p_f32)):
        if not np.isfinite(arr).all():
            raise RuntimeError(f"non-finite {name}; investigate model numerics")
    return labels, logits, p_off, p_f32, logits_dtype, sigmoid_dtype


def per_label_rows(labels, probs, logits, names, variant):
    from sklearn.metrics import average_precision_score

    rows = []
    for j, name in enumerate(names):
        y = labels[:, j].astype(np.int32)
        s = probs[:, j]
        lg = logits[:, j]
        pos = int(y.sum())
        ap = None
        if 0 < pos < y.size:
            ap = float(average_precision_score(y, s))
        row = {
            "label": name,
            "positives": pos,
            "average_precision": ap,
            "metric_variant": variant,
            "logit_unique_values": int(np.unique(lg).size),
            "logit_largest_tie_group": tie_stats(lg)["largest_tie_group"],
            "prob_unique_values": int(np.unique(s).size),
        }
        row.update({f"prob_{k}": v for k, v in tie_stats(s).items()})
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def cmd_run(args) -> int:
    ckpt_path, expected_sha = CHECKPOINTS[args.checkpoint]
    out_root = guard_out_root(Path(args.out_root) if args.out_root else TEMP_ROOT)
    out_dir = out_root / args.tag

    print("=" * 78)
    print("Raman Validation-only reproducibility probe")
    print(f"  tag         : {args.tag}")
    print(f"  checkpoint  : {args.checkpoint}  ({ckpt_path})")
    print(f"  output root : {out_root}")
    print(f"  repeats     : {args.repeat}   dry-run: {args.dry_run}")
    print("=" * 78)

    if not ckpt_path.is_file():
        raise SystemExit(f"STOP: checkpoint not found: {ckpt_path}")
    sha_now = sha256_file(ckpt_path)
    print(f"  sha256      : {sha_now}")
    if args.allow_any_checkpoint:
        print("  WARNING: expected-hash gate disabled by --allow-any-checkpoint")
    elif sha_now != expected_sha:
        raise SystemExit(
            "STOP: checkpoint sha256 does not match the audited value.\n"
            f"  expected {expected_sha}\n  observed {sha_now}\n"
            "The file changed since the Phase 3 audit; do not proceed."
        )
    else:
        print("  PASS: checkpoint sha256 matches the audited value")

    # Validation split only. The Test split is never loaded by this script.
    with np.load(diag.SPLIT_PATH) as sp:
        val_idx = np.asarray(sp["val"], dtype=np.int64)
    if val_idx.size != EXPECTED_VAL:
        raise SystemExit(f"STOP: expected {EXPECTED_VAL} validation rows, got {val_idx.size}")
    if len(np.unique(val_idx)) != val_idx.size:
        raise SystemExit("STOP: duplicated indices inside the validation split")
    print(f"  PASS: validation rows = {val_idx.size} (unique); Test split not loaded")

    label_names = json.loads(diag.LABEL_NAMES_PATH.read_text(encoding="utf-8"))
    if len(label_names) != EXPECTED_LABELS:
        raise SystemExit("STOP: unexpected label schema")

    run_id = next_run_id(out_dir, args.tag)
    planned = [f"{args.tag}__{run_id}__summary.json",
               f"{args.tag}__{run_id}__probs.npy",
               f"{args.tag}__{run_id}__logits.npy",
               f"{args.tag}__{run_id}__labels.npy"]
    if args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\n  DRY RUN: nothing will be executed.")
        print(f"  next run id : {run_id}")
        for name in planned:
            target = out_dir / name
            print(f"  will create : {target}   exists={target.exists()}")
        print("  All targets are exclusive-create ('xb'); no file is overwritten.")
        return 0

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("STOP: --device cuda requested but CUDA is not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"  torch {torch.__version__} | cuda {torch.version.cuda} | "
              f"gpu {torch.cuda.get_device_name(0)}")
    else:
        print(f"  torch {torch.__version__} | CPU mode")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = diag.make_model(checkpoint, len(label_names), device)
    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    if len(bn_layers) != 12:
        raise SystemExit(f"STOP: expected 12 BatchNorm1d layers, found {len(bn_layers)}")
    if any(m.training for m in model.modules()):
        raise SystemExit("STOP: model contains a module in training mode")
    print(f"  PASS: 12 BN layers, entire model in eval mode (no BN recalibration)")

    loader = diag.make_loader(val_idx, device)
    print(f"  loader: batch_size={diag.BATCH_SIZE} num_workers={diag.NUM_WORKERS} "
          f"shuffle=False drop_last=False")

    passes = []
    for k in range(args.repeat):
        labels, logits, p_off, p_f32, ldt, sdt = collect(model, loader, device)
        m_off = diag.get_metrics(labels, p_off)
        m_f32 = diag.get_metrics(labels, p_f32)
        row = {
            "repeat_index": k + 1,
            "logits_dtype": ldt,
            "sigmoid_output_dtype": sdt,
            "metrics_official": m_off,
            "metrics_sigmoid_fp32": m_f32,
            "sigmoid_variants_identical_metrics": all(
                abs(m_off[key] - m_f32[key]) < 1e-12 for key in METRIC_KEYS
            ),
            "probs_sha256": array_sha256(p_off),
            "logits_sha256": array_sha256(logits),
            "labels_sha256": array_sha256(labels, "<i4"),
            "probs_mean": float(p_off.mean()),
            "probs_min": float(p_off.min()),
            "probs_max": float(p_off.max()),
            "per_label": per_label_rows(labels, p_off, logits, label_names, "official"),
            "per_label_sigmoid_fp32": per_label_rows(labels, p_f32, logits,
                                                     label_names, "sigmoid_fp32"),
        }
        nitro = next(r for r in row["per_label"] if r["label"] == "nitro")
        row["nitro"] = nitro
        passes.append((row, labels, logits, p_off, p_f32))

        print(f"\n  --- validation pass {k + 1}/{args.repeat} ---")
        print(f"  logits.dtype={ldt}  sigmoid(logits).dtype={sdt}  "
              f"autocast={'fp16' if device.type == 'cuda' else 'off'}")
        for key in METRIC_KEYS:
            print(f"    official {key:16s} {m_off[key]:.6f}   "
                  f"fp32-sigmoid {m_f32[key]:.6f}   "
                  f"diff {m_f32[key] - m_off[key]:+.3e}")
        print(f"  nitro: positives={nitro['positives']} AP={nitro['average_precision']:.6f} "
              f"unique_probs={nitro['prob_unique_values']} "
              f"largest_tie={nitro['prob_largest_tie_group']}")
        print(f"  probs sha256 = {row['probs_sha256']}")

    cross = None
    if args.cross_check:
        y2, lg2, p2 = diag.collect(model, loader, device)
        labels0, logits0, p0 = passes[0][1], passes[0][2], passes[0][3]
        cross = {
            "official_collect_labels_equal": bool(np.array_equal(y2, labels0)),
            "official_collect_probs_bitwise_equal": bool(np.array_equal(p2, p0)),
            "official_collect_logits_bitwise_equal": bool(np.array_equal(lg2, logits0)),
            "official_collect_max_abs_prob_diff":
                float(np.abs(p2 - p0).max()) if p2.shape == p0.shape else None,
        }
        print(f"\n  cross-check vs diag.collect: "
              f"probs bitwise equal={cross['official_collect_probs_bitwise_equal']} "
              f"max|d|={cross['official_collect_max_abs_prob_diff']}")

    in_process = None
    if args.repeat > 1:
        a, b = passes[0], passes[1]
        d_p = np.abs(a[3] - b[3])
        d_l = np.abs(a[2] - b[2])
        in_process = {
            "probs_bitwise_equal": bool(np.array_equal(a[3], b[3])),
            "logits_bitwise_equal": bool(np.array_equal(a[2], b[2])),
            "max_abs_prob_diff": float(d_p.max()),
            "mean_abs_prob_diff": float(d_p.mean()),
            "elements_differing": int((d_p > 0).sum()),
            "probs_sha256_equal": a[0]["probs_sha256"] == b[0]["probs_sha256"],
            "map_pass1": a[0]["metrics_official"]["mAP"],
            "map_pass2": b[0]["metrics_official"]["mAP"],
            "map_delta": b[0]["metrics_official"]["mAP"] - a[0]["metrics_official"]["mAP"],
        }
        print(f"\n  in-process repeat: probs bitwise equal="
              f"{in_process['probs_bitwise_equal']} max|d|={in_process['max_abs_prob_diff']:.3e} "
              f"dMAP={in_process['map_delta']:+.6f}")

    sha_after = sha256_file(ckpt_path)
    if sha_after != sha_now:
        raise SystemExit("CRITICAL: checkpoint changed during the run")
    print("\n  PASS: checkpoint unchanged after the run")

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for k, (row, labels_k, logits_k, p_off_k, p_f32_k) in enumerate(passes):
        suffix = "" if k == 0 else f"__p{k + 1}"
        f_probs = out_dir / f"{args.tag}__{run_id}__probs{suffix}.npy"
        f_logits = out_dir / f"{args.tag}__{run_id}__logits{suffix}.npy"
        write_new_binary(f_probs, lambda fh, a=p_off_k: np.save(fh, a))
        write_new_binary(f_logits, lambda fh, a=logits_k: np.save(fh, a))
        row["files"] = {"probs": str(f_probs), "logits": str(f_logits)}
        if k == 0:
            files["probs"] = str(f_probs)
            files["logits"] = str(f_logits)

    labels_path = out_dir / f"{args.tag}__{run_id}__labels.npy"
    write_new_binary(labels_path, lambda fh: np.save(fh, passes[0][1]))
    files["labels"] = str(labels_path)

    summary = {
        "tag": args.tag,
        "run_id": run_id,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "checkpoint_kind": args.checkpoint,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha_now,
        "checkpoint_sha256_after": sha_after,
        "diagnostic_record": str(DIAGNOSTIC_JSON),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "batch_size": diag.BATCH_SIZE,
        "num_workers": diag.NUM_WORKERS,
        "shuffle": False,
        "autocast": "fp16" if device.type == "cuda" else "off",
        "n_val": int(val_idx.size),
        "label_names": label_names,
        "threshold": diag.THRESHOLD,
        "passes": [p[0] for p in passes],
        "in_process_repeat": in_process,
        "official_collect_cross_check": cross,
        "files": files,
    }
    summary_path = out_dir / f"{args.tag}__{run_id}__summary.json"
    write_new_text(summary_path,
                   lambda fh: json.dump(summary, fh, ensure_ascii=False, indent=2))

    best = passes[0][0]["metrics_official"]
    print("\n" + "-" * 78)
    print(f"  RESULT {args.tag}/{run_id}  mAP={best['mAP']:.6f}  "
          f"microF1={best['micro_f1']:.6f}  macroF1={best['macro_f1']:.6f}  "
          f"AUROC={best['macro_AUROC']:.6f}")
    print(f"  SUMMARY: {summary_path}")
    print("-" * 78)
    return 0


# --------------------------------------------------------------------------- #
# read-only readers
# --------------------------------------------------------------------------- #
def load_latest(tag: str, out_root: Path) -> tuple:
    out_dir = out_root / tag
    cands = list(out_dir.glob(f"{tag}__r*__summary.json"))
    if not cands:
        return None, None

    def rid(p: Path) -> int:
        mid = p.name[len(tag) + 2:].split("__")[0]
        return int(mid[1:]) if mid.startswith("r") and mid[1:].isdigit() else 0

    path = max(cands, key=rid)   # highest run id, not lexicographic order
    return path, json.loads(path.read_text(encoding="utf-8"))


def fmt(x, nd=6):
    return "None" if x is None else f"{x:.{nd}f}"


def cmd_compare(args) -> int:
    out_root = guard_out_root(Path(args.out_root) if args.out_root else TEMP_ROOT)
    pa, a = load_latest(args.tag_a, out_root)
    pb, b = load_latest(args.tag_b, out_root)
    if a is None or b is None:
        raise SystemExit(f"STOP: missing result for {args.tag_a if a is None else args.tag_b} "
                         f"under {out_root}")
    print("=" * 100)
    print(f"COMPARE  {args.tag_a}/{a['run_id']}  vs  {args.tag_b}/{b['run_id']}   (read-only)")
    print(f"  {pa}\n  {pb}")
    print("=" * 100)
    print(f"  checkpoint sha256 equal : {a['checkpoint_sha256'] == b['checkpoint_sha256']}")
    print(f"  probs sha256 equal      : "
          f"{a['passes'][0]['probs_sha256'] == b['passes'][0]['probs_sha256']}")

    fa, fb = a["files"]["probs"], b["files"]["probs"]
    if Path(fa).is_file() and Path(fb).is_file():
        xa, xb = np.load(fa), np.load(fb)
        if xa.shape != xb.shape:
            print(f"  SKIP array diff: shape {xa.shape} != {xb.shape}")
        else:
            d = np.abs(xa - xb)
            print(f"  probs shape {xa.shape} dtype {xa.dtype}")
            print(f"  bitwise identical       : {bool(np.array_equal(xa, xb))}")
            print(f"  max |dprob|             : {d.max():.3e}")
            print(f"  mean |dprob|            : {d.mean():.3e}")
            print(f"  elements differing      : {int((d > 0).sum()):,} / {d.size:,}")
            print(f"  other_probs sha256 A/B  : {array_sha256(xa)[:16]} / {array_sha256(xb)[:16]}")
    else:
        print("  NOTE: probability arrays not found; only summary fields compared.")

    print("\n  METRICS (official protocol)")
    print(f"  {'metric':18s} {args.tag_a:>12s} {args.tag_b:>12s} {'delta':>12s}")
    for key in METRIC_KEYS:
        va = a["passes"][0]["metrics_official"][key]
        vb = b["passes"][0]["metrics_official"][key]
        print(f"  {key:18s} {va:12.6f} {vb:12.6f} {vb - va:+12.6f}")

    na = next(r for r in a["passes"][0]["per_label"] if r["label"] == "nitro")
    nb = next(r for r in b["passes"][0]["per_label"] if r["label"] == "nitro")
    print(f"\n  NITRO: positives={na['positives']}  AP {fmt(na['average_precision'])} -> "
          f"{fmt(nb['average_precision'])}  delta={nb['average_precision'] - na['average_precision']:+.6f}")
    print(f"         unique_probs={na['prob_unique_values']} vs {nb['prob_unique_values']}  "
          f"largest_tie={na['prob_largest_tie_group']} vs {nb['prob_largest_tie_group']}")

    print(f"\n  {'label':12s} {'pos':>6s} {'AP_A':>10s} {'AP_B':>10s} {'delta':>10s} "
          f"{'tieA':>6s} {'tieB':>6s}")
    for ra, rb in zip(a["passes"][0]["per_label"], b["passes"][0]["per_label"]):
        print(f"  {ra['label']:12s} {ra['positives']:6d} "
              f"{fmt(ra['average_precision'], 6):>10s} {fmt(rb['average_precision'], 6):>10s} "
              f"{rb['average_precision'] - ra['average_precision']:+10.6f} "
              f"{ra['prob_largest_tie_group']:6d} {rb['prob_largest_tie_group']:6d}")
    return 0


def cmd_report(args) -> int:
    out_root = guard_out_root(Path(args.out_root) if args.out_root else TEMP_ROOT)
    found = []
    if out_root.is_dir():
        for d in sorted(out_root.iterdir()):
            if d.is_dir():
                _, s = load_latest(d.name, out_root)
                if s is not None:
                    found.append(s)
    if not found:
        raise SystemExit(f"STOP: no results under {out_root}")
    found.sort(key=lambda s: (s["tag"], s["run_id"]))

    line = "=" * 118
    print(line)
    print("RAMAN VALIDATION REPRODUCIBILITY REPORT (read-only)")
    print(f"generated : {_dt.datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"temp root : {out_root}")
    print(line)

    rec = None
    if DIAGNOSTIC_JSON.is_file():
        rec = json.loads(DIAGNOSTIC_JSON.read_text(encoding="utf-8"))
        b = rec["before"]["validation"]
        aft = rec["after"]["validation"]
        print("\nPHASE 2 APPROVED RECORD (runs/RAMAN-BN-DIAGNOSTICS/EXP-RAMAN-001-FULL)")
        print(f"  before.validation  mAP={b['mAP']:.6f} mF1={b['micro_f1']:.6f} "
              f"MaF1={b['macro_f1']:.6f} P={b['micro_precision']:.6f} "
              f"R={b['micro_recall']:.6f} AUROC={b['macro_AUROC']:.6f}")
        print(f"  after.validation   mAP={aft['mAP']:.6f} mF1={aft['micro_f1']:.6f} "
              f"MaF1={aft['macro_f1']:.6f} P={aft['micro_precision']:.6f} "
              f"R={aft['micro_recall']:.6f} AUROC={aft['macro_AUROC']:.6f}")

    print("\nRUNS")
    hdr = (f"  {'tag':10s} {'run':4s} {'ckpt':9s} {'device':7s} {'logits':8s} "
           f"{'sigmoid':8s} {'mAP':>9s} {'mF1':>9s} {'MaF1':>9s} {'P':>9s} {'R':>9s} "
           f"{'AUROC':>9s} {'nitroAP':>9s}  probs_sha256[:16]")
    print(hdr)
    for s in found:
        p = s["passes"][0]
        m = p["metrics_official"]
        n = p["nitro"]
        print(f"  {s['tag']:8s} {s['run_id']:4s} {s['checkpoint_kind']:9s} "
              f"{s['device']:7s} {p['logits_dtype'].replace('torch.', ''):8s} "
              f"{p['sigmoid_output_dtype'].replace('torch.', ''):8s} "
              f"{m['mAP']:9.6f} {m['micro_f1']:9.6f} {m['macro_f1']:9.6f} "
              f"{m['micro_precision']:9.6f} {m['micro_recall']:9.6f} {m['macro_AUROC']:9.6f} "
              f"{n['average_precision']:9.6f}  {p['probs_sha256'][:16]}")

    # ---- T1A vs T1B -------------------------------------------------------
    a = next((s for s in found if s["tag"].upper() == "T1A"), None)
    b = next((s for s in found if s["tag"].upper() == "T1B"), None)
    c = next((s for s in found if s["tag"].upper() == "T1C"), None)
    t2 = next((s for s in found if s["tag"].upper() == "T2"), None)

    print("\nT1-A vs T1-B  (two independent Python processes, same orphan checkpoint)")
    if a and b:
        print(f"  sha256 probs equal      : "
              f"{a['passes'][0]['probs_sha256'] == b['passes'][0]['probs_sha256']}")
        print(f"  max|dprob| / mean|dprob| / n_differing : ", end="")
        try:
            xa, xb = np.load(a["files"]["probs"]), np.load(b["files"]["probs"])
            d = np.abs(xa - xb)
            print(f"{d.max():.3e} / {d.mean():.3e} / {int((d > 0).sum()):,}")
        except Exception as exc:  # pragma: no cover
            print(f"unavailable ({exc})")
        print(f"  {'metric':18s} {'T1A':>12s} {'T1B':>12s} {'delta':>12s}")
        for key in METRIC_KEYS:
            va = a["passes"][0]["metrics_official"][key]
            vb = b["passes"][0]["metrics_official"][key]
            print(f"  {key:18s} {va:12.6f} {vb:12.6f} {vb - va:+12.6f}")
        na = a["passes"][0]["nitro"]
        nb = b["passes"][0]["nitro"]
        print(f"  nitro AP               : {fmt(na['average_precision'])} -> "
              f"{fmt(nb['average_precision'])} "
              f"(delta {nb['average_precision'] - na['average_precision']:+.6f})")
        print(f"  per-label AP delta     : " + ", ".join(
            f"{ra['label']}={rb['average_precision'] - ra['average_precision']:+.5f}"
            for ra, rb in zip(a["passes"][0]["per_label"], b["passes"][0]["per_label"])))
    else:
        print("  (missing T1A and/or T1B)")

    print("\nT1-C  (same process, consecutive passes)")
    if c:
        ip = c.get("in_process_repeat") or {}
        print(f"  probs bitwise equal={ip.get('probs_bitwise_equal')} "
              f"max|d|={ip.get('max_abs_prob_diff')} "
              f"n_differing={ip.get('elements_differing')} "
              f"dMAP={ip.get('map_delta')}")
    else:
        print("  (missing T1C)")

    print("\nT2 vs PHASE 2 'before' (original checkpoint)")
    if t2 and rec:
        m = t2["passes"][0]["metrics_official"]
        bef = rec["before"]["validation"]
        print(f"  {'metric':18s} {'T2':>12s} {'before':>12s} {'diff':>12s}")
        for key in METRIC_KEYS:
            print(f"  {key:18s} {m[key]:12.6f} {bef[key]:12.6f} "
                  f"{m[key] - bef[key]:+12.6f}")
        print(f"  NOTE: T2 runs on {t2['device']}; the record is GPU fp16.")
    else:
        print("  (missing T2)")

    print("\nT1 (orphan) vs PHASE 2 'after' (recalibrated BN)")
    if a and rec:
        m = a["passes"][0]["metrics_official"]
        aft = rec["after"]["validation"]
        print(f"  {'metric':18s} {'T1A':>12s} {'after':>12s} {'diff':>12s}")
        for key in METRIC_KEYS:
            print(f"  {key:18s} {m[key]:12.6f} {aft[key]:12.6f} "
                  f"{m[key] - aft[key]:+12.6f}")
    else:
        print("  (missing T1A)")

    print("\nNITRO / TIE DIAGNOSTICS (pass 1, every run)")
    print(f"  {'tag':8s} {'pos':>4s} {'AP':>10s} {'logit_uniq':>11s} "
          f"{'logit_max_tie':>14s} {'prob_uniq':>10s} {'prob_max_tie':>13s} "
          f"{'prob_tied_frac':>15s} {'sigmoid':>9s}")
    for s in found:
        n = s["passes"][0]["nitro"]
        print(f"  {s['tag']:8s} {n['positives']:4d} {fmt(n['average_precision'], 6):>10s} "
              f"{n['logit_unique_values']:11d} {n['logit_largest_tie_group']:14d} "
              f"{n['prob_unique_values']:10d} {n['prob_largest_tie_group']:13d} "
              f"{n['prob_tied_fraction']:15.4f} {s['passes'][0]['sigmoid_output_dtype']:>9s}")

    print("\nEVIDENCE NOTES")
    for s in found:
        p = s["passes"][0]
        print(f"  {s['tag']}/{s['run_id']}: device={s['device']} autocast={s['autocast']} "
              f"logits.dtype={p['logits_dtype'].replace('torch.', '')} "
              f"sigmoid.dtype={p['sigmoid_output_dtype'].replace('torch.', '')} "
              f"official_vs_fp32sigmoid_identical={p['sigmoid_variants_identical_metrics']} "
              f"cudnn.benchmark={s['cudnn_benchmark']} "
              f"cudnn.deterministic={s['cudnn_deterministic']}")

    print("\nVERDICT HINTS (read the numbers, do not assume)")
    print("  * T1A/T1B probs bitwise identical and both close to the same mAP")
    print("      -> inference is deterministic; the Phase 2 vs Phase 3 gap is a")
    print("         cross-process numeric-path difference or a BN-buffer difference.")
    print("  * T1A/T1B probs differ at all")
    print("      -> genuine run-to-run nondeterminism; a 1e-3 Validation-mAP gate")
    print("         sits below the noise floor of this metric.")
    print("  * T2 reproduces 'before' within ~1e-5")
    print("      -> the Original path reproduces; the anomaly is specific to the")
    print("         recalibrated checkpoint or to the Phase 3 process.")
    print("  * nitro AP moves by ~0.02 while other labels move by <0.01")
    print("      -> the 14-label mean is dominated by an 11-positive label.")
    print(line)
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="diagnose_raman_validation_repro.py",
        description="Validation-only reproducibility probe for EXP-RAMAN-001-FULL. "
                    "Read-only; writes only to the system TEMP directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Validation-only forward pass(es) for one checkpoint.",
                       description="Validation-only forward pass(es). No training, no BN "
                                   "recalibration, no Test set, no writes outside TEMP.")
    r.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True,
                   help="original = runs/EXP-RAMAN-001-FULL/best_model.pt; "
                        "orphan = the recalibrated checkpoint left by the failed save")
    r.add_argument("--tag", required=True, help="result label, e.g. T1A / T1B / T1C / T2")
    r.add_argument("--repeat", type=int, default=1,
                   help="number of consecutive validation passes in this process (default 1)")
    r.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    r.add_argument("--out-root", default=None,
                   help="override the TEMP output root (must be outside the project tree)")
    r.add_argument("--dry-run", action="store_true",
                   help="validate paths and print the planned files, then exit")
    r.add_argument("--no-cross-check", dest="cross_check", action="store_false",
                   help="skip the extra diag.collect() agreement pass")
    r.add_argument("--allow-any-checkpoint", action="store_true",
                   help="disable the audited-sha256 gate (not recommended)")
    r.set_defaults(func=cmd_run, cross_check=True)

    c = sub.add_parser("compare", help="Read-only diff of two run results.")
    c.add_argument("--tag-a", required=True)
    c.add_argument("--tag-b", required=True)
    c.add_argument("--out-root", default=None)
    c.set_defaults(func=cmd_compare)

    rep = sub.add_parser("report", help="Read-only consolidated report of everything in TEMP.")
    rep.add_argument("--out-root", default=None)
    rep.set_defaults(func=cmd_report)
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
    try:
        sys.exit(main())
    except BrokenPipeError:          # e.g. piping into Select-Object / more
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.exit(0)
