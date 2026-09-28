#!/usr/bin/env python3
"""Train an offline residual router with validation-based early stopping."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


FEATURES = [
    "used",
    "remaining",
    "stagnation",
    "is_baseline",
    "surrogate_score",
    "surrogate_mu",
    "surrogate_sigma",
    "fitness_best_before",
    "fitness_worst_before",
    "state_best_z",
    "state_spread_z",
    "state_top_gap",
    "state_diversity",
    "state_elite_diversity",
    "state_centroid_shift",
    "state_remaining",
    "state_used",
    "state_stagnation",
    "state_archive_pressure",
    "state_dim_scaled",
]

OP_NAMES = [
    "baseline",
    "amortized_elite",
    "current_to_pbest_archive",
    "covariance_elite",
    "trust_region",
    "coordinate_pattern",
    "opposition_restart",
]


class RouterMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def parse_float(value: str) -> Tuple[float, float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0, 1.0
    if math.isnan(out) or math.isinf(out):
        return 0.0, 1.0
    return out, 0.0


def expand_paths(patterns: List[str]) -> List[str]:
    out = []
    for item in patterns:
        matches = sorted(glob.glob(item))
        out.extend(matches if matches else [item])
    return out


def load_rows(paths: List[str], label: str, include_baseline: bool, max_rows: int, seed: int):
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not include_baseline and int(row.get("is_baseline", 0)) == 1:
                    continue
                rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_rows > 0 and len(rows) > max_rows:
        rows = rows[:max_rows]

    y = np.asarray([int(r[label]) for r in rows], dtype=np.float32)
    fids = np.asarray([str(r.get("target", "")) + ":" + str(r.get("fid", "")) for r in rows])
    x_cols = []
    feature_names = []
    for name in FEATURES:
        vals, missing = [], []
        for r in rows:
            v, m = parse_float(r.get(name, ""))
            vals.append(v)
            missing.append(m)
        x_cols.append(vals)
        feature_names.append(name)
        if any(missing):
            x_cols.append(missing)
            feature_names.append(name + "_missing")
    for op in OP_NAMES:
        x_cols.append([1.0 if r.get("op_name", "") == op else 0.0 for r in rows])
        feature_names.append("op_" + op)
    x = np.asarray(x_cols, dtype=np.float32).T
    return x, y, fids, feature_names, rows


def split_by_fid(fids: np.ndarray, holdout_ratio: float, seed: int, holdout_fids_arg: str):
    if holdout_fids_arg.strip():
        holdout = {x.strip() if ":" in x else f"generated36:{x.strip()}" for x in holdout_fids_arg.split(",") if x.strip()}
    else:
        unique = sorted(set(fids.tolist()))
        if len(unique) < 2:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(fids))
            n_val = max(1, int(round(len(fids) * holdout_ratio)))
            val = np.zeros(len(fids), dtype=bool)
            val[order[:n_val]] = True
            return ~val, val, unique
        rng = random.Random(seed)
        rng.shuffle(unique)
        n_holdout = max(1, int(round(len(unique) * holdout_ratio)))
        holdout = set(unique[:n_holdout])
    val = np.asarray([fid in holdout for fid in fids])
    train = ~val
    return train, val, sorted(holdout)


def metrics(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    prob = torch.sigmoid(logits)
    pred = prob >= 0.5
    target = y >= 0.5
    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    tn = (~pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    acc = (tp + tn) / max(tp + fp + tn + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    out = {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": target.float().mean().item(),
        "pred_positive_rate": pred.float().mean().item(),
    }
    base_rate = out["positive_rate"]
    n = y.numel()
    order = torch.argsort(prob, descending=True)
    for frac in [0.01, 0.05, 0.10, 0.20]:
        k = max(1, int(round(n * frac)))
        top = target[order[:k]].float().mean().item()
        out[f"p_at_{int(frac * 100)}"] = top
        out[f"lift_at_{int(frac * 100)}"] = top / max(base_rate, 1e-12)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--label", choices=["admitted", "improved_best", "beat_baseline"], default="improved_best")
    parser.add_argument("--out", default="research_v41_surrogate_portfolio/results_v58_router/router_generated36_earlystop.pt")
    parser.add_argument("--metrics-out", default="")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--holdout-ratio", type=float, default=0.20)
    parser.add_argument("--holdout-fids", default="")
    parser.add_argument("--max-rows", type=int, default=1200000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--include-baseline", action="store_true")
    args = parser.parse_args()

    paths = expand_paths(args.csv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    x, y, fids, feature_names, _rows = load_rows(
        paths,
        label=args.label,
        include_baseline=args.include_baseline,
        max_rows=args.max_rows,
        seed=args.seed,
    )
    if len(y) == 0:
        raise RuntimeError("No training rows loaded")
    train_mask, val_mask, holdout_fids = split_by_fid(fids, args.holdout_ratio, args.seed, args.holdout_fids)
    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    if len(y_val) == 0:
        raise RuntimeError("Validation split is empty")

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RouterMLP(x_train.shape[1], args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    xv = torch.tensor(x_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_score = float("-inf")
    best_epoch = 0
    best_payload = None
    history = []
    wait = 0
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(xt.size(0))
        losses = []
        model.train()
        for start in range(0, xt.size(0), args.batch_size):
            idx = perm[start : start + args.batch_size]
            xb = xt[idx].to(device)
            yb = yt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        model.eval()
        with torch.no_grad():
            val_logits = model(xv)
            val_loss = float(loss_fn(val_logits, yv).detach().cpu().item())
            m = metrics(val_logits.detach().cpu(), torch.tensor(y_val))
        train_loss = float(np.mean(losses))
        score = m["p_at_5"] + 0.25 * m["p_at_10"] + 0.05 * m["f1"] - 0.01 * val_loss
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "score": score,
            **m,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"score={score:.5f} val_f1={m['f1']:.4f} p@1={m['p_at_1']:.4f} "
            f"p@5={m['p_at_5']:.4f} p@10={m['p_at_10']:.4f} "
            f"lift@5={m['lift_at_5']:.2f} pred_pos={m['pred_positive_rate']:.4f}",
            flush=True,
        )
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            wait = 0
            best_payload = {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "feature_names": feature_names,
                "mean": mean.squeeze(0).astype(float).tolist(),
                "std": std.squeeze(0).astype(float).tolist(),
                "label": args.label,
                "include_baseline": bool(args.include_baseline),
                "holdout_fids": holdout_fids,
                "op_names": OP_NAMES,
                "metrics": row,
                "best_epoch": best_epoch,
                "best_score": float(best_score),
                "training_source": "generated36_only",
                "csv": paths,
            }
        else:
            wait += 1
            if wait >= args.patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_score={best_score:.5f}", flush=True)
                break

    if best_payload is None:
        raise RuntimeError("No checkpoint selected")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_payload, out)
    metrics_out = Path(args.metrics_out) if args.metrics_out else out.with_suffix(".metrics.json")
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_score": best_score,
                "holdout_fids": holdout_fids,
                "n_rows": int(len(y)),
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
                "positive_rate_train": float(y_train.mean()),
                "positive_rate_val": float(y_val.mean()),
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"saved: {out}")
    print(f"saved metrics: {metrics_out}")
    print(f"best_epoch={best_epoch} best_score={best_score:.5f} holdout={','.join(holdout_fids)}")


if __name__ == "__main__":
    main()
