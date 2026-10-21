#!/usr/bin/env python3
"""Compare reproduced ROOPF results with an archived reference table."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict


def read_rows(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = {row["fid"]: row for row in csv.DictReader(stream)}
    if not rows:
        raise ValueError(f"No result rows found in {path}")
    return rows


def trajectory_count(row: dict) -> int:
    if row.get("total_trajectories"):
        return int(row["total_trajectories"])
    batch = row.get("batch_size", row.get("batchsize"))
    repeats = row.get("repeat_batches", row.get("runs"))
    if batch and repeats:
        return int(batch) * int(repeats)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.05,
        help="Relative difference used to mark a reproduced mean as close (default: 0.05).",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=1.96,
        help="Standard-error threshold for statistical consistency (default: 1.96).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero for missing rows or statistically clear degradation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = read_rows(args.reference)
    candidate = read_rows(args.candidate)
    missing = sorted(set(reference) - set(candidate))
    shared = sorted(set(reference) & set(candidate), key=lambda value: (len(value), value))
    if not shared:
        raise RuntimeError("The two tables do not share any function IDs.")

    degraded = []
    counts = {"close": 0, "consistent": 0, "improved": 0, "review": 0}
    print("fid,reference_mean,candidate_mean,relative_difference,z_score,status")
    for fid in shared:
        ref_row = reference[fid]
        cur_row = candidate[fid]
        ref = float(ref_row["mean"])
        cur = float(cur_row["mean"])
        scale = max(abs(ref), 1.0e-12)
        relative = abs(cur - ref) / scale
        n_ref = trajectory_count(ref_row)
        n_cur = trajectory_count(cur_row)
        z_score = math.nan
        if n_ref > 1 and n_cur > 1 and ref_row.get("std") and cur_row.get("std"):
            standard_error = math.sqrt(
                float(ref_row["std"]) ** 2 / n_ref + float(cur_row["std"]) ** 2 / n_cur
            )
            if standard_error > 0:
                z_score = abs(cur - ref) / standard_error

        if math.isfinite(relative) and relative <= args.relative_tolerance:
            status = "close"
        elif math.isfinite(z_score) and z_score <= args.z_threshold:
            status = "consistent"
        elif cur < ref:
            status = "improved"
        else:
            status = "review"
            degraded.append(fid)
        counts[status] += 1
        z_text = f"{z_score:.3f}" if math.isfinite(z_score) else "n/a"
        print(f"{fid},{ref:.9e},{cur:.9e},{relative:.6f},{z_text},{status}")

    print(
        f"Shared functions: {len(shared)}; close: {counts['close']}; "
        f"consistent: {counts['consistent']}; improved: {counts['improved']}; "
        f"review: {counts['review']}; missing: {len(missing)}"
    )
    if missing:
        print("Missing candidate rows: " + ", ".join(missing))
    if degraded:
        print("Statistically clear degradation to review: " + ", ".join(degraded))
    if args.strict and (degraded or missing):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
