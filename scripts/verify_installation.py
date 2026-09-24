#!/usr/bin/env python3
"""Validate the ROOPF package before running expensive experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def tensors_are_on_cpu(value) -> bool:
    if torch.is_tensor(value):
        return value.device.type == "cpu"
    if isinstance(value, dict):
        return all(tensors_are_on_cpu(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(tensors_are_on_cpu(item) for item in value)
    return True


def main() -> None:
    config_path = ROOT / "configs" / "paper_protocol.json"
    require(config_path.is_file(), f"Missing protocol file: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    for relative, expected in config["artifacts"].items():
        path = ROOT / relative
        require(path.is_file(), f"Missing artifact: {relative}")
        actual = sha256(path)
        require(actual == expected, f"SHA-256 mismatch for {relative}: {actual}")
        print(f"[OK] {relative}: {actual}")

    offsets_path = ROOT / "data" / "bbob_offsets_d10.pkl"
    with offsets_path.open("rb") as stream:
        offsets = pickle.load(stream)
    require(set(offsets) == set(range(1, 25)), "BBOB offsets must contain IDs 1 through 24.")
    require(tensors_are_on_cpu(offsets), "BBOB offsets must be stored as portable CPU tensors.")
    print("[OK] BBOB offsets: 24 portable function transformations")

    bbob_rows = count_rows(ROOT / "reference_results" / "roopf_bbob_d10_nfe300.csv")
    cec_rows = count_rows(ROOT / "reference_results" / "roopf_cec_style_d10_nfe300.csv")
    require(bbob_rows == 24, f"Expected 24 BBOB reference rows, found {bbob_rows}.")
    require(cec_rows == 6, f"Expected 6 CEC-style reference rows, found {cec_rows}.")
    print("[OK] Reference tables: 24 BBOB rows and 6 CEC-style rows")

    from roopf import ROOPFOptimizer
    from roopf.benchmarks.bbobfunctions import FUNCTIONS as bbob_functions
    from roopf.benchmarks.cecfunctions import FUNCTIONS as cec_functions

    require(ROOPFOptimizer.__name__ == "ROOPFOptimizer", "ROOPF optimizer import failed.")
    require(len(bbob_functions) == 24, "Expected 24 BBOB function definitions.")
    require(len(cec_functions) == 6, "Expected six CEC-style function definitions.")
    print("[OK] Python imports and benchmark registries")
    print(f"[OK] Python {sys.version.split()[0]}")
    print(f"[OK] PyTorch {torch.__version__}; device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"[OK] NumPy {np.__version__}")
    print("ROOPF package verification passed.")


if __name__ == "__main__":
    main()

