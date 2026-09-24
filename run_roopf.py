#!/usr/bin/env python3
"""Reproduce the ROOPF BBOB and CEC-style experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from roopf import ROOPFOptimizer
from roopf.benchmarks.bbobfunctions import FUNCTIONS as BBOB_FUNCTIONS
from roopf.benchmarks.cecfunctions import FUNCTIONS as CEC_FUNCTIONS
from roopf.benchmarks.utils import getFitness, setOffset


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEED = 20260630

PAPER_CONFIG = {
    "dimension": 10,
    "population_size": 100,
    "evaluation_budget": 300,
    "anchor_candidates": 2,
    "candidates_per_operator": 6,
    "surrogate_members": 5,
    "hidden_dimension": 200,
    "repeat_batches": 10,
    "batch_size": 10,
    "residual_weight": 0.008,
}


class BBOBProblem:
    def __init__(self, function: Dict, dimension: int):
        self.fun = function
        self.dim = dimension

    def repaire(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.fun["xlb"], self.fun["xub"])

    def calfitness(self, x: torch.Tensor) -> torch.Tensor:
        x = self.repaire(x)
        batch, population, dimension = x.shape
        values = getFitness(x.reshape(-1, dimension), self.fun)
        return values.view(batch, population)

    def genRandomPop(self, shape: Sequence[int]) -> torch.Tensor:
        lower = self.fun["xlb"]
        upper = self.fun["xub"]
        return torch.rand(shape, device=DEVICE) * (upper - lower) + lower

    def setfun(self, function: Dict) -> None:
        self.fun = function

    def getfunname(self):
        return self.fun["fid"]


class CECStyleProblem(BBOBProblem):
    def calfitness(self, x: torch.Tensor) -> torch.Tensor:
        x = self.repaire(x)
        values = getFitness(x, self.fun)
        if values.dim() == 3 and values.size(-1) == 1:
            values = values.squeeze(-1)
        return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_tensors(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(item, device) for item in value)
    return value


def load_bbob_offsets(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing fixed BBOB offsets: {path}")
    with path.open("rb") as stream:
        offsets = pickle.load(stream)
    if set(offsets) != set(range(1, 25)):
        raise ValueError("The BBOB offset file must contain function IDs 1 through 24.")
    return move_tensors(offsets, DEVICE)


def parse_function_ids(text: str, valid_ids: Iterable[int]) -> List[int]:
    valid = list(valid_ids)
    if not text or text.strip().lower() == "all":
        return valid
    selected = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if token:
            selected.append(int(token))
    invalid = sorted(set(selected) - set(valid))
    if invalid:
        raise ValueError(f"Invalid function IDs: {invalid}; valid IDs are {valid}.")
    return selected


def build_optimizer(args: argparse.Namespace) -> ROOPFOptimizer:
    anchor_checkpoint = ROOT / "checkpoints" / "anchor_policy_d10.pt"
    residual_checkpoint = ROOT / "checkpoints" / "residual_selector_generated36_d10.pt"
    for checkpoint in (anchor_checkpoint, residual_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing ROOPF checkpoint: {checkpoint}")

    optimizer = ROOPFOptimizer(
        dim=args.dimension,
        hidden_dim=PAPER_CONFIG["hidden_dimension"],
        popSize=args.population_size,
        max_nfe=args.budget,
        k_nums=PAPER_CONFIG["anchor_candidates"],
        pool_per_op=PAPER_CONFIG["candidates_per_operator"],
        surrogate_members=PAPER_CONFIG["surrogate_members"],
        ablation="roopf",
        baseline_ckpt=str(anchor_checkpoint),
        router_ckpt=str(residual_checkpoint),
        router_weight=PAPER_CONFIG["residual_weight"],
    ).to(DEVICE)
    optimizer.eval()
    return optimizer


def evaluate_problem(
    optimizer: ROOPFOptimizer,
    problem,
    dimension: int,
    population_size: int,
    batch_size: int,
    repeat_batches: int,
) -> Dict:
    final_values = []
    trails = []
    evaluation_counts = []
    with torch.no_grad():
        for _ in range(repeat_batches):
            population = problem.genRandomPop((batch_size, population_size, dimension))
            _, trail, evaluations, _ = optimizer(population, problem)
            final_values.append(trail[:, -1].detach().cpu())
            trails.append(trail.detach().cpu())
            evaluation_counts.append(int(evaluations))

    final = torch.cat(final_values, dim=0)
    return {
        "final_values": final.numpy(),
        "trails": torch.cat(trails, dim=0).numpy(),
        "evalnums": max(evaluation_counts),
        "mean": float(final.mean().item()),
        "std": float(final.std(unbiased=False).item()),
    }


def result_path(args: argparse.Namespace, benchmark: str, function_ids: List[int]) -> Path:
    suffix = ""
    all_ids = list(range(1, 25 if benchmark == "bbob" else 7))
    if function_ids != all_ids:
        suffix = "_f" + "-".join(str(fid) for fid in function_ids)
    return args.output_dir / f"roopf_{benchmark}_d{args.dimension}_nfe{args.budget}{suffix}.csv"


def make_row(args: argparse.Namespace, benchmark: str, fid: str, result: Dict) -> Dict:
    return {
        "target": benchmark,
        "fid": fid,
        "dim": args.dimension,
        "population_size": args.population_size,
        "evaluation_budget": args.budget,
        "batch_size": args.batch_size,
        "repeat_batches": args.repeat_batches,
        "total_trajectories": args.batch_size * args.repeat_batches,
        "mean": result["mean"],
        "std": result["std"],
        "evalnums": result["evalnums"],
    }


def run_bbob(args: argparse.Namespace, optimizer: ROOPFOptimizer) -> Path:
    function_ids = parse_function_ids(args.functions, range(1, 25))
    offsets = load_bbob_offsets(ROOT / "data" / "bbob_offsets_d10.pkl")
    problem = BBOBProblem(BBOB_FUNCTIONS[1], args.dimension)
    rows = []
    trails = {}

    for fid in function_ids:
        function = BBOB_FUNCTIONS[fid]
        function["xlb"] = -5
        function["xub"] = 5
        setOffset(function, offsets[fid])
        problem.setfun(function)
        result = evaluate_problem(
            optimizer,
            problem,
            args.dimension,
            args.population_size,
            args.batch_size,
            args.repeat_batches,
        )
        rows.append(make_row(args, "bbob", str(fid), result))
        trails[str(fid)] = result
        print(f"ROOPF BBOB-f{fid}: mean={result['mean']:.6e}, std={result['std']:.6e}")

    output = result_path(args, "bbob", function_ids)
    write_results(output, rows)
    if args.save_trails:
        write_trails(output.with_suffix(".trails.pkl"), trails)
    return output


def run_cec_style(args: argparse.Namespace, optimizer: ROOPFOptimizer) -> Path:
    function_ids = parse_function_ids(args.functions, range(1, 7))
    rows = []
    trails = {}

    for fid in function_ids:
        function = CEC_FUNCTIONS[f"cecf{fid}"]
        function["xlb"] = function.get("xlb", -100)
        function["xub"] = function.get("xub", 100)
        problem = CECStyleProblem(function, args.dimension)
        result = evaluate_problem(
            optimizer,
            problem,
            args.dimension,
            args.population_size,
            args.batch_size,
            args.repeat_batches,
        )
        rows.append(make_row(args, "cec_style", f"cecf{fid}", result))
        trails[f"cecf{fid}"] = result
        print(f"ROOPF CEC-style F{fid}: mean={result['mean']:.6e}, std={result['std']:.6e}")

    output = result_path(args, "cec_style", function_ids)
    write_results(output, rows)
    if args.save_trails:
        write_trails(output.with_suffix(".trails.pkl"), trails)
    return output


def write_results(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results: {path}")


def write_trails(path: Path, trails: Dict) -> None:
    with path.open("wb") as stream:
        pickle.dump(trails, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved trajectories: {path}")


def write_run_metadata(args: argparse.Namespace, outputs: List[Path]) -> None:
    checkpoint_paths = {
        "anchor": ROOT / "checkpoints" / "anchor_policy_d10.pt",
        "residual_selector": ROOT / "checkpoints" / "residual_selector_generated36_d10.pt",
        "bbob_offsets": ROOT / "data" / "bbob_offsets_d10.pkl",
    }
    metadata = {
        "method": "ROOPF",
        "seed": args.seed,
        "device": str(DEVICE),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "paper_configuration": PAPER_CONFIG,
        "sha256": {name: sha256(path) for name, path in checkpoint_paths.items()},
        "outputs": [str(path) for path in outputs],
    }
    path = args.output_dir / "run_metadata.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Saved metadata: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ROOPF under the paper's 10-D BBOB and CEC-style protocol."
    )
    parser.add_argument(
        "--benchmark",
        choices=["bbob", "cec", "all"],
        default="all",
        help="Benchmark group to evaluate.",
    )
    parser.add_argument(
        "--functions",
        default="all",
        help="Comma-separated function IDs or 'all'. Use one benchmark at a time for subsets.",
    )
    parser.add_argument("--dimension", type=int, default=PAPER_CONFIG["dimension"])
    parser.add_argument("--population-size", type=int, default=PAPER_CONFIG["population_size"])
    parser.add_argument("--budget", type=int, default=PAPER_CONFIG["evaluation_budget"])
    parser.add_argument("--batch-size", type=int, default=PAPER_CONFIG["batch_size"])
    parser.add_argument("--repeat-batches", type=int, default=PAPER_CONFIG["repeat_batches"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--save-trails", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run function 1 with one trajectory and 120 evaluations as a smoke test.",
    )
    args = parser.parse_args()

    if args.dimension != 10:
        parser.error("The released checkpoints reproduce the paper only at dimension 10.")
    if args.budget <= args.population_size:
        parser.error("The evaluation budget must exceed the initial population size.")
    if args.benchmark == "all" and args.functions.lower() != "all":
        parser.error("Use --functions only with --benchmark bbob or --benchmark cec.")
    if args.quick:
        if args.benchmark == "all":
            args.benchmark = "bbob"
        args.functions = "1"
        args.batch_size = 1
        args.repeat_batches = 1
        args.budget = 120
    return args


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    if args.benchmark in {"bbob", "all"}:
        set_seed(args.seed)
        optimizer = build_optimizer(args)
        outputs.append(run_bbob(args, optimizer))
    if args.benchmark in {"cec", "all"}:
        # Each benchmark group starts from the same released initialization.
        set_seed(args.seed)
        optimizer = build_optimizer(args)
        outputs.append(run_cec_style(args, optimizer))
    write_run_metadata(args, outputs)


if __name__ == "__main__":
    main()
