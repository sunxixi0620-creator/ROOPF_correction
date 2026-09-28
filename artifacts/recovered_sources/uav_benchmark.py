"""Continuous UAV path-planning proxy benchmarks.

Each function is a 10-dimensional black-box objective. The decision vector
contains five two-dimensional intermediate waypoints. The UAV starts from
(-4.5, -4.5) and must reach (4.5, 4.5). Lower objective values are better.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


START = torch.tensor([-4.5, -4.5])
GOAL = torch.tensor([4.5, 4.5])


@dataclass(frozen=True)
class UAVScenario:
    fid: str
    obstacles: Tuple[Tuple[float, float, float, float], ...]
    nofly: Tuple[Tuple[float, float, float, float], ...]
    wind: Tuple[float, float]
    smooth_weight: float
    energy_weight: float
    corridor_weight: float


SCENARIOS: List[UAVScenario] = [
    UAVScenario(
        fid="uavf1",
        obstacles=((-1.2, -0.8, 1.1, 35.0), (1.5, 1.4, 1.0, 30.0), (0.2, 2.8, 0.8, 20.0)),
        nofly=((-2.8, 1.8, 0.9, 18.0),),
        wind=(0.35, -0.10),
        smooth_weight=1.2,
        energy_weight=0.10,
        corridor_weight=0.10,
    ),
    UAVScenario(
        fid="uavf2",
        obstacles=((-2.0, -1.0, 0.9, 28.0), (-0.2, 0.4, 1.2, 40.0), (2.0, 2.2, 1.0, 32.0)),
        nofly=((1.8, -1.2, 0.8, 20.0), (-2.2, 2.4, 0.7, 18.0)),
        wind=(-0.25, 0.30),
        smooth_weight=1.6,
        energy_weight=0.08,
        corridor_weight=0.14,
    ),
    UAVScenario(
        fid="uavf3",
        obstacles=((-2.8, -2.2, 0.7, 22.0), (-0.8, 1.2, 1.0, 36.0), (2.4, -0.3, 1.1, 34.0), (2.8, 2.8, 0.8, 24.0)),
        nofly=((0.5, 2.5, 0.9, 22.0),),
        wind=(0.10, 0.38),
        smooth_weight=1.0,
        energy_weight=0.14,
        corridor_weight=0.12,
    ),
    UAVScenario(
        fid="uavf4",
        obstacles=((-2.6, 0.2, 1.0, 35.0), (-0.2, -1.5, 0.9, 32.0), (1.6, 0.8, 1.2, 42.0), (3.0, -2.0, 0.8, 25.0)),
        nofly=((-1.0, 2.6, 0.8, 20.0), (2.8, 2.2, 0.9, 24.0)),
        wind=(-0.35, -0.20),
        smooth_weight=1.8,
        energy_weight=0.12,
        corridor_weight=0.08,
    ),
    UAVScenario(
        fid="uavf5",
        obstacles=((-3.0, -0.8, 0.7, 24.0), (-1.1, 1.0, 1.0, 34.0), (0.9, -0.4, 1.1, 38.0), (2.7, 1.7, 0.9, 30.0)),
        nofly=((0.0, 2.7, 0.8, 20.0), (-2.7, 2.2, 0.8, 18.0)),
        wind=(0.28, 0.22),
        smooth_weight=1.4,
        energy_weight=0.16,
        corridor_weight=0.16,
    ),
]


def _as_points(x: torch.Tensor) -> torch.Tensor:
    waypoints = x.view(*x.shape[:-1], 5, 2)
    start = START.to(device=x.device, dtype=x.dtype).view(*([1] * (x.dim() - 1)), 1, 2)
    goal = GOAL.to(device=x.device, dtype=x.dtype).view(*([1] * (x.dim() - 1)), 1, 2)
    start = start.expand(*x.shape[:-1], 1, 2)
    goal = goal.expand(*x.shape[:-1], 1, 2)
    return torch.cat([start, waypoints, goal], dim=-2)


def _segment_samples(points: torch.Tensor, samples_per_segment: int = 9) -> torch.Tensor:
    a = points[..., :-1, :]
    b = points[..., 1:, :]
    t = torch.linspace(0.0, 1.0, samples_per_segment, device=points.device, dtype=points.dtype)
    shape = *([1] * (points.dim() - 2)), 1, samples_per_segment, 1
    t = t.view(shape)
    return a.unsqueeze(-2) * (1.0 - t) + b.unsqueeze(-2) * t


def uav_objective(x: torch.Tensor, scenario: UAVScenario) -> torch.Tensor:
    x = torch.clamp(x, -5.0, 5.0)
    points = _as_points(x)
    seg = points[..., 1:, :] - points[..., :-1, :]
    seg_len = torch.linalg.norm(seg, dim=-1).clamp_min(1e-8)
    length = seg_len.sum(dim=-1)

    samples = _segment_samples(points)
    threat = torch.zeros_like(length)
    collision = torch.zeros_like(length)
    for cx, cy, radius, weight in scenario.obstacles:
        center = torch.tensor([cx, cy], device=x.device, dtype=x.dtype)
        dist = torch.linalg.norm(samples - center, dim=-1)
        threat = threat + weight * torch.exp(-2.0 * (dist / radius) ** 2).mean(dim=(-1, -2))
        collision = collision + 80.0 * torch.relu(radius * 0.65 - dist).pow(2).mean(dim=(-1, -2))

    nofly_penalty = torch.zeros_like(length)
    for cx, cy, radius, weight in scenario.nofly:
        center = torch.tensor([cx, cy], device=x.device, dtype=x.dtype)
        dist = torch.linalg.norm(samples - center, dim=-1)
        nofly_penalty = nofly_penalty + weight * torch.relu(radius - dist).pow(2).mean(dim=(-1, -2))

    unit = seg / seg_len.unsqueeze(-1)
    turns = unit[..., 1:, :] - unit[..., :-1, :]
    smooth = scenario.smooth_weight * torch.linalg.norm(turns, dim=-1).pow(2).sum(dim=-1)

    wind = torch.tensor(scenario.wind, device=x.device, dtype=x.dtype)
    headwind = torch.relu(-(unit * wind).sum(dim=-1))
    energy = scenario.energy_weight * (seg_len * (1.0 + 2.0 * headwind)).sum(dim=-1)

    direct = torch.linspace(-4.5, 4.5, points.size(-2), device=x.device, dtype=x.dtype)
    corridor = torch.stack([direct, direct], dim=-1)
    corridor_dev = torch.linalg.norm(points - corridor, dim=-1).mean(dim=-1)
    corridor_penalty = scenario.corridor_weight * corridor_dev.pow(2)

    boundary = torch.relu(torch.abs(points) - 4.8).pow(2).sum(dim=(-1, -2))
    return length + threat + collision + nofly_penalty + smooth + energy + corridor_penalty + 20.0 * boundary


def make_function(scenario: UAVScenario) -> Dict:
    def fun(x: torch.Tensor, **_kw) -> torch.Tensor:
        return uav_objective(x, scenario)

    return {
        "fid": scenario.fid,
        "fun": fun,
        "bias": None,
        "xlb": -5.0,
        "xub": 5.0,
        "_dim": 10,
        "_scenario": scenario,
    }


FUNCTIONS: Dict[str, Dict] = {sc.fid: make_function(sc) for sc in SCENARIOS}
