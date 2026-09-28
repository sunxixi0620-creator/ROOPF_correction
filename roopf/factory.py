"""Explicit entry for the frozen unlocked method and its matched ablations."""
from pathlib import Path
import torch
from .supplement import SupplementOptimizer

ROOT = Path(__file__).resolve().parents[1]


def build_final(variant='full', warmup=.70, diagnostic_stride=0, *, device='cpu',
                dim=10, budget=300, population=100, model_seed=20260630,
                anchor=None, residual=None):
    device = torch.device(device)
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(model_seed)
        opt = SupplementOptimizer(dim=dim, hidden_dim=200, popSize=population,
            max_nfe=budget, k_nums=2, pool_per_op=6, surrogate_members=5,
            ablation='roopf', baseline_ckpt=str(anchor or ROOT/'checkpoints/anchor_policy_d10.pt'),
            router_ckpt=str(residual or ROOT/'checkpoints/residual_selector_generated36_d10.pt'),
            router_weight=.008).to(device).eval().configure(variant, warmup, diagnostic_stride)
    original_predict = opt.surrogate.predict
    def measured_predict(*args, **kwargs):
        with opt.measured('proxy_seconds'):
            return original_predict(*args, **kwargs)
    opt.surrogate.predict = measured_predict
    return opt
