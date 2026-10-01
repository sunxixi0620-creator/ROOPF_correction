"""Version 2: fitted references with cold restart after a failed warm fit.

Independent implementation of the algorithm described in the official TuRBO
tutorial. Sequential analytic LogEI replaces the tutorial's batch MC EI.
No objective or source-label access is available to these proposal functions.
"""
from dataclasses import dataclass, asdict
import time
import warnings

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from roopf.experiment_io import seed_for


@dataclass
class TrustRegion:
    length: float = .8
    successes: int = 0
    failures: int = 0
    best: float = 0.
    start: int = 0
    restarts: int = 0

    def update(self, value):
        if value < self.best - 1e-3 * abs(self.best):
            self.successes += 1
            self.failures = 0
        else:
            self.successes = 0
            self.failures += 1
        # Tutorial success tolerance 10, dimension / q = 20 failures.
        if self.successes == 10:
            self.length = min(2 * self.length, 1.6)
            self.successes = 0
        if self.failures == 20:
            self.length /= 2
            self.failures = 0
        self.best = min(self.best, value)


def model_for(x, y, method, device='cpu'):
    tx = (x.to(device=device, dtype=torch.double) + 5) / 10
    ty = -y.to(device=device, dtype=torch.double).unsqueeze(-1)
    kw = {}
    if method == 'TuRBO_LogEI':
        kw = dict(likelihood=GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3)),
                  covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=20,
                      lengthscale_constraint=Interval(.005, 4.))))
    # Default GP: dimension-scaled RBF prior; observed-data Standardize for both.
    return SingleTaskGP(tx, ty, **kw).to(device=device, dtype=torch.double)


def propose(x, y, method, identity, state=None, previous=None, device='cpu'):
    start = time.perf_counter()
    seed = seed_for('strong_online_v1', identity, method, len(x)) % (2**31 - 1)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    offset = state.start if state else 0
    tx, ty = x[offset:], y[offset:]
    model = model_for(tx, ty, method, device)
    if previous:
        # Warm-start only hyperparameters, never the outcome normalization buffers.
        named = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in previous.items():
                named[name].copy_(value.to(device))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        recovery = False
        try:
            fit_gpytorch_mll(mll, optimizer_kwargs={'options': {'maxiter': 100, 'ftol': 1e-9}})
        except ModelFittingError:
            # Use exactly the same data and model with fresh default parameters.
            # No objective query, outcome-based tuning or kernel change.
            recovery = True
            model = model_for(tx, ty, method, device)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll, optimizer_kwargs={'options': {'maxiter': 100, 'ftol': 1e-9}})
        fit_seconds = time.perf_counter() - start
        bounds = torch.stack((torch.zeros(20), torch.ones(20))).to(device=device, dtype=torch.double)
        if state:
            center = model.train_inputs[0][ty.argmin()]
            weights = model.covar_module.base_kernel.lengthscale.detach().flatten()
            weights = weights / weights.log().mean().exp()
            bounds = torch.stack(((center - weights * state.length / 2).clamp(0, 1),
                                  (center + weights * state.length / 2).clamp(0, 1)))
        acq = LogExpectedImprovement(model, best_f=(-ty).max().to(device=device, dtype=torch.double))
        candidates, _ = optimize_acqf(acq, bounds, q=1, num_restarts=10,
            raw_samples=512, options={'maxiter': 200, 'batch_limit': 5}, return_best_only=False)
        # Preserve all optimized starts and a Sobol fallback. Score float32 points
        # actually sent to the existing objective; exclude previously paid points.
        sobol = torch.quasirandom.SobolEngine(20, scramble=True, seed=seed).draw(512).to(bounds)
        raw = bounds[0] + (bounds[1] - bounds[0]) * sobol
        points = (torch.cat((candidates.squeeze(1), raw)) * 10 - 5).float().cpu()
        distance = torch.cdist(points.double(), x.double()).amin(-1)
        valid = distance > 1e-6
        assert valid.any(), 'No new point in trust region'
        unit = (points.to(device=device, dtype=torch.double) + 5) / 10
        with torch.no_grad():
            scores = acq(unit.unsqueeze(1)).cpu()
        scores[~valid] = -float('inf')
        k = int(scores.argmax())
        assert torch.isfinite(scores[k])
    params = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    info = dict(cold_fit_recovery=recovery, seed=seed, offset=offset, state=asdict(state) if state else None,
                bounds=bounds.detach().cpu(), selected=k, logei=float(scores[k]),
                fit_seconds=fit_seconds, total_seconds=time.perf_counter()-start,
                warnings=[str(w.message) for w in caught], parameters=params,
                point=points[k].clone())
    return points[k:k+1], info, params


def replay_score(x, y, method, info, device='cpu'):
    model = model_for(x[info['offset']:], y[info['offset']:], method, device)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(info['parameters'][name].to(p))
    model.eval()
    acq = LogExpectedImprovement(model, best_f=(-y[info['offset']:]).max().to(device=device, dtype=torch.double))
    unit = (info['point'].to(device=device, dtype=torch.double) + 5) / 10
    with torch.no_grad():
        return float(acq(unit.reshape(1, 1, 20)).cpu())
