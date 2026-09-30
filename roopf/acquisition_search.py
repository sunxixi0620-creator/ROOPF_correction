"""Bounded multistart LogEI search with a frozen prior and fixed exact GP.

Only acquisition optimization changes. Returned float32 candidates are rescored
by the original predictor; all original candidates remain eligible.
"""
from copy import deepcopy
import math
import numpy as np
import torch
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import ndtr, log_ndtr

RESTARTS = 8
MAXITER = 40
MAXFUN = 65
LOG2PI = math.log(2 * math.pi)


def log_ei_parts(mu, sd, best):
    """Log EI and positive derivative multipliers -d/dmu and d/dsd.

    For u < -20 use the Gaussian-tail asymptotic expansion of phi(u)+u*Phi(u).
    Eleven terms have negligible truncation error at this boundary in float64.
    """
    sd = np.maximum(np.asarray(sd, dtype=float), 1e-14)
    u = (best - np.asarray(mu, dtype=float)) / sd
    logphi = -.5*u*u - .5*LOG2PI
    logh = np.empty_like(u)
    tail = u < -20
    t = u[tail]
    inv = 1/(t*t)
    term = np.ones_like(t)
    series = term.copy()
    for k in range(1, 11):
        term *= -(2*k+1)*inv
        series += term
    logh[tail] = logphi[tail] - 2*np.log(-t) + np.log(series)
    v = u[~tail]
    h = np.exp(logphi[~tail]) + v*ndtr(v)
    assert (h > 0).all()
    logh[~tail] = np.log(h)
    value = np.log(sd) + logh
    return value, np.exp(log_ndtr(u)-value), np.exp(logphi-value)


class SmoothPrior:
    """Double-precision extension of the frozen original mean function."""
    def __init__(self, p, device='cpu'):
        self.kind = p.kind
        self.device = device
        self.cx = p.cx.double().to(device)
        # Preserve the already fixed center/scale and float32 context transform.
        self.cy = ((p.cy-p.center)/p.scale).double().to(device)
        self.model = deepcopy(p.model).double().to(device) if p.model is not None else None
        cr = (p.cx/5).square().mean(-1)
        self.radius_center = float(cr.mean())
        self.radius_scale = float(cr.std(unbiased=False).clamp_min(.01))

    def value_gradient(self, q):
        q = np.asarray(q, dtype=float)
        if self.kind == 'zero':
            return np.zeros(len(q)), np.zeros_like(q)
        if self.kind == 'analytic':
            return ((q*q/25).mean(1)-self.radius_center)/self.radius_scale, 2*q/(25*20*self.radius_scale)
        with torch.enable_grad():
            z = torch.tensor(q, dtype=torch.double, device=self.device, requires_grad=True)
            m = self.model(self.cx[None], self.cy[None], z[None])[0]
            grad, = torch.autograd.grad(m.sum(), z)
        return m.detach().cpu().numpy(), grad.detach().cpu().numpy()


class SmoothAcquisition:
    def __init__(self, gp, prior, active, best):
        self.gp, self.prior, self.active, self.best = gp, prior, active, best
        self.alpha = cho_solve((gp.L, True), gp.y, check_finite=False)

    def posterior(self, q):
        q = np.asarray(q, dtype=float).reshape(-1,20)
        diff = q[None,:,:]-self.gp.x[:,None,:]
        r = np.sqrt(np.sum(diff*diff, axis=2)/125)
        a = np.sqrt(5)*r
        e = np.exp(-a)
        k = (1+a+a*a/3)*e
        dk = -(5/(3*125))*(1+a)[:,:,None]*e[:,:,None]*diff
        beta = cho_solve((self.gp.L, True), k, check_finite=False)
        mu = k.T@self.alpha
        var = 1-np.sum(k*beta,axis=0)
        assert var.min() > -1e-8
        sd = np.sqrt(np.maximum(var,1e-14))
        dm = np.einsum('n,nmd->md', self.alpha, dk)
        ds = -np.einsum('nm,nmd->md', beta, dk)/sd[:,None]
        ds[var<=1e-14] = 0
        if self.active:
            pm, pd = self.prior.value_gradient(q)
            mu += pm
            dm += pd
        return mu, sd, dm, ds

    def value_gradient(self, q):
        mu, sd, dm, ds = self.posterior(q)
        value, cm, cs = log_ei_parts(mu, sd, self.best)
        gradient = -cm[:,None]*dm + cs[:,None]*ds
        assert np.isfinite(value).all() and np.isfinite(gradient).all()
        return value, gradient


def original_scores(q, x, y, p, gp, active):
    pm = p(q) if active else np.zeros(len(q))
    gm, sd = gp.predict(q)
    mu = pm+gm
    best = (float(min(y))-p.center)/p.scale
    logei = log_ei_parts(mu, sd, best)[0]
    duplicate = cdist(q, x.numpy()).min(1) <= 1e-6
    logei[duplicate] = -np.inf
    return logei, pm, mu, sd


def propose(q, x, y, p, gp, active, smooth_prior):
    start_scores = original_scores(q, x, y, p, gp, active)[0]
    order = np.argsort(-start_scores, kind='stable')[:RESTARTS]
    acq = SmoothAcquisition(gp,smooth_prior,active,(float(min(y))-p.center)/p.scale)
    best_seen = [-np.inf, q[order[0]].astype(float).copy()]
    def objective(flat):
        z = flat.reshape(RESTARTS,20)
        value, grad = acq.value_gradient(z)
        j = int(value.argmax())
        if value[j] > best_seen[0]:best_seen[:] = [float(value[j]),z[j].copy()]
        return -float(value.sum()), -grad.ravel()
    result = minimize(objective, q[order].astype(float).ravel(), method='L-BFGS-B', jac=True,
                      bounds=[(-5.,5.)]*(RESTARTS*20),
                      options=dict(maxiter=MAXITER,maxfun=MAXFUN,maxls=10,ftol=1e-9,gtol=1e-6,maxcor=10))
    extra = np.vstack((result.x.reshape(RESTARTS,20),best_seen[1])).clip(-5,5).astype('float32')
    candidates = np.vstack((q,extra))
    logei, pm, mu, sd = original_scores(candidates,x,y,p,gp,active)
    j = int(logei.argmax())
    assert np.isfinite(logei[j]) and logei[j] >= start_scores.max()
    meta = dict(logei=float(logei[j]), original_logei=float(start_scores.max()),
                continuous_selected=bool(j>=len(q)),nfev=int(result.nfev),nit=int(result.nit),
                status=int(result.status),success=bool(result.success),extra=extra)
    chosen=(j,float(pm[j]),float(mu[j]),float(sd[j]),float(np.exp(logei[j])))
    return candidates,chosen,meta
