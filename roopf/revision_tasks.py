"""Predeclared procedural training distribution, with no benchmark/ELA imports.

All 12 weight recipes x 3 scales are retained. No landscape is selected using
performance. This is a new controlled distribution, not reconstructed history.
"""
import torch
import torch.nn.functional as F
from .experiment_io import seed_for

# quadratic, curved coupling, periodic, radial wells, smooth shelves
RECIPES = (
    (1., .1, .0, .0, .0), (1., .8, .0, .0, .0),
    (.3, 1., .1, .0, .0), (.1, .5, 1., .0, .0),
    (.2, .0, 1., .3, .0), (.1, .0, .2, 1., .0),
    (.1, .1, .0, 1., .3), (.3, .0, .0, .2, 1.),
    (.1, .3, .2, .0, 1.), (.5, .5, .5, .0, .0),
    (.2, .2, .2, .6, .6), (.4, .4, .4, .4, .4),
)


class ProceduralTask:
    def __init__(self, fid, instance, split, dim=10, device='cpu'):
        self.fid, self.instance, self.split, self.dim = fid, instance, split, dim
        self.fun = {'fid': f'procedural_{fid}', 'xlb': -5., 'xub': 5.}
        self.points = 0
        self.diagnostic_points = 0
        self.initial_values = None
        self.parameter_seed = seed_for('revision_v1', split, 'task', fid, instance, dim)
        g = torch.Generator().manual_seed(self.parameter_seed)
        rand = lambda *s: torch.rand(s, generator=g)
        normal = lambda *s: torch.randn(s, generator=g)
        q, _ = torch.linalg.qr(normal(dim, dim))
        params = dict(rotation=q, shift=6*rand(dim)-3,
            weights=torch.tensor(RECIPES[fid//3]),
            axis=torch.exp(torch.linspace(0, (fid%3+1)*1.4, dim)),
            frequency=(fid%3+1)*.8+rand(dim), phase=6.283185*rand(dim),
            centers=6*rand(6, dim)-3, depths=.5+rand(6),
            dirs=normal(6, dim)/dim**.5, thresholds=2*rand(6)-1,
            objective_scale=torch.exp(2*rand()-1), objective_bias=4*rand()-2)
        self.params = {k: v.to(device) for k, v in params.items()}

    def getfunname(self):
        return self.fun['fid']

    def repaire(self, x):
        return x.clamp(-5, 5)

    def value(self, x):
        p = self.params
        z = (self.repaire(x)-p['shift']) @ p['rotation']
        quadratic = (p['axis']*z.square()).mean(-1)/p['axis'].mean()
        curved = (z[..., 1:]-.35*z[..., :-1].square()).square().mean(-1)
        periodic = (1-torch.cos(z*p['frequency']+p['phase'])).mean(-1)
        d2 = (z.unsqueeze(-2)-p['centers']).square().mean(-1)
        wells = -(p['depths']*torch.exp(-d2/2)).sum(-1)
        shelves = torch.tanh(z@p['dirs'].T-p['thresholds']).square().mean(-1)
        terms = torch.stack((quadratic, curved, periodic, wells, shelves), -1)
        return (terms*p['weights']).sum(-1)*p['objective_scale']+p['objective_bias']

    def calfitness(self, x):
        self.points += x.shape[0]*x.shape[1]
        y = self.value(x)
        if self.initial_values is None:
            self.initial_values = y.detach().clone()
        if not torch.isfinite(y).all():
            raise ValueError('Nonfinite procedural objective')
        return y

    def diagnostic_fitness(self, x):
        self.diagnostic_points += x.shape[0]*x.shape[1]
        return self.value(x)


def population(fid, instance, split, count=4, dim=10, device='cpu'):
    g = torch.Generator().manual_seed(seed_for('revision_v1', split, 'population', fid, instance, dim))
    return (10*torch.rand(count, 100, dim, generator=g)-5).to(device)
