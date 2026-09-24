from .utils import *
import numpy as np
import torch

################ CEC Test Functions (F1-F6) ####################

FUNCTIONS = dict()

# --- F1: Shifted Sphere ---
def cecfun1(x, b=None): 
    """F1: Shifted Sphere Function"""
    if b is not None:
        z = x - b
    else:
        z = x
    sc = torch.sum(torch.pow(z, 2), dim=2)
    return sc    

FUNCTIONS['cecf1'] = {
    'fid': 'cecf1',
    'fun': cecfun1,
    'bias': None,
    'xub': 100, 'xlb': -100,
    'bub': 50, 'blb': -50,
}


# --- F2: Shifted Schwefel 2.21 ---
def cecfun2(x, b=None): 
    if b is not None:
        z = x - b
    else:
        z = x
    z = torch.abs(z)
    sc = torch.max(z, dim=2)[0]
    return sc

FUNCTIONS['cecf2'] = {
    'fid': 'cecf2',
    'fun': cecfun2,
    'bias': None,
    'xub': 100, 'xlb': -100,
    'bub': 50, 'blb': -50,
}


# --- F3: Shifted Rosenbrock ---
def cecfun3(x, b=None): 
    if b is not None:
        z = x - b
    else:
        z = x
    x1 = z[:, :, :-1]
    x2 = z[:, :, 1:]
    return torch.sum(100 * torch.pow((torch.pow(x1, 2) - x2), 2) + torch.pow((x1 - 1), 2), dim=2)

FUNCTIONS['cecf3'] = {
    'fid': 'cecf3',
    'fun': cecfun3,
    'bias': None,
    'xub': 100, 'xlb': -100,
    'bub': 50, 'blb': -50,
}


# --- F4: Shifted Rastrigin ---
def cecfun4(x, b=None):     
    if b is not None:
        z = x - b
    else:
        z = x
    sc = torch.sum(torch.pow(z, 2) - 10 * torch.cos(2 * np.pi * z) + 10, dim=2)
    return sc

FUNCTIONS['cecf4'] = {
    'fid': 'cecf4',
    'fun': cecfun4,
    'bias': None,
    'xub': 5, 'xlb': -5,
    'bub': 2.5, 'blb': -2.5,
}


# --- F5: Shifted Griewank ---
def cecfun5(x, b=None):  
    if b is not None:
        z = x - b
    else:
        z = x
    i = torch.from_numpy(np.array([i + 1 for i in range(x.shape[2])])).view(-1).to(DEVICE).view(1, 1, x.shape[2])
    sc = torch.sum(torch.pow(z, 2) / 4000, dim=2) - torch.prod(torch.cos(z / torch.sqrt(i)), dim=2) + 1
    return sc

FUNCTIONS['cecf5'] = {
    'fid': 'cecf5',
    'fun': cecfun5,
    'bias': None,
    'xub': 600, 'xlb': -600,
    'bub': 300, 'blb': -300,
}


# --- F6: Shifted Ackley ---
def cecfun6(x, b=None):   
    if b is not None:
        z = x - b
    else:
        z = x
    dim = x.shape[2]
    term1 = -0.2 * torch.sqrt((1 / dim) * torch.sum(torch.pow(z, 2), dim=2))
    term2 = (1 / dim) * torch.sum(torch.cos(2 * np.pi * z), dim=2)
    sc = -20 * torch.exp(term1) - torch.exp(term2) + 20 + np.e
    return sc

FUNCTIONS['cecf6'] = {
    'fid': 'cecf6',
    'fun': cecfun6,
    'bias': None,
    'xub': 32, 'xlb': -32,
    'bub': 16, 'blb': -16,
}
