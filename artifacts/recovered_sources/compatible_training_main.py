import os
import sys
import pickle
import numpy as np
import torch
import argparse
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from matplotlib import pyplot as plt
from tqdm import tqdm

from NeurGO.problem import Problem
from NeurGO.model import NeurGO
from train_sets import TRAIN_FUNCTIONS, gen_train_offset, get_train_fitness
from Benchmark.cecfunctions import FUNCTIONS as F
from Benchmark.bbobfunctions import FUNCTIONS as BBOBF
from Benchmark.utils import getFitness, genOffset, setOffset, getOffset

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIM = 200  # 与训练时保持一致

class myProblem(Problem):
    def __init__(self, fun=None, repaire=True, dim=None):
        super().__init__()
        self.fun = fun
        self.useRepaire = repaire
        self.dim = dim

    def repaire(self, x):
        """Clamp the population within the search boundaries."""
        x = torch.clamp(x, self.fun['xlb'], self.fun['xub'])
        return x

    def calfitness(self, x):
        if self.useRepaire:
            x1 = self.repaire(x)
        else:
            x1 = x

        if self.fun['fid'].startswith('lf'):
            r = get_train_fitness(x1, self.fun)
        elif self.fun['fid'].startswith('gen'):  # ← 新增
            r = self.fun['fun'](x1)  # ← 直接调用 PyTorch 函数
        else:
            r = getFitness(x1, self.fun)

        return r

    def genRandomPop(self, batchShape):
        lb = self.fun['xlb']
        ub = self.fun['xub']
        return torch.rand(batchShape, device=DEVICE) * (ub - lb) + lb

    def reoffset(self):
        genOffset(self.dim, self.fun)

    def setOffset(self, offset):
        for key in offset.keys():
            self.fun[key] = offset[key]

    def lossFunc(self, father, all_kcand, lamda=0.3):
        """NeurGO Quality-Diversity (QD) Loss"""
        father = self.repaire(father)
        all_kcand = self.repaire(all_kcand)
        fit_father = self.calfitness(father)
        fit_kcand = self.calfitness(all_kcand)
        tau = 1e-8

        # Z-score normalization
        fit_father = (fit_father - fit_father.mean(dim=1, keepdim=True)) / (fit_father.std(dim=1, keepdim=True) + tau)
        fit_kcand = (fit_kcand - fit_kcand.mean(dim=1, keepdim=True)) / (fit_kcand.std(dim=1, keepdim=True) + tau)

        # 1. Quality Loss
        best_father, _ = torch.min(fit_father, dim=1)
        best_kcand, _ = torch.min(fit_kcand, dim=1)
        r = best_kcand - best_father
        loss_quality = torch.mean(r)

        # 2. Diversity Loss — 改为对候选解坐标做 Z-score，消除域尺度影响
        b, K, d = all_kcand.shape
        cand_mean = all_kcand.mean(dim=1, keepdim=True)
        cand_std = all_kcand.std(dim=1, keepdim=True).clamp(min=tau)
        cand_norm = (all_kcand - cand_mean) / cand_std  # 归一化到单位方差
        cand_dist = torch.cdist(cand_norm, cand_norm, p=2)
        loss_diversity = -torch.mean(cand_dist.mean(dim=(1, 2)))

        loss = loss_quality + lamda * loss_diversity
        return loss

    def getfunname(self):
        return self.fun['fid']

    def setfun(self, fun):
        self.fun = fun

class hpoProblem(Problem):
    def __init__(self, fun=None, repaire=True, dim=None):
        super().__init__()
        self.fun = fun
        self.useRepaire = repaire
        self.dim = dim

    def repaire(self, x):
        x = torch.clamp(x, self.fun['xlb'], self.fun['xub'])
        return x

    def calfitness(self, x):
        if self.useRepaire:
            x1 = self.repaire(x)
        else:
            x1 = x
        # HPO 函数直接接受 (batch, n, dim)，无需 reshape
        r = self.fun['fun'](x1)
        return r

    def genRandomPop(self, batchShape):
        lb = self.fun['xlb']
        ub = self.fun['xub']
        return torch.rand(batchShape, device=DEVICE) * (ub - lb) + lb

    def reoffset(self):
        pass

    def setOffset(self, offset):
        pass

    def lossFunc(self, father, all_kcand, lamda=0.3):
        father = self.repaire(father)
        all_kcand = self.repaire(all_kcand)
        fit_father = self.calfitness(father)
        fit_kcand = self.calfitness(all_kcand)
        tau = 1e-8
        fit_father = (fit_father - fit_father.mean(dim=1, keepdim=True)) / (fit_father.std(dim=1, keepdim=True) + tau)
        fit_kcand  = (fit_kcand  - fit_kcand.mean(dim=1,  keepdim=True)) / (fit_kcand.std(dim=1,  keepdim=True)  + tau)
        best_father, _ = torch.min(fit_father, dim=1)
        best_kcand,  _ = torch.min(fit_kcand,  dim=1)
        loss_quality = torch.mean(best_kcand - best_father)
        b, K, d = all_kcand.shape
        cand_dist = torch.cdist(all_kcand, all_kcand, p=2)
        scale = (self.fun['xub'] - self.fun['xlb']) * np.sqrt(d)
        loss_diversity = -torch.mean(cand_dist.mean(dim=(1, 2)) / (scale + tau))
        return loss_quality + lamda * loss_diversity

    def getfunname(self):
        return self.fun['fid']

    def setfun(self, fun):
        self.fun = fun
        self.dim = fun['_dim']
class bbobProblem(Problem):
    def __init__(self, fun=None, repaire=True, dim=None):
        super().__init__()
        self.fun = fun
        self.useRepaire = repaire
        self.dim = dim

    def repaire(self, x):
        x = torch.clamp(x, self.fun['xlb'], self.fun['xub'])
        return x

    def calfitness(self, x):
        if self.useRepaire:
            x1 = self.repaire(x)
        else:
            x1 = x
        b, n, d = x.shape
        x1 = x1.reshape(-1, d)
        r = getFitness(x1, self.fun)
        r = torch.unsqueeze(r, -1)
        r = r.view((b, n))
        return r

    def genRandomPop(self, batchShape):
        lb = self.fun['xlb']
        ub = self.fun['xub']
        return torch.rand(batchShape, device=DEVICE) * (ub - lb) + lb

    def reoffset(self):
        genOffset(self.dim, self.fun)

    def setOffset(self, offset):
        for key in offset.keys():
            self.fun[key] = offset[key]

    def lossFunc(self, father, all_kcand, lamda=0.3):
        father = self.repaire(father)
        all_kcand = self.repaire(all_kcand)
        fit_father = self.calfitness(father)
        fit_kcand = self.calfitness(all_kcand)
        tau = 1e-8
        fit_father = (fit_father - fit_father.mean(dim=1, keepdim=True)) / (fit_father.std(dim=1, keepdim=True) + tau)
        fit_kcand = (fit_kcand - fit_kcand.mean(dim=1, keepdim=True)) / (fit_kcand.std(dim=1, keepdim=True) + tau)
        best_father, _ = torch.min(fit_father, dim=1)
        best_kcand, _ = torch.min(fit_kcand, dim=1)
        r = best_kcand - best_father
        loss_quality = torch.mean(r)
        b, K, d = all_kcand.shape
        cand_dist = torch.cdist(all_kcand, all_kcand, p=2)
        scale = (self.fun['xub'] - self.fun['xlb']) * np.sqrt(d)
        mean_dist = cand_dist.mean(dim=(1, 2))
        loss_diversity = -torch.mean(mean_dist / (scale + tau))
        loss = loss_quality + lamda * loss_diversity
        return loss

    def getfunname(self):
        return self.fun['fid']

    def setfun(self, fun):
        self.fun = fun

class DimAdapter:
    """
    将高维问题适配到低维模型的随机嵌入适配器。
    A: (low_dim, high_dim) 的随机正交投影矩阵
    优化在低维子空间中进行，再映射回高维。
    """
    def __init__(self, high_dim, low_dim, device):
        self.high_dim = high_dim
        self.low_dim  = low_dim
        # 生成随机正交基（列正交）
        A = torch.randn(high_dim, low_dim, device=device)
        A, _ = torch.linalg.qr(A)          # (high_dim, low_dim)
        self.A = A                           # 高维→低维的投影基
        self.center = torch.zeros(high_dim, device=device)

    def to_low(self, x_high):
        """x_high: (b, n, high_dim) → (b, n, low_dim)"""
        return x_high @ self.A              # (b,n,d_high)@(d_high,d_low)

    def to_high(self, x_low):
        """x_low: (b, n, low_dim) → (b, n, high_dim)"""
        return x_low @ self.A.T             # (b,n,d_low)@(d_low,d_high)
# === Training Function ===
def train(expname=None, dim=None, hiddendim=None, popsize=100, problem=None,
          maxepoch=None, lr=None, batchsize=None, T=10, funset=None, needsave=True):
    # Initialization

    model = NeurGO(dim=dim, hidden_dim=hiddendim, popSize=popsize).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    batchShape = (batchsize, popsize, dim)
    bar = tqdm(range(maxepoch), ncols=120)
    losslist = []
    minloss = None
    ACCUM_STEPS = 4

    os.makedirs('./imgs/trainloss', exist_ok=True)
    os.makedirs('./ckpt', exist_ok=True)

    for epoch in bar:
        lamda = 0.5 * (1 - epoch / maxepoch)
        if (epoch + 1) % 100 == 0:
            for param_group in opt.param_groups:
                param_group['lr'] *= 0.9

        if epoch == 0 or (epoch + 1) % T == 0:
            for fun in funset:
                gen_train_offset(dim, fun)
            if len(losslist) > 0:
                plt.figure(figsize=(12, 9))
                plt.plot(losslist)
                plt.savefig(f'./imgs/trainloss/{expname}_d{dim}.png')
                plt.close()

        # train() 中替换 funset 循环部分

        totalloss = 0.0
        valid_count = 0
        opt.zero_grad()

        for i, fun in enumerate(funset):
            problem.setfun(fun)
            pop = problem.genRandomPop(batchShape)

            offpop, trail, evalnums, all_candidates = model(pop, problem)
            loss = problem.lossFunc(pop, all_candidates, lamda) / len(funset)

            if not torch.isfinite(loss):
                print(f"\n[WARN] epoch={epoch} fun={fun['fid']} loss={loss.item():.4f}, skipped")
                continue

            loss.backward()
            totalloss += loss.item()
            valid_count += 1

            if valid_count % ACCUM_STEPS == 0 or (i + 1) == len(funset):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10, norm_type=2)
                opt.step()
                opt.zero_grad()

        losslist.append(totalloss)

        if (minloss is None or totalloss < minloss) and needsave:
            minloss = totalloss
            torch.save(model.state_dict(), f'./ckpt/{expname}_d{dim}.pth')

        bar.set_description(f"NeurGO({expname})_dim({dim})_loss:{totalloss:.6f}|min:{minloss:.6f}")


def test(expname=None, dim=None, hiddendim=None, popsize=100, problem=None,
         batchsize=None, runs=1):
    model = NeurGO(dim=dim, hidden_dim=hiddendim, popSize=popsize).to(DEVICE)

    ckpt_path = f'./ckpt/{expname}_d{dim}.pth'
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        print(f'NeurGO checkpoint {ckpt_path} loaded!')
    else:
        print(f'Checkpoint not found: {ckpt_path}')
        return

    model.eval()
    batchShape = (batchsize, popsize, dim)
    bar = tqdm(range(runs))
    final_fits = []

    offset = {'bias': torch.zeros(dim, device=DEVICE)}
    problem.setOffset(offset)

    with torch.no_grad():
        for run in bar:
            pop = problem.genRandomPop(batchShape)
            offpop, trail, evalnums, all_candidates = model(pop, problem)
            batch_final_fit = trail[:, -1]
            final_fits.append(batch_final_fit)

    all_final_fits = torch.cat(final_fits, dim=0)
    print(f"NeurGO-{problem.getfunname()} | mean:{torch.mean(all_final_fits):.2E}({torch.std(all_final_fits):.2E})")

    totaltrail = {
        'final_fits': all_final_fits.detach().cpu().numpy(),
        'evalnums': evalnums,
        'mean': np.mean(all_final_fits.detach().cpu().numpy()),
        'std': np.std(all_final_fits.detach().cpu().numpy())
    }

    os.makedirs('./trails', exist_ok=True)
    with open(f'./trails/exp({expname})_f({problem.getfunname()})_dim({dim}).pkl', 'wb') as f:
        pickle.dump(totaltrail, f)


def expForFunction(dim=5, expname='neurgo_base',
                   popsize=100, maxepoch=1000, lr=0.001):
    print('Start training NeurGO...')
    from train_sets_gen import GEN_TRAIN_FUNCTIONS  # ← 新增这行
    funset = TRAIN_FUNCTIONS + GEN_TRAIN_FUNCTIONS  # ← 改这行（原来是 funset = TRAIN_FUNCTIONS）
    # funset = TRAIN_FUNCTIONS
    problem = myProblem(fun=TRAIN_FUNCTIONS[0], dim=dim, repaire=True)
    hiddendim = 200
    batchsize = 64
    T = 20
    train(expname=expname, dim=dim, hiddendim=hiddendim, popsize=popsize,
          problem=problem, maxepoch=maxepoch, lr=lr, batchsize=batchsize, T=T, funset=funset)


def testSysFuns(dim=5, expname='neurgo_base', popsize=100):
    problem = myProblem(fun=None, dim=dim, repaire=True)
    hiddendim = 200
    batchsize = 10
    testfunset = [F[f'cecf{i}'] for i in range(1, 7)]
    for fun in testfunset:
        problem.setfun(fun)
        test(expname=expname, dim=dim, hiddendim=hiddendim, popsize=popsize, problem=problem, batchsize=batchsize,
             runs=1)


def testBBOBFuns(expname=None, dim=None, hiddendim=200, popsize=100, problem=None,
                 batchsize=10, runs=1):
    model = NeurGO(dim=dim, hidden_dim=hiddendim, popSize=popsize).to(DEVICE)
    ckpt_path = f'./ckpt/{expname}_d{dim}.pth'
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print(f"Error: Checkpoint {ckpt_path} not found.")
        return
    model.eval()
    batchShape = (batchsize, popsize, dim)
    bar = tqdm(range(runs))
    final_fits = []
    with torch.no_grad():
        for run in bar:
            pop = problem.genRandomPop(batchShape)
            offpop, trail, evalnums, all_candidates = model(pop, problem)
            batch_final_fit = trail[:, -1]
            final_fits.append(batch_final_fit)
    all_final_fits = torch.cat(final_fits, dim=0)
    print(f"NeurGO-{problem.getfunname()} | mean:{torch.mean(all_final_fits):.2E}({torch.std(all_final_fits):.2E})")
    totaltrail = {
        'final_fits': all_final_fits.detach().cpu().numpy(),
        'evalnums': evalnums,
        'mean': np.mean(all_final_fits.detach().cpu().numpy()),
        'std': np.std(all_final_fits.detach().cpu().numpy())
    }
    with open(f'./trails/exp({expname})_f({problem.getfunname()})_dim({dim}).pkl', 'wb') as f:
        pickle.dump(totaltrail, f)
    return (problem.getfunname(), 'NeurGO', f'{totaltrail["mean"]:.2E}({totaltrail["std"]:.2E})')

def testHPOFuns(expname=None, dim=None, hiddendim=200, popsize=100, problem=None,
       batchsize=10, runs=1):
    model = NeurGO(dim=dim, hidden_dim=hiddendim, popSize=popsize).to(DEVICE)
    ckpt_path = f'./ckpt/{expname}_d{dim}.pth'
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print(f"Error: Checkpoint {ckpt_path} not found.")
        return
    model.eval()
    batchShape = (batchsize, popsize, dim)
    bar = tqdm(range(runs))
    final_fits = []
    with torch.no_grad():
        for run in bar:
            pop = problem.genRandomPop(batchShape)
            offpop, trail, evalnums, all_candidates = model(pop, problem)
            batch_final_fit = trail[:, -1]
            final_fits.append(batch_final_fit)
    all_final_fits = torch.cat(final_fits, dim=0)
    print(f"NeurGO-{problem.getfunname()} | mean:{torch.mean(all_final_fits):.2E}({torch.std(all_final_fits):.2E})")
    totaltrail = {
        'final_fits': all_final_fits.detach().cpu().numpy(),
        'evalnums': evalnums,
        'mean': np.mean(all_final_fits.detach().cpu().numpy()),
        'std': np.std(all_final_fits.detach().cpu().numpy())
    }
    os.makedirs('./trails', exist_ok=True)
    with open(f'./trails/exp({expname})_f({problem.getfunname()})_dim({dim}).pkl', 'wb') as f:
        pickle.dump(totaltrail, f)
    return (problem.getfunname(), 'NeurGO',
            f'{totaltrail["mean"]:.2E}({totaltrail["std"]:.2E})')

def testHighDimFuns(expname=None, high_dim=100, low_dim=10,
                    hiddendim=200, popsize=100, problem=None,
                    batchsize=10, runs=1):

    model = NeurGO(dim=low_dim, hidden_dim=hiddendim, popSize=popsize).to(DEVICE)
    ckpt_path = f'./ckpt/{expname}_d{low_dim}.pth'
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        print(f'Loaded checkpoint: {ckpt_path}')
    else:
        print(f"Error: Checkpoint {ckpt_path} not found.")
        return
    model.eval()

    adapter = DimAdapter(high_dim, low_dim, DEVICE)

    # ── 构造低维临时 problem，专门用于 generator 内部的 repaire ──
    low_dim_fun = {
        'fid': str(problem.getfunname()) + '_low',
        'xlb': problem.fun['xlb'],
        'xub': problem.fun['xub'],
    }
    low_problem = myProblem(fun=low_dim_fun, dim=low_dim, repaire=True)

    final_fits = []

    with torch.no_grad():
        for run in range(runs):
            pop_high = problem.genRandomPop((batchsize, popsize, high_dim))

            evalnum = 0
            trail = None
            fatherfit = problem.calfitness(pop_high)
            evalnum += popsize

            fatherfit, fitindex = torch.sort(fatherfit, dim=-1)
            pop_high_sorted = torch.zeros_like(pop_high)
            for i in range(pop_high.shape[0]):
                pop_high_sorted[i] = pop_high[i][fitindex[i]]
            pop_high = pop_high_sorted

            while evalnum < model.MaxNFE:
                pop_low = adapter.to_low(pop_high)

                # ── 传入 low_problem 而非 None ──
                k_cand_low = model.generator(pop_low, problem=low_problem, xfit=fatherfit)

                k_cand_high = adapter.to_high(k_cand_low)
                k_cand_high = problem.repaire(k_cand_high)

                rest_NFE = min(model.k_nums, model.MaxNFE - evalnum)
                if rest_NFE == 0:
                    break

                fits_list = []
                for i in range(rest_NFE):
                    ind = k_cand_high[:, i:i+1, :]
                    fits_list.append(problem.calfitness(ind))

                all_fits = torch.cat(fits_list, dim=1)
                evalnum += rest_NFE

                extended_pop = torch.cat([pop_high, k_cand_high[:, :rest_NFE, :]], dim=1)
                extended_fit = torch.cat([fatherfit, all_fits], dim=1)
                sorted_fit, fitindex = torch.sort(extended_fit, dim=-1)
                sorted_pop = torch.zeros_like(extended_pop)
                for i in range(extended_pop.shape[0]):
                    sorted_pop[i] = extended_pop[i][fitindex[i]]

                pop_high  = sorted_pop[:, :popsize, :]
                fatherfit = sorted_fit[:, :popsize]

                t = torch.min(fatherfit, dim=-1)[0].view(-1, 1)
                trail = t if trail is None else torch.cat((trail, t), dim=-1)

            final_fits.append(trail[:, -1])

    all_final_fits = torch.cat(final_fits, dim=0)
    print(f"NeurGO(d{low_dim}→d{high_dim})-{problem.getfunname()} | "
          f"mean:{torch.mean(all_final_fits):.2E}"
          f"({torch.std(all_final_fits):.2E})")

    totaltrail = {
        'final_fits': all_final_fits.detach().cpu().numpy(),
        'mean': np.mean(all_final_fits.detach().cpu().numpy()),
        'std':  np.std(all_final_fits.detach().cpu().numpy())
    }
    os.makedirs('./trails', exist_ok=True)
    with open(f'./trails/exp({expname}_d{low_dim}to{high_dim})'
              f'_f({problem.getfunname()}).pkl', 'wb') as f:
        pickle.dump(totaltrail, f)
    return (problem.getfunname(), f'NeurGO_d{low_dim}to{high_dim}',
            f'{totaltrail["mean"]:.2E}({totaltrail["std"]:.2E})')
def genBBOBoffset(dim=10):
    offsets = dict()
    bar = tqdm(range(1, 25))
    for fid in bar:
        f = BBOBF[fid]
        genOffset(dim, f)
        # if not fid in [5, 24]:
        #     f['xopt'] = torch.zeros((dim,)).cuda()
        # f['fopt'] = 0
        offsets[fid] = getOffset(f)
    with open(f'bbobOffsets_dim{dim}.pkl', 'wb') as f:
        pickle.dump(offsets, f)
# ============================================================
# 将以下内容追加到 NeurGO main.py 的末尾（parseargs() 之前）
# ============================================================

import importlib.util, sys as _sys

def _load_generated_funset(training_set_dir: str, dim: int, device):
    """
    从 EoB 生成的 training_set 目录加载训练函数，
    转换为 NeurGO 所需的 fun 字典列表。
    """
    func_dir = os.path.join(training_set_dir, "functions")
    func_files = sorted([
        f for f in os.listdir(func_dir)
        if f.startswith("func_") and f.endswith(".py")
    ])

    funset = []
    for fname in func_files:
        fpath = os.path.join(func_dir, fname)
        mod_name = fname.replace(".py", "")
        spec = importlib.util.spec_from_file_location(mod_name, fpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        prob = mod.get_problem(dim=dim, lb=-5.0, ub=5.0)

        fun = {
            'fid': mod_name,                           # e.g. 'func_0000'
            'xlb': torch.tensor(-5.0, device=device),
            'xub': torch.tensor( 5.0, device=device),
            '_prob': prob,                             # TrainingProblem 实例
        }
        funset.append(fun)

    print(f"[genFunset] Loaded {len(funset)} generated training functions from '{training_set_dir}'")
    return funset


class genProblem(Problem):
    """
    NeurGO Problem 接口，适配 EoB 生成的 TrainingProblem。
    fun 字典需包含 '_prob' 键（TrainingProblem 实例）。
    """
    def __init__(self, fun=None, repaire=True, dim=None):
        super().__init__()
        self.fun = fun
        self.useRepaire = repaire
        self.dim = dim

    def repaire(self, x):
        return torch.clamp(x, self.fun['xlb'], self.fun['xub'])

    # def calfitness(self, x):
    #     """x: (batch, n, dim) tensor → (batch, n) tensor"""
    #     if self.useRepaire:
    #         x = self.repaire(x)
    #     b, n, d = x.shape
    #     # 转 numpy → 调用生成函数 → 转回 tensor
    #     x_np = x.reshape(-1, d).detach().cpu().numpy()
    #     y_np = self.fun['_prob'].func(x_np)           # shape: (b*n,)
    #     import numpy as _np
    #     y_np = _np.nan_to_num(y_np, nan=1e10, posinf=1e10, neginf=1e10)
    #     y = torch.tensor(y_np, dtype=torch.float32, device=x.device)
    #     return y.view(b, n)
    def calfitness(self, x):
        if self.useRepaire:
            x = self.repaire(x)
        b, n, d = x.shape
        x_np = x.reshape(-1, d).detach().cpu().numpy()
        y_np = self.fun['_prob'].func(x_np)

        # 过滤无效值
        y_np = np.nan_to_num(y_np, nan=0.0, posinf=0.0, neginf=0.0)

        # 归一化到 [-1, 1] 量级，消除函数间量级差异
        y_std = np.std(y_np)
        if y_std > 1e-8:
            y_np = (y_np - np.mean(y_np)) / y_std

        y = torch.tensor(y_np, dtype=torch.float32, device=x.device)
        return y.view(b, n)
    def genRandomPop(self, batchShape):
        lb = self.fun['xlb']
        ub = self.fun['xub']
        return torch.rand(batchShape, device=DEVICE) * (ub - lb) + lb

    def reoffset(self):
        pass   # 生成函数无 offset 机制

    def setOffset(self, offset):
        pass

    def lossFunc(self, father, all_kcand, lamda=0.3):
        """与 myProblem.lossFunc 完全相同"""
        father    = self.repaire(father)
        all_kcand = self.repaire(all_kcand)
        fit_father = self.calfitness(father)
        fit_kcand  = self.calfitness(all_kcand)
        tau = 1e-8

        fit_father = (fit_father - fit_father.mean(dim=1, keepdim=True)) / \
                     (fit_father.std(dim=1,  keepdim=True) + tau)
        fit_kcand  = (fit_kcand  - fit_kcand.mean(dim=1,  keepdim=True)) / \
                     (fit_kcand.std(dim=1,   keepdim=True) + tau)

        best_father, _ = torch.min(fit_father, dim=1)
        best_kcand,  _ = torch.min(fit_kcand,  dim=1)
        loss_quality   = torch.mean(best_kcand - best_father)

        b, K, d = all_kcand.shape
        cand_mean = all_kcand.mean(dim=1, keepdim=True)
        cand_std  = all_kcand.std(dim=1,  keepdim=True).clamp(min=tau)
        cand_norm = (all_kcand - cand_mean) / cand_std
        cand_dist = torch.cdist(cand_norm, cand_norm, p=2)
        loss_diversity = -torch.mean(cand_dist.mean(dim=(1, 2)))

        return loss_quality + lamda * loss_diversity

    def getfunname(self):
        return self.fun['fid']

    def setfun(self, fun):
        self.fun = fun


def expForGeneratedFunctions(
    training_set_dir: str,
    dim: int = 10,
    expname: str = 'neurgo_gen',
    popsize: int = 100,
    maxepoch: int = 1000,
    lr: float = 0.001,
):
    """
    使用 EoB 生成的训练集训练 NeurGO。

    Args:
        training_set_dir: EoB 输出的 training_set 目录路径
        dim:       问题维度（与生成时保持一致）
        expname:   实验名，checkpoint 保存为 ckpt/{expname}_d{dim}.pth
        popsize:   种群大小
        maxepoch:  训练轮数
        lr:        学习率
    """
    funset  = _load_generated_funset(training_set_dir, dim, DEVICE)
    problem = genProblem(fun=funset[0], dim=dim, repaire=True)

    print(f"[expForGeneratedFunctions] Training NeurGO with {len(funset)} generated functions")
    print(f"  expname={expname}, dim={dim}, popsize={popsize}, maxepoch={maxepoch}, lr={lr}")

    train(
        expname   = expname,
        dim       = dim,
        hiddendim = HIDDEN_DIM,
        popsize   = popsize,
        problem   = problem,
        maxepoch  = maxepoch,
        lr        = lr,
        batchsize = 64,
        T         = 20,
        funset    = funset,
        needsave  = True,
    )

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', '-d', required=True, type=int, help='a integer stands for the dimension')
    parser.add_argument('--expname', '-expname', required=True, type=str, default='neurgo_test')
    parser.add_argument('--popsize', '-popsize', required=True, type=int, default=100)
    parser.add_argument('--maxepoch', '-maxepoch', required=False, type=int, default=1000)
    parser.add_argument('--lr', '-lr', required=False, type=float, default=0.001)
    parser.add_argument('--mode', '-mode', required=True, type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--target', '-target', required=False, type=str, default='bbob', choices=['sys', 'bbob','hpo', 'bbob100', 'sys100', 'gen'])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()
    dim = args.dim
    expname = args.expname
    popsize = args.popsize
    maxepoch = args.maxepoch
    lr = args.lr
    mode = args.mode
    target = args.target

    if mode == 'train':
        print(f'Ready to train NeurGO (dim:{dim}), expname:{expname}')
        # expForFunction(dim=dim, expname=expname, popsize=popsize, maxepoch=maxepoch, lr=lr)

        if target == 'gen':  # ← 新增这个分支
            expForGeneratedFunctions(
                training_set_dir='/home/skq/桌面/xixi/main-scheme/exps/training_set_v2',
                dim=dim,
                expname=expname,
                popsize=popsize,
                maxepoch=maxepoch,
                lr=lr,
            )
        else:  # ← 原来的训练
            expForFunction(dim=dim, expname=expname, popsize=popsize, maxepoch=maxepoch, lr=lr)
    else:
        if target == 'sys':
            print('Ready to test on CEC functions (F1-F6)')
            testSysFuns(dim=dim, expname=expname, popsize=popsize)

        if target == 'bbob':
            print('Ready to test on bbob functions')
            testfunset = [i for i in range(1, 25)]
            problem = bbobProblem(fun=BBOBF[1], dim=dim, repaire=True)
            if not os.path.exists(f'bbobOffsets_dim{dim}.pkl'):
                genBBOBoffset(dim)
            with open(f'bbobOffsets_dim{dim}.pkl', 'rb') as f:
                offsets = pickle.load(f)
                for fid in testfunset:
                    fun = BBOBF[fid]
                    fun['xlb'] = -5;
                    fun['xub'] = 5
                    offset = offsets[fun['fid']]
                    setOffset(fun, offset)
                    problem.setfun(fun)
                    try:
                        testBBOBFuns(dim=dim, expname=expname, popsize=popsize, problem=problem)
                    except Exception as e:
                        print(f"Error testing function {fid}: {e}")
        if target == 'hpo':
            from Benchmark.hpofunctions import FUNCTIONS_D10 as HPOF

            print('Ready to test on HPO functions')
            os.makedirs('./trails', exist_ok=True)
            for fid, fun in HPOF.items():
                problem = hpoProblem(fun=fun, dim=dim, repaire=True)
                try:
                    testHPOFuns(dim=dim, expname=expname, popsize=popsize,
                                problem=problem, hiddendim=HIDDEN_DIM)
                except Exception as e:
                    print(f"Error testing function {fid}: {e}")
        if target == 'bbob100':
            print('Ready to test d10→d100 on BBOB functions')
            high_dim = 100
            low_dim = 10
            testfunset = [i for i in range(1, 25)]
            problem = bbobProblem(fun=BBOBF[1], dim=high_dim, repaire=True)

            # 生成100维的 BBOB offset
            if not os.path.exists(f'bbobOffsets_dim{high_dim}.pkl'):
                genBBOBoffset(high_dim)
            with open(f'bbobOffsets_dim{high_dim}.pkl', 'rb') as f:
                offsets = pickle.load(f)

            for fid in testfunset:
                fun = BBOBF[fid]
                fun['xlb'] = -5;
                fun['xub'] = 5
                setOffset(fun, offsets[fun['fid']])
                problem.setfun(fun)
                try:
                    testHighDimFuns(expname=expname, high_dim=high_dim, low_dim=low_dim,
                                    hiddendim=HIDDEN_DIM, popsize=popsize,
                                    problem=problem, batchsize=10)
                except Exception as e:
                    print(f"Error testing function {fid}: {e}")

        if target == 'sys100':
            print('Ready to test d10→d100 on CEC functions')
            high_dim = 100
            low_dim = 10
            testfunset = [F[f'cecf{i}'] for i in range(1, 7)]
            for fun in testfunset:
                fun['xlb'] = fun.get('xlb', -100)
                fun['xub'] = fun.get('xub', 100)
                problem = myProblem(fun=fun, dim=high_dim, repaire=True)
                try:
                    testHighDimFuns(expname=expname, high_dim=high_dim, low_dim=low_dim,
                                    hiddendim=HIDDEN_DIM, popsize=popsize,
                                    problem=problem, batchsize=10)
                except Exception as e:
                    print(f"Error testing function {fun['fid']}: {e}")

