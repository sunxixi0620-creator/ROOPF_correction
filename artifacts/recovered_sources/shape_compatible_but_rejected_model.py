import torch.nn as nn
import torch
from NeurGO.imports import *
from NeurGO.problem import *


class RankFitnessAttention(nn.Module):
    """
    Rank-Fitness Dual-Aware Attention mechanism.
    Updated with Learnable Value (V) Projection.
    """

    def __init__(self, popSize=100, hiddenDim=128, dim=128):  # 增加了 dim 参数
        super().__init__()
        self.popSize = popSize
        self.dim = dim

        # 1. 定义 Q, K 的映射（用于计算“谁关注谁”）
        self.q = nn.Sequential(nn.Linear(1, hiddenDim))
        self.k = nn.Sequential(nn.Linear(1, hiddenDim))
        self.num_heads = 2

        # 2. 新增：定义 V 的映射（用于决定“提取什么信息”）
        # 将原始特征 x 映射到一个新的空间
        self.v_proj = nn.Linear(dim, dim)
        self.v_dropout = nn.Dropout(p=0.15)  # ← 新增

        # Learnable rank-based attention matrix (A_rank)
        self.attn_rank = nn.Parameter(torch.randn((1, self.popSize, self.popSize)), requires_grad=True)

        # Fusion weights for combining rank and fitness attention
        self.fusion_weights = nn.Parameter(torch.randn((2,)), requires_grad=True)

    def forward(self, x, fitx):
        B, N, C = fitx.shape

        # --- A. 计算注意力权重 (Q, K 交互) ---
        q = self.q(fitx).view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(fitx).view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        fitattn = q @ k.transpose(2, 3) * (self.dim ** -0.5)
        # 聚合多头权重得到 [B, N, N]
        fitattn = torch.squeeze(fitattn.softmax(dim=-1).mean(dim=1, keepdim=True), dim=1)

        # --- B. 计算 Value (V 映射) ---
        # 核心修改：不再直接用 x，而是用经过线性变换后的 v
        v = self.v_proj(x)  # 形状保持 [B, N, dim]
        v = self.v_dropout(v)  # ← 新增，训练时随机置零 15% 的神经元

        # --- C. 加权融合 (Weighted Sum) ---
        # Dual-branch attention 应用于转换后的 v
        # y1 = self.attn_rank.softmax(dim=-1) @ v
        # y2 = fitattn @ v

        y1 = self.attn_rank.softmax(dim=-1) @ x
        y2 = fitattn @ x

        # Weighted fusion
        weights = self.fusion_weights.softmax(-1)
        y = y1 * weights[0] + y2 * weights[1]
        return y


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def sortpop(self, x, fitness):
        """
        Sort population based on fitness values.
        """
        fitness, fitindex = torch.sort(fitness, dim=-1)
        y = torch.zeros_like(x)
        for index, pop in enumerate(x):
            pop = x[index]
            y[index] = torch.index_select(pop, 0, fitindex[index])
        return y, fitness


class EliteGenerator(BaseModel):
    """
    Generative Module consisting of PCE & ESD.
    """

    def __init__(self, dim=128, hidden_dim=128, popSize=10, temid=0, k_nums=2):  # 修改 dim 和 hidden_dim 为 128
        super().__init__()
        self.dim = dim
        self.k_nums = k_nums

        # Population Context Encoder (PCE)
        self.pce_attn = RankFitnessAttention(popSize=popSize, hiddenDim=hidden_dim,dim=dim)
        self.pce_ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),  # 使用平滑的 GELU 激活函数
            nn.Linear(dim, dim)
        )

        # Residual connection weights
        self.beta1 = nn.Parameter(torch.randn((1, popSize, 1)), requires_grad=True)
        self.beta2 = nn.Parameter(torch.randn((1, popSize, 1)), requires_grad=True)
        self.beta3 = nn.Parameter(torch.randn((1, popSize, 1)), requires_grad=True)

        # Elite Synthesis Decoder (ESD)
        self.esd_decoder = nn.Sequential(
            nn.Linear(2 * dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # 使用平滑的 GELU 激活函数
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),  # 使用平滑的 GELU 激活函数
            nn.Linear(hidden_dim // 2, k_nums)
        )

    def forward(self, x, problem, xfit=None):
        b, n, d = x.shape
        if xfit is not None:
            fatherfit = xfit
        else:
            fatherfit = problem.calfitness(x)

        fitx = fatherfit.softmax(dim=-1).view(b, n, 1)

        # PCE Forward Pass
        crosspop = self.pce_attn(x, fitx)
        offpop = self.pce_ffn(crosspop)
        encoded_features = self.beta1 * x + self.beta2 * crosspop + self.beta3 * offpop

        # ESD Forward Pass
        combined = torch.cat([x, fitx, encoded_features], dim=-1)
        scores = self.esd_decoder(combined)
        weights = scores.softmax(dim=1)

        # Synthesize elite candidates
        features_T = encoded_features.transpose(1, 2)
        candidates_T = features_T @ weights
        k_candidates = candidates_T.transpose(1, 2)

        # Boundary repair
        k_candidates = problem.repaire(k_candidates)
        return k_candidates


class NeurGO(BaseModel):
    """
    The main NeurGO framework.
    """

    def __init__(self, dim=128, hidden_dim=128, popSize=100):  # 修改默认参数为 128
        super().__init__()
        self.popsize = popSize
        self.k_nums = 2
        self.MaxNFE = 300

        # Initialize the generator
        self.generator = EliteGenerator(dim, hidden_dim, popSize, 0, k_nums=self.k_nums)

    def forward(self, x, problem):
        # Initialization
        self.evalnum = 0
        self.trail = None
        fatherfit = problem.calfitness(x)
        self.evalnum += (x.shape[1])
        self.all_k_candidates = []

        # Sort initial population
        x, fatherfit = self.sortpop(x, fatherfit)
        all_k_candidates = []

        # Optimization Loop
        while self.evalnum < self.MaxNFE:
            # Generate candidates
            k_candidates = self.generator(x, problem, fatherfit)

            # Check remaining budget
            rest_NFE = min(self.k_nums, self.MaxNFE - self.evalnum)
            if rest_NFE == 0:
                break

            # Evaluate candidates
            fits_list = []
            for i in range(rest_NFE):
                individual = k_candidates[:, i:i + 1, :]
                individual_fit = problem.calfitness(individual)
                fits_list.append(individual_fit)

            all_candidates = k_candidates[:, :rest_NFE, :]
            all_fits = torch.cat(fits_list, dim=1)
            self.evalnum += rest_NFE

            all_k_candidates.append(all_candidates)

            # Elitist Selection: Combine and select top-N individuals
            extended_pop = torch.cat([x, all_candidates], dim=1)
            extended_fit = torch.cat([fatherfit, all_fits], dim=1)

            sorted_pop, sorted_fit = self.sortpop(extended_pop, extended_fit)
            x = sorted_pop[:, :self.popsize, :]
            fatherfit = sorted_fit.squeeze(-1)[:, :self.popsize]

            # Update optimization trajectory
            trail = torch.min(fatherfit, dim=-1)[0].view(-1, 1)
            if self.trail is None:
                self.trail = trail
            else:
                self.trail = torch.cat((self.trail, trail), dim=-1)

        if len(all_k_candidates) > 0:
            self.all_k_candidates = torch.cat(all_k_candidates, dim=1)
        else:
            self.all_k_candidates = torch.empty((x.shape[0], 0, x.shape[2])).to(x.device)

        return x, self.trail, self.evalnum, self.all_k_candidates