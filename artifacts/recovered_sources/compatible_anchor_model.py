import torch
import torch.nn as nn


class RankFitnessAttentionVParameter(nn.Module):
    def __init__(self, popSize=100, hiddenDim=200, dim=10):
        super().__init__()
        self.popSize = popSize
        self.num_heads = 2
        self.attn_rank = nn.Parameter(torch.randn((1, self.popSize, self.popSize)), requires_grad=True)
        self.fusion_weights = nn.Parameter(torch.randn((2,)), requires_grad=True)
        self.q = nn.Sequential(nn.Linear(1, hiddenDim))
        self.k = nn.Sequential(nn.Linear(1, hiddenDim))
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x, fitx):
        b, n, _ = fitx.shape
        q = self.q(fitx).view(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(fitx).view(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
        fitattn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)
        fitattn = torch.squeeze(fitattn.softmax(dim=-1).mean(dim=1, keepdim=True), dim=1)
        v = self.v_proj(x)
        y_rank = self.attn_rank[:, :n, :n].softmax(dim=-1) @ v
        y_fit = fitattn @ v
        weights = self.fusion_weights.softmax(-1)
        return y_rank * weights[0] + y_fit * weights[1]


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def sortpop(self, x, fitness):
        if fitness.dim() == 3 and fitness.size(-1) == 1:
            fitness = fitness.squeeze(-1)
        fitness, fitindex = torch.sort(fitness, dim=-1)
        y = torch.gather(x, dim=1, index=fitindex.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        return y, fitness


class EliteGeneratorVParameter(BaseModel):
    def __init__(self, dim=10, hidden_dim=200, popSize=100, temid=0, k_nums=2):
        super().__init__()
        self.dim = dim
        self.k_nums = k_nums
        self.pce_attn = RankFitnessAttentionVParameter(popSize=popSize, hiddenDim=hidden_dim, dim=dim)
        self.pce_ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.beta1 = nn.Parameter(torch.randn((1, popSize, 1)), requires_grad=True)
        self.beta2 = nn.Parameter(torch.randn((1, popSize, 1)), requires_grad=True)
        self.beta3 = nn.Parameter(torch.randn((1, popSize, 1)), requires_grad=True)
        self.esd_decoder = nn.Sequential(
            nn.Linear(2 * dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, k_nums),
        )

    def forward(self, x, problem, xfit=None):
        b, n, _ = x.shape
        fatherfit = xfit if xfit is not None else problem.calfitness(x)
        if fatherfit.dim() == 3 and fatherfit.size(-1) == 1:
            fatherfit = fatherfit.squeeze(-1)
        fitx = fatherfit.softmax(dim=-1).view(b, n, 1)
        crosspop = self.pce_attn(x, fitx)
        offpop = self.pce_ffn(crosspop)
        encoded_features = self.beta1 * x + self.beta2 * crosspop + self.beta3 * offpop
        combined = torch.cat([x, fitx, encoded_features], dim=-1)
        scores = self.esd_decoder(combined)
        weights = scores.softmax(dim=1)
        candidates = (encoded_features.transpose(1, 2) @ weights).transpose(1, 2)
        return problem.repaire(candidates)


class AnchorPolicyBackbone(BaseModel):
    def __init__(self, dim=10, hidden_dim=200, popSize=100):
        super().__init__()
        self.popsize = popSize
        self.k_nums = 2
        self.MaxNFE = 300
        self.generator = EliteGeneratorVParameter(dim, hidden_dim, popSize, 0, k_nums=self.k_nums)

    def forward(self, x, problem):
        self.evalnum = 0
        self.trail = None
        self.all_k_candidates = []
        fatherfit = problem.calfitness(x)
        self.evalnum += x.shape[1]
        x, fatherfit = self.sortpop(x, fatherfit)
        all_k_candidates = []
        while self.evalnum < self.MaxNFE:
            k_candidates = self.generator(x, problem, fatherfit)
            rest_nfe = min(self.k_nums, self.MaxNFE - self.evalnum)
            if rest_nfe == 0:
                break
            fits = []
            for i in range(rest_nfe):
                fits.append(problem.calfitness(k_candidates[:, i:i + 1, :]))
            all_candidates = k_candidates[:, :rest_nfe, :]
            all_fits = torch.cat(fits, dim=1)
            if all_fits.dim() == 3 and all_fits.size(-1) == 1:
                all_fits = all_fits.squeeze(-1)
            self.evalnum += rest_nfe
            all_k_candidates.append(all_candidates)
            x, fatherfit = self.sortpop(torch.cat([x, all_candidates], dim=1), torch.cat([fatherfit, all_fits], dim=1))
            x = x[:, :self.popsize, :]
            fatherfit = fatherfit[:, :self.popsize]
            trail = fatherfit[:, 0].view(-1, 1)
            self.trail = trail if self.trail is None else torch.cat([self.trail, trail], dim=-1)
        if all_k_candidates:
            self.all_k_candidates = torch.cat(all_k_candidates, dim=1)
        else:
            self.all_k_candidates = torch.empty((x.shape[0], 0, x.shape[2]), device=x.device, dtype=x.dtype)
        return x, self.trail, self.evalnum, self.all_k_candidates
