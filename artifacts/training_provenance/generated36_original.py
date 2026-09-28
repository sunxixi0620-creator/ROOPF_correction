# === NeurGO trainset v3: 36 high-quality diverse streaming landscapes ===
# 自包含训练集；每次 gen_train_offset 都重新抽参，不依赖 theta.npz。
# 目标：36 个不同景观入口，覆盖不同黑盒优化景观，同时避免直接复刻 BBOB/CEC 命名函数。

import math
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_rotation(dim, device):
    H = torch.randn(dim, dim, device=device)
    Q, R = torch.linalg.qr(H)
    s = torch.sign(torch.diag(R))
    s = torch.where(s == 0, torch.ones_like(s), s)
    return Q * s.unsqueeze(0)


def _unit_vec(dim, device):
    v = torch.randn(dim, device=device)
    return v / (torch.linalg.norm(v) + 1e-12)


def _base_offset(dim, device, span=8.0, rotate=True):
    return {
        "R": _make_rotation(dim, device) if rotate else torch.eye(dim, device=device),
        "b": (torch.rand(dim, device=device) - 0.5) * span,
    }


# ============================================================
# 1-3: RBF 多漏斗族
# ============================================================
def offset_rbf_sparse(dim, device):
    p = _base_offset(dim, device)
    M = int(torch.randint(4, 8, (1,)).item())
    sign = (torch.rand(M, device=device) < 0.55).float() * 2.0 - 1.0
    p.update({
        "c": (torch.rand(M, dim, device=device) - 0.5) * 8.0,
        "depth": torch.empty(M, device=device).uniform_(0.4, 2.2) * sign,
        "width": torch.empty(M, device=device).uniform_(0.9, 2.8),
        "base": float(torch.empty(1).uniform_(0.015, 0.08).item()),
    })
    return p


def offset_rbf_dense(dim, device):
    p = _base_offset(dim, device)
    M = int(torch.randint(9, 16, (1,)).item())
    sign = (torch.rand(M, device=device) < 0.5).float() * 2.0 - 1.0
    p.update({
        "c": (torch.rand(M, dim, device=device) - 0.5) * 8.0,
        "depth": torch.empty(M, device=device).uniform_(0.2, 1.6) * sign,
        "width": torch.empty(M, device=device).uniform_(0.5, 1.8),
        "base": float(torch.empty(1).uniform_(0.005, 0.05).item()),
    })
    return p


def offset_rbf_deceptive(dim, device):
    p = _base_offset(dim, device)
    M = int(torch.randint(5, 11, (1,)).item())
    p.update({
        "c": (torch.rand(M, dim, device=device) - 0.5) * 8.5,
        "depth": torch.cat([
            torch.empty(1, device=device).uniform_(-3.0, -1.8),
            torch.empty(M - 1, device=device).uniform_(0.2, 1.8)
        ]),
        "width": torch.empty(M, device=device).uniform_(0.6, 2.3),
        "base": float(torch.empty(1).uniform_(0.01, 0.06).item()),
    })
    return p


def eval_rbf(x, p):
    z = (x - p["b"]) @ p["R"].T
    base = p["base"] * torch.sum(z * z, dim=2) / z.shape[-1]
    diff = z.unsqueeze(2) - p["c"].view(1, 1, p["c"].shape[0], p["c"].shape[1])
    d2 = torch.sum(diff * diff, dim=3)
    bumps = (p["depth"] * torch.exp(-d2 / (2.0 * p["width"] ** 2 + 1e-9))).sum(dim=2)
    return base + bumps


# ============================================================
# 4-6: soft plateau / soft step-like 平台族
# ============================================================
def offset_plateau_soft(dim, device):
    p = _base_offset(dim, device)
    M = int(torch.randint(3, 7, (1,)).item())
    dirs = torch.randn(M, dim, device=device)
    dirs = dirs / (torch.linalg.norm(dirs, dim=1, keepdim=True) + 1e-12)
    p.update({
        "dirs": dirs,
        "shift": torch.empty(M, device=device).uniform_(-2.5, 2.5),
        "sharp": float(torch.empty(1).uniform_(0.4, 1.4).item()),
        "amp": torch.empty(M, device=device).uniform_(0.2, 1.0),
        "base": float(torch.empty(1).uniform_(0.005, 0.06).item()),
        "tilt": float(torch.empty(1).uniform_(-0.12, 0.12).item()),
        "tilt_dir": _unit_vec(dim, device).view(dim, 1),
    })
    return p


def offset_plateau_sharp(dim, device):
    p = offset_plateau_soft(dim, device)
    p["sharp"] = float(torch.empty(1).uniform_(1.8, 4.0).item())
    p["base"] = float(torch.empty(1).uniform_(0.002, 0.04).item())
    return p


def offset_plateau_tilted(dim, device):
    p = offset_plateau_soft(dim, device)
    p["tilt"] = float(torch.empty(1).uniform_(0.10, 0.28).item()) * (1.0 if torch.rand(()) > 0.5 else -1.0)
    return p


def eval_plateau(x, p):
    z = (x - p["b"]) @ p["R"].T
    proj = z @ p["dirs"].T - p["shift"].view(1, 1, -1)
    sat = torch.tanh(p["sharp"] * proj)
    shelves = (p["amp"].view(1, 1, -1) * sat * sat).mean(dim=2)
    base = p["base"] * torch.sum(z * z, dim=2) / z.shape[-1]
    tilt = p["tilt"] * torch.tanh((z @ p["tilt_dir"]).squeeze(-1))
    return base + shelves + tilt


# ============================================================
# 7-9: curved valley 弯曲狭谷族
# ============================================================
def offset_valley_smooth(dim, device):
    p = _base_offset(dim, device)
    side_dim = max(dim - 1, 1)
    p.update({
        "dir_side": _unit_vec(side_dim, device),
        "freq": float(torch.empty(1).uniform_(0.35, 1.2).item()),
        "phase": float(torch.empty(1).uniform_(0, 2 * math.pi).item()),
        "curve": float(torch.empty(1).uniform_(-0.45, 0.45).item()),
        "ripple": float(torch.empty(1).uniform_(0.01, 0.12).item()),
        "w": torch.empty(2, device=device).uniform_(0.25, 1.0),
    })
    return p


def offset_valley_wavy(dim, device):
    p = offset_valley_smooth(dim, device)
    p["freq"] = float(torch.empty(1).uniform_(1.2, 2.8).item())
    p["ripple"] = float(torch.empty(1).uniform_(0.10, 0.35).item())
    return p


def offset_valley_narrow(dim, device):
    p = offset_valley_smooth(dim, device)
    p["w"] = torch.tensor([1.25, 0.45], device=device) * torch.empty(2, device=device).uniform_(0.7, 1.3)
    p["curve"] = float(torch.empty(1).uniform_(-0.65, 0.65).item())
    return p


def eval_valley(x, p):
    z = (x - p["b"]) @ p["R"].T
    u = z[..., :1]
    tail = z[..., 1:]
    along = 0.035 * u.squeeze(-1) * u.squeeze(-1)
    along = along + p["ripple"] * torch.sin(0.7 * p["freq"] * u.squeeze(-1) + p["phase"]) ** 2
    if tail.shape[-1] == 0:
        return p["w"][1] * along
    side = p["dir_side"].view(1, 1, -1)
    target = 0.35 * torch.sin(p["freq"] * u + p["phase"]) * side
    target = target + p["curve"] * (u * u / (1.0 + u * u)) * side
    valley = torch.mean((tail - target) ** 2, dim=2)
    return p["w"][0] * valley + p["w"][1] * along


# ============================================================
# 10-12: composition gate 平滑组合门控族
# ============================================================
def offset_comp_local(dim, device):
    p = _base_offset(dim, device)
    K = int(torch.randint(3, 6, (1,)).item())
    p.update({
        "centers": (torch.rand(K, dim, device=device) - 0.5) * 7.5,
        "lam": torch.empty(K, dim, device=device).uniform_(0.3, 1.8),
        "omega": torch.randn(K, dim, device=device) * torch.empty(K, 1, device=device).uniform_(0.2, 1.2),
        "phase": torch.rand(K, device=device) * 2 * math.pi,
        "amp": torch.empty(K, device=device).uniform_(0.02, 0.35),
        "gate": float(torch.empty(1).uniform_(0.35, 1.2).item()),
        "bias": torch.empty(K, device=device).uniform_(-0.3, 0.3),
    })
    return p


def offset_comp_global(dim, device):
    p = offset_comp_local(dim, device)
    p["gate"] = float(torch.empty(1).uniform_(0.08, 0.35).item())
    p["amp"] = torch.empty_like(p["amp"]).uniform_(0.15, 0.75)
    return p


def offset_comp_rugged(dim, device):
    p = offset_comp_local(dim, device)
    K = p["centers"].shape[0]
    p["omega"] = torch.randn(K, dim, device=device) * torch.empty(K, 1, device=device).uniform_(1.0, 3.2)
    p["amp"] = torch.empty(K, device=device).uniform_(0.25, 0.85)
    return p


def eval_comp(x, p):
    z = (x - p["b"]) @ p["R"].T
    diff = z.unsqueeze(2) - p["centers"].view(1, 1, p["centers"].shape[0], p["centers"].shape[1])
    d2 = torch.sum(diff * diff, dim=3)
    logits = -p["gate"] * d2 / z.shape[-1] + p["bias"].view(1, 1, -1)
    gate = torch.softmax(logits, dim=2)
    quad = torch.sum(p["lam"].view(1, 1, p["lam"].shape[0], p["lam"].shape[1]) * diff * diff, dim=3) / z.shape[-1]
    proj = torch.sum(z.unsqueeze(2) * p["omega"].view(1, 1, p["omega"].shape[0], p["omega"].shape[1]), dim=3) + p["phase"].view(1, 1, -1)
    local = quad + p["amp"].view(1, 1, -1) * torch.sin(proj) ** 2 + 0.08 * torch.tanh(proj)
    return torch.sum(gate * local, dim=2)


# ============================================================
# 13-15: random Fourier field 随机傅里叶场
# ============================================================
def offset_fourier_low(dim, device):
    p = _base_offset(dim, device)
    K = int(torch.randint(4, 9, (1,)).item())
    p.update({
        "omega": torch.randn(K, dim, device=device) * torch.empty(K, 1, device=device).uniform_(0.15, 0.8),
        "phase": torch.rand(K, device=device) * 2 * math.pi,
        "amp": torch.empty(K, device=device).uniform_(0.1, 0.8),
        "base": float(torch.empty(1).uniform_(0.01, 0.09).item()),
    })
    return p


def offset_fourier_mid(dim, device):
    p = offset_fourier_low(dim, device)
    K = p["omega"].shape[0]
    p["omega"] = torch.randn(K, dim, device=device) * torch.empty(K, 1, device=device).uniform_(0.7, 1.8)
    return p


def offset_fourier_mixed(dim, device):
    p = _base_offset(dim, device)
    K = int(torch.randint(8, 15, (1,)).item())
    scale = torch.cat([
        torch.empty(K // 2, 1, device=device).uniform_(0.15, 0.8),
        torch.empty(K - K // 2, 1, device=device).uniform_(1.2, 2.8),
    ], dim=0)
    p.update({
        "omega": torch.randn(K, dim, device=device) * scale,
        "phase": torch.rand(K, device=device) * 2 * math.pi,
        "amp": torch.empty(K, device=device).uniform_(0.05, 0.65),
        "base": float(torch.empty(1).uniform_(0.005, 0.06).item()),
    })
    return p


def eval_fourier(x, p):
    z = (x - p["b"]) @ p["R"].T
    proj = z @ p["omega"].T + p["phase"]
    wave = (p["amp"] * (torch.sin(proj) + 0.5 * torch.cos(0.7 * proj))).mean(dim=2)
    base = p["base"] * torch.sum(z * z, dim=2) / z.shape[-1]
    return base + wave


# ============================================================
# 16-18: low-rank coupling 低秩耦合族
# ============================================================
def offset_lowrank_soft(dim, device):
    p = _base_offset(dim, device)
    r = int(torch.randint(2, min(6, dim) + 1, (1,)).item())
    p.update({
        "A": torch.randn(dim, r, device=device) / math.sqrt(dim),
        "w": torch.empty(r, device=device).uniform_(0.3, 1.2),
        "base": float(torch.empty(1).uniform_(0.01, 0.08).item()),
        "nonlin": 0,
    })
    return p


def offset_lowrank_tanh(dim, device):
    p = offset_lowrank_soft(dim, device)
    p["nonlin"] = 1
    return p


def offset_lowrank_wave(dim, device):
    p = offset_lowrank_soft(dim, device)
    p["nonlin"] = 2
    p["freq"] = float(torch.empty(1).uniform_(0.8, 2.5).item())
    return p


def eval_lowrank(x, p):
    z = (x - p["b"]) @ p["R"].T
    u = z @ p["A"]
    if p["nonlin"] == 0:
        term = torch.sum(p["w"] * u * u, dim=2)
    elif p["nonlin"] == 1:
        term = torch.sum(p["w"] * torch.tanh(u) ** 2, dim=2)
    else:
        term = torch.sum(p["w"] * (torch.sin(p["freq"] * u) ** 2 + 0.15 * u * u), dim=2)
    base = p["base"] * torch.sum(z * z, dim=2) / z.shape[-1]
    return base + term / max(p["A"].shape[1], 1)


# ============================================================
# 19-21: ring / shell 环形盆地族
# ============================================================
def offset_ring_single(dim, device):
    p = _base_offset(dim, device)
    p.update({
        "radius": float(torch.empty(1).uniform_(1.0, 3.8).item()),
        "width": float(torch.empty(1).uniform_(0.25, 1.0).item()),
        "base": float(torch.empty(1).uniform_(0.005, 0.05).item()),
        "wave": float(torch.empty(1).uniform_(0.02, 0.20).item()),
        "freq": float(torch.empty(1).uniform_(0.8, 2.2).item()),
    })
    return p


def offset_ring_double(dim, device):
    p = offset_ring_single(dim, device)
    p["radius2"] = float(torch.empty(1).uniform_(2.0, 4.5).item())
    p["mode"] = 2
    return p


def offset_ring_wavy(dim, device):
    p = offset_ring_single(dim, device)
    p["wave"] = float(torch.empty(1).uniform_(0.15, 0.45).item())
    p["freq"] = float(torch.empty(1).uniform_(2.0, 4.2).item())
    return p


def eval_ring(x, p):
    z = (x - p["b"]) @ p["R"].T
    r = torch.sqrt(torch.sum(z * z, dim=2) / z.shape[-1] + 1e-9)
    v1 = ((r - p["radius"]) / p["width"]) ** 2
    if p.get("mode", 1) == 2:
        v2 = ((r - p["radius2"]) / p["width"]) ** 2 + 0.25
        val = torch.minimum(v1, v2)
    else:
        val = v1
    return val + p["base"] * torch.sum(z * z, dim=2) / z.shape[-1] + p["wave"] * torch.sin(p["freq"] * r) ** 2


# ============================================================
# 22-24: spiral / angular 螺旋角向族
# ============================================================
def offset_spiral_soft(dim, device):
    p = _base_offset(dim, device)
    p.update({
        "freq": float(torch.empty(1).uniform_(0.8, 2.0).item()),
        "twist": float(torch.empty(1).uniform_(0.3, 1.2).item()),
        "base": float(torch.empty(1).uniform_(0.02, 0.10).item()),
        "amp": float(torch.empty(1).uniform_(0.1, 0.5).item()),
    })
    return p


def offset_spiral_strong(dim, device):
    p = offset_spiral_soft(dim, device)
    p["twist"] = float(torch.empty(1).uniform_(1.2, 2.6).item())
    p["amp"] = float(torch.empty(1).uniform_(0.35, 0.9).item())
    return p


def offset_spiral_decay(dim, device):
    p = offset_spiral_soft(dim, device)
    p["decay"] = float(torch.empty(1).uniform_(0.03, 0.18).item())
    return p


def eval_spiral(x, p):
    z = (x - p["b"]) @ p["R"].T
    a = z[..., 0]
    b = z[..., 1] if z.shape[-1] > 1 else z[..., 0] * 0.0
    r = torch.sqrt(a * a + b * b + 1e-9)
    theta = torch.atan2(b, a + 1e-9)
    radial = p["base"] * torch.sum(z * z, dim=2) / z.shape[-1]
    phase = p["freq"] * r + p["twist"] * theta
    amp = p["amp"] * torch.exp(-p.get("decay", 0.0) * r)
    return radial + amp * torch.sin(phase) ** 2


# ============================================================
# 25-27: saddle / mixed curvature 鞍点混合族
# ============================================================
def offset_saddle_soft(dim, device):
    p = _base_offset(dim, device)
    sign = torch.ones(dim, device=device)
    sign[::2] = -1.0
    p.update({
        "lam": torch.empty(dim, device=device).uniform_(0.2, 1.5) * sign,
        "base": float(torch.empty(1).uniform_(0.05, 0.18).item()),
        "smooth": float(torch.empty(1).uniform_(0.4, 1.5).item()),
    })
    return p


def offset_saddle_wave(dim, device):
    p = offset_saddle_soft(dim, device)
    p["omega"] = torch.randn(5, dim, device=device) * torch.empty(5, 1, device=device).uniform_(0.3, 1.5)
    p["phase"] = torch.rand(5, device=device) * 2 * math.pi
    p["amp"] = torch.empty(5, device=device).uniform_(0.05, 0.35)
    return p


def offset_saddle_tilt(dim, device):
    p = offset_saddle_soft(dim, device)
    p["tilt_dir"] = _unit_vec(dim, device).view(dim, 1)
    p["tilt"] = float(torch.empty(1).uniform_(-0.25, 0.25).item())
    return p


def eval_saddle(x, p):
    z = (x - p["b"]) @ p["R"].T
    raw = torch.sum(p["lam"] * z * z, dim=2) / z.shape[-1]
    val = p["base"] * torch.sqrt(raw * raw + p["smooth"] ** 2)
    if "omega" in p:
        proj = z @ p["omega"].T + p["phase"]
        val = val + (p["amp"] * torch.sin(proj) ** 2).mean(dim=2)
    if "tilt_dir" in p:
        val = val + p["tilt"] * torch.tanh((z @ p["tilt_dir"]).squeeze(-1))
    return val


# ============================================================
# 28-30: multi-scale separable / nonseparable 混合尺度族
# ============================================================
def offset_multiscale_smooth(dim, device):
    p = _base_offset(dim, device)
    scale = 10.0 ** torch.empty(dim, device=device).uniform_(-1.0, 1.0)
    p.update({
        "scale": scale / scale.mean(),
        "freq": torch.empty(dim, device=device).uniform_(0.2, 1.2),
        "phase": torch.rand(dim, device=device) * 2 * math.pi,
        "amp": float(torch.empty(1).uniform_(0.05, 0.25).item()),
    })
    return p


def offset_multiscale_hard(dim, device):
    p = offset_multiscale_smooth(dim, device)
    scale = 10.0 ** torch.empty(dim, device=device).uniform_(-1.2, 1.2)
    p["scale"] = scale / scale.mean()
    p["amp"] = float(torch.empty(1).uniform_(0.20, 0.65).item())
    p["A"] = torch.randn(dim, max(2, min(5, dim)), device=device) / math.sqrt(dim)
    return p


def offset_multiscale_coupled(dim, device):
    p = offset_multiscale_hard(dim, device)
    p["A"] = torch.randn(dim, max(2, min(5, dim)), device=device) / math.sqrt(dim)
    return p


def eval_multiscale(x, p):
    z = (x - p["b"]) @ p["R"].T
    val = torch.sum(p["scale"] * z * z, dim=2) / z.shape[-1]
    val = val + p["amp"] * torch.mean(torch.sin(p["freq"] * z + p["phase"]) ** 2, dim=2)
    if "A" in p:
        u = z @ p["A"]
        val = val + 0.2 * torch.mean(torch.tanh(u) ** 2, dim=2)
    return val


# ============================================================
# 31-33: directional ridge 方向脊族
# ============================================================
def offset_dirridge_single(dim, device):
    p = _base_offset(dim, device)
    p.update({
        "d1": _unit_vec(dim, device).view(dim, 1),
        "w": torch.empty(3, device=device).uniform_(0.2, 1.2),
        "freq": float(torch.empty(1).uniform_(0.5, 2.5).item()),
        "noise_amp": float(torch.empty(1).uniform_(0.03, 0.12).item()),
        "noise_w": torch.randn(5, dim, device=device) * torch.empty(5, 1, device=device).uniform_(0.8, 2.0),
        "noise_p": torch.rand(5, device=device) * 2 * math.pi,
    })
    return p


def offset_dirridge_multi(dim, device):
    p = _base_offset(dim, device)
    K = int(torch.randint(2, 5, (1,)).item())
    dirs = torch.randn(dim, K, device=device)
    dirs = dirs / (torch.linalg.norm(dirs, dim=0, keepdim=True) + 1e-12)
    p.update({
        "dirs": dirs,
        "w": torch.empty(K, device=device).uniform_(0.2, 1.4),
        "freq": torch.empty(K, device=device).uniform_(0.5, 2.2),
        "base": float(torch.empty(1).uniform_(0.02, 0.10).item()),
    })
    return p


def offset_dirridge_noisy(dim, device):
    p = offset_dirridge_multi(dim, device)
    p["noise_amp"] = float(torch.empty(1).uniform_(0.04, 0.18).item())
    p["noise_w"] = torch.randn(6, dim, device=device) * torch.empty(6, 1, device=device).uniform_(1.0, 2.5)
    p["noise_p"] = torch.rand(6, device=device) * 2 * math.pi
    return p


def eval_dirridge(x, p):
    z = (x - p["b"]) @ p["R"].T
    if "d1" in p:
        proj = (z @ p["d1"]).squeeze(-1)
        total = torch.sum(z * z, dim=2)
        rest = torch.clamp(total - proj * proj, min=0.0)
        val = p["w"][0] * proj * proj + p["w"][1] * rest / z.shape[-1] + p["w"][2] * torch.sin(p["freq"] * proj) ** 2
    else:
        proj = z @ p["dirs"]
        val = torch.mean(p["w"] * torch.sin(p["freq"] * proj) ** 2 + 0.15 * proj * proj, dim=2)
        val = val + p["base"] * torch.sum(z * z, dim=2) / z.shape[-1]
    if "noise_amp" in p:
        val = val + p["noise_amp"] * torch.mean(torch.sin(z @ p["noise_w"].T + p["noise_p"]) ** 2, dim=2)
    return val


# ============================================================
# 34-36: wave-packet / envelope 波包包络族
# ============================================================
def offset_wavepacket_local(dim, device):
    p = _base_offset(dim, device)
    K = int(torch.randint(3, 7, (1,)).item())
    p.update({
        "center": (torch.rand(dim, device=device) - 0.5) * 5.0,
        "width": float(torch.empty(1).uniform_(1.0, 3.5).item()),
        "omega": torch.randn(K, dim, device=device) * torch.empty(K, 1, device=device).uniform_(0.5, 2.0),
        "phase": torch.rand(K, device=device) * 2 * math.pi,
        "amp": torch.empty(K, device=device).uniform_(0.1, 0.8),
        "base": float(torch.empty(1).uniform_(0.01, 0.08).item()),
    })
    return p


def offset_wavepacket_multi(dim, device):
    p = _base_offset(dim, device)
    C = int(torch.randint(2, 5, (1,)).item())
    K = int(torch.randint(4, 8, (1,)).item())
    p.update({
        "centers": (torch.rand(C, dim, device=device) - 0.5) * 7.0,
        "width": torch.empty(C, device=device).uniform_(1.4, 3.8),
        "omega": torch.randn(K, dim, device=device) * torch.empty(K, 1, device=device).uniform_(0.5, 2.4),
        "phase": torch.rand(K, device=device) * 2 * math.pi,
        "amp": torch.empty(K, device=device).uniform_(0.25, 1.0),
        "base": float(torch.empty(1).uniform_(0.02, 0.09).item()),
        "multi": True,
    })
    return p


def offset_wavepacket_drift(dim, device):
    p = offset_wavepacket_local(dim, device)
    p["drift"] = _unit_vec(dim, device).view(dim, 1)
    p["drift_amp"] = float(torch.empty(1).uniform_(-0.18, 0.18).item())
    return p


def eval_wavepacket(x, p):
    z = (x - p["b"]) @ p["R"].T
    if p.get("multi", False):
        diff = z.unsqueeze(2) - p["centers"].view(1, 1, p["centers"].shape[0], p["centers"].shape[1])
        d2 = torch.sum(diff * diff, dim=3)
        env = torch.exp(-d2 / (2.0 * p["width"].view(1, 1, -1) ** 2 + 1e-9)).sum(dim=2)
    else:
        d2 = torch.sum((z - p["center"]) ** 2, dim=2)
        env = torch.exp(-d2 / (2.0 * p["width"] ** 2 + 1e-9))
    proj = z @ p["omega"].T + p["phase"]
    wave = torch.mean(p["amp"] * torch.sin(proj) ** 2, dim=2)
    val = p["base"] * torch.sum(z * z, dim=2) / z.shape[-1] + env * wave
    if "drift" in p:
        val = val + p["drift_amp"] * torch.tanh((z @ p["drift"]).squeeze(-1))
    return val


TRAIN_FUNCTIONS = [
    {"fid": "rbf_sparse", "fun": eval_rbf, "gen_offset": offset_rbf_sparse, "xlb": -5.0, "xub": 5.0},
    {"fid": "rbf_dense", "fun": eval_rbf, "gen_offset": offset_rbf_dense, "xlb": -5.0, "xub": 5.0},
    {"fid": "rbf_deceptive", "fun": eval_rbf, "gen_offset": offset_rbf_deceptive, "xlb": -5.0, "xub": 5.0},

    {"fid": "plateau_soft", "fun": eval_plateau, "gen_offset": offset_plateau_soft, "xlb": -5.0, "xub": 5.0},
    {"fid": "plateau_sharp", "fun": eval_plateau, "gen_offset": offset_plateau_sharp, "xlb": -5.0, "xub": 5.0},
    {"fid": "plateau_tilted", "fun": eval_plateau, "gen_offset": offset_plateau_tilted, "xlb": -5.0, "xub": 5.0},

    {"fid": "valley_smooth", "fun": eval_valley, "gen_offset": offset_valley_smooth, "xlb": -5.0, "xub": 5.0},
    {"fid": "valley_wavy", "fun": eval_valley, "gen_offset": offset_valley_wavy, "xlb": -5.0, "xub": 5.0},
    {"fid": "valley_narrow", "fun": eval_valley, "gen_offset": offset_valley_narrow, "xlb": -5.0, "xub": 5.0},

    {"fid": "comp_local", "fun": eval_comp, "gen_offset": offset_comp_local, "xlb": -5.0, "xub": 5.0},
    {"fid": "comp_global", "fun": eval_comp, "gen_offset": offset_comp_global, "xlb": -5.0, "xub": 5.0},
    {"fid": "comp_rugged", "fun": eval_comp, "gen_offset": offset_comp_rugged, "xlb": -5.0, "xub": 5.0},

    {"fid": "fourier_low", "fun": eval_fourier, "gen_offset": offset_fourier_low, "xlb": -5.0, "xub": 5.0},
    {"fid": "fourier_mid", "fun": eval_fourier, "gen_offset": offset_fourier_mid, "xlb": -5.0, "xub": 5.0},
    {"fid": "fourier_mixed", "fun": eval_fourier, "gen_offset": offset_fourier_mixed, "xlb": -5.0, "xub": 5.0},

    {"fid": "lowrank_soft", "fun": eval_lowrank, "gen_offset": offset_lowrank_soft, "xlb": -5.0, "xub": 5.0},
    {"fid": "lowrank_tanh", "fun": eval_lowrank, "gen_offset": offset_lowrank_tanh, "xlb": -5.0, "xub": 5.0},
    {"fid": "lowrank_wave", "fun": eval_lowrank, "gen_offset": offset_lowrank_wave, "xlb": -5.0, "xub": 5.0},

    {"fid": "ring_single", "fun": eval_ring, "gen_offset": offset_ring_single, "xlb": -5.0, "xub": 5.0},
    {"fid": "ring_double", "fun": eval_ring, "gen_offset": offset_ring_double, "xlb": -5.0, "xub": 5.0},
    {"fid": "ring_wavy", "fun": eval_ring, "gen_offset": offset_ring_wavy, "xlb": -5.0, "xub": 5.0},

    {"fid": "spiral_soft", "fun": eval_spiral, "gen_offset": offset_spiral_soft, "xlb": -5.0, "xub": 5.0},
    {"fid": "spiral_strong", "fun": eval_spiral, "gen_offset": offset_spiral_strong, "xlb": -5.0, "xub": 5.0},
    {"fid": "spiral_decay", "fun": eval_spiral, "gen_offset": offset_spiral_decay, "xlb": -5.0, "xub": 5.0},

    {"fid": "saddle_soft", "fun": eval_saddle, "gen_offset": offset_saddle_soft, "xlb": -5.0, "xub": 5.0},
    {"fid": "saddle_wave", "fun": eval_saddle, "gen_offset": offset_saddle_wave, "xlb": -5.0, "xub": 5.0},
    {"fid": "saddle_tilt", "fun": eval_saddle, "gen_offset": offset_saddle_tilt, "xlb": -5.0, "xub": 5.0},

    {"fid": "multiscale_smooth", "fun": eval_multiscale, "gen_offset": offset_multiscale_smooth, "xlb": -5.0, "xub": 5.0},
    {"fid": "multiscale_hard", "fun": eval_multiscale, "gen_offset": offset_multiscale_hard, "xlb": -5.0, "xub": 5.0},
    {"fid": "multiscale_coupled", "fun": eval_multiscale, "gen_offset": offset_multiscale_coupled, "xlb": -5.0, "xub": 5.0},

    {"fid": "dirridge_single", "fun": eval_dirridge, "gen_offset": offset_dirridge_single, "xlb": -5.0, "xub": 5.0},
    {"fid": "dirridge_multi", "fun": eval_dirridge, "gen_offset": offset_dirridge_multi, "xlb": -5.0, "xub": 5.0},
    {"fid": "dirridge_noisy", "fun": eval_dirridge, "gen_offset": offset_dirridge_noisy, "xlb": -5.0, "xub": 5.0},

    {"fid": "wavepacket_local", "fun": eval_wavepacket, "gen_offset": offset_wavepacket_local, "xlb": -5.0, "xub": 5.0},
    {"fid": "wavepacket_multi", "fun": eval_wavepacket, "gen_offset": offset_wavepacket_multi, "xlb": -5.0, "xub": 5.0},
    {"fid": "wavepacket_drift", "fun": eval_wavepacket, "gen_offset": offset_wavepacket_drift, "xlb": -5.0, "xub": 5.0},
]


def gen_train_offset(dim, fun):
    fun["params"] = fun["gen_offset"](dim, DEVICE)


def _params_device(params):
    for _v in params.values():
        if torch.is_tensor(_v):
            return _v.device
    return None


def get_train_fitness(x, fun):
    # 关键修复：训练/测试时 x 可能在 CPU 或 CUDA，params 必须跟 x 在同一设备。
    if "params" not in fun:
        fun["params"] = fun["gen_offset"](x.shape[-1], x.device)
    else:
        _dev = _params_device(fun["params"])
        if _dev is not None and _dev != x.device:
            fun["params"] = fun["gen_offset"](x.shape[-1], x.device)
    y = fun["fun"](x, fun["params"])
    return torch.nan_to_num(y, nan=0.0, posinf=1e20, neginf=-1e20)
