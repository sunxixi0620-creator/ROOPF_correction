"""原发布版六臂复现；保留原件，只在构造前恢复已核查的历史消融开关。"""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
import run_roopf as release

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('full', 'no_proxy', 'no_residual', 'historical_no_safety', 'anchor_only', 'online_only')
MODEL_SHA = '5312962f3864557e646a5532d88c76cc822616f49cf242dd04a13b151c648908'
RESIDUAL = {'learned_router', 'protected_residual_selector', 'residual_safety_gate',
            'pool_rank_router', 'router_veto', 'weak_router_veto', 'highdim_residual_router'}
ANCHOR = {'baseline_guard', 'baseline_backbone', 'baseline_safe_selector', 'structured_system_lock'}


def intervention(arm):
    """历史no_safety确实移除anchor保护路径，不能将它误报为纯门控干预。"""
    assert arm in ARMS
    if arm == 'no_residual': return RESIDUAL
    if arm == 'historical_no_safety': return ANCHOR
    if arm == 'online_only': return ANCHOR | RESIDUAL
    return set()


def build(arm):
    """full直接调用发布入口；完全去掉离线的组在加载权重前移除全部离线路径。"""
    assert arm in ARMS and release.DEVICE.type == 'cpu'
    args = SimpleNamespace(dimension=10, population_size=100, budget=300)
    if arm == 'full': return release.build_optimizer(args)
    if arm == 'anchor_only':
        from roopf.anchor_backbone import AnchorPolicyBackbone
        model = AnchorPolicyBackbone(dim=10, hidden_dim=200, popSize=100)
        model.load_state_dict(torch.load(ROOT/'checkpoints/anchor_policy_d10.pt', weights_only=True, map_location='cpu'), strict=True)
        return model.eval()
    source = (ROOT/'roopf/model.py').read_bytes()
    assert hashlib.sha256(source).hexdigest() == MODEL_SHA
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ROOPFOptimizer')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    slots = [i for i,n in enumerate(init.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='self.ablation.update']
    assert len(slots) == 1
    # 只插入可审计的集合干预，不改候选、评分、反馈及预算循环。
    removed = sorted(intervention(arm))
    init.body.insert(slots[0]+1, ast.parse('self.ablation.difference_update('+repr(removed)+')').body[0])
    namespace = {'__name__':'roopf.zip_replication820'}
    exec(compile(ast.fix_missing_locations(tree), '<zip_replication820>', 'exec'),namespace)
    model = namespace['ROOPFOptimizer'](dim=10, hidden_dim=200, popSize=100, max_nfe=300,
        k_nums=2, pool_per_op=6, surrogate_members=5,
        ablation='roopf,no_surrogate' if arm=='no_proxy' else 'roopf', router_weight=.008,
        baseline_ckpt=None if arm in ('online_only','historical_no_safety') else str(ROOT/'checkpoints/anchor_policy_d10.pt'),
        router_ckpt=None if arm in ('online_only','no_residual') else str(ROOT/'checkpoints/residual_selector_generated36_d10.pt'))
    if arm=='online_only': assert model.baseline_model is None and model.router_model is None
    if arm=='no_residual': assert model.router_model is None and model.baseline_model is not None
    return model.eval()


def problem(group, fid):
    """BBOB和CEC严格沿用发布入口；UAV复用有来源记录的外部任务实现。"""
    if group=='BBOB':
        fun=dict(release.BBOB_FUNCTIONS[fid], xlb=-5, xub=5)
        release.setOffset(fun,release.load_bbob_offsets(ROOT/'data/bbob_offsets_d10.pkl')[fid])
        return release.BBOBProblem(fun,10)
    if group=='CEC-style':
        fun=dict(release.CEC_FUNCTIONS[f'cecf{fid}'])
        fun['xlb']=fun.get('xlb',-100);fun['xub']=fun.get('xub',100)
        return release.CECStyleProblem(fun,10)
    assert group=='UAV'
    import importlib.util
    spec=importlib.util.spec_from_file_location('zip820_uav',ROOT/'artifacts/recovered_sources/uav_benchmark.py')
    module=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    p=release.BBOBProblem(dict(fid=f'uav{fid}',xlb=-5,xub=5),10)
    p.calfitness=lambda x:module.uav_objective(x,module.SCENARIOS[fid-1])
    return p
