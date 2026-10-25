"""One-time generation of reviewable strict ablation sources from frozen R10.

Generated Python is checked in and frozen before live evaluation. No runtime
source rewriting is used. The unchanged population/archive updates are copied.
"""
import ast
import hashlib
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]


def method(path, cls, name):
    source=(ROOT/path).read_text();tree=ast.parse(source)
    owner=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==cls)
    node=next(node for node in owner.body if isinstance(node,ast.FunctionDef) and node.name==name)
    return textwrap.dedent('\n'.join(source.splitlines()[node.lineno-1:node.end_lineno]))


def main():
    header=(ROOT/'scripts/r12_model_header.txt').read_text()
    forward=method('roopf/model.py','ROOPFOptimizer','forward')
    start=forward.index('        pool, op_ids, op_prior = self._candidate_pool(')
    stop=forward.index('        fits = []',start)
    forward=forward[:start]+'''        # Preserve the original stream at every objective boundary, without
        # executing any removed generator or creating dummy candidates.
        iteration=(self.evalnum-self.popsize)//self.k_nums
        torch.set_rng_state(self.objective_rng_schedule[1+2*iteration])
        op_prior=x.new_zeros((x.size(0),self.num_ops*self.pool_per_op))
        cand,selected_ops,cand_score,cand_mu,cand_sigma,cand_router_prob = self._anchor_round(
            x,fitness,problem,rest,archive_x,archive_y,state_raw,remaining,stagnation,used,
            portfolio_seen,portfolio_best_count,portfolio_admit_count,portfolio_no_admit)
'''+forward[stop:]
    numerical=method('roopf/current_evidence_agent.py','CurrentEvidenceAgent','numerical_options')
    numerical=numerical.replace('range(38)','range(points.size(1))')
    numerical=numerical.replace("        if float(self.local_scale[i]) != 1:\n            supported[i, [22, 23, 24, 28, 29, 30]] = 0\n",'')
    numerical=numerical.replace('(q[1] - q[0])','(q[1] - q[0] if len(q)>1 else 0.)')
    numerical=numerical.replace('original_selection=int(kwargs', 'candidate_count=points.size(1), original_selection=int(kwargs')
    numeric_choose=method('roopf/evidence_fusion_agent.py','EvidenceFusionAgent','choose')
    numeric_choose=numeric_choose.replace('def choose(', 'def _numeric_choose(',1)
    numeric_choose=numeric_choose.replace("    if points.size(1) != 38:\n        raise RuntimeError('Joint controller requires38 competing candidates')\n",'')
    numeric_choose=numeric_choose.replace('candidate_count=38','candidate_count=points.size(1)')
    numeric_choose=numeric_choose.replace("slot == 37 and self.extra_origin[index] == 'anchor_variant'","slot == points.size(1)-1 and self.extra_origin[index] == 'anchor_variant'")
    choose=method('roopf/current_evidence_agent.py','CurrentEvidenceAgent','choose')
    choose=choose.replace('super().choose(**kwargs)','self._numeric_choose(**kwargs)')
    assert 'range(38)' not in numerical and 'supported[i, [22' not in numerical
    assert 'requires38' not in numeric_choose and 'candidate_count=38' not in numeric_choose
    target=ROOT/'roopf/strict_component_ablations.py'
    model_header,agent_header=header.split('# AGENT_CLASS_START\n')
    source=model_header+'\n'+textwrap.indent(forward,'    ')+'\n\n'+agent_header
    for body in (numerical,numeric_choose,choose):source+='\n'+textwrap.indent(body,'    ')+'\n'
    ast.parse(source);target.write_text(source)
    print('Generated',target.relative_to(ROOT),'sha256',hashlib.sha256(source.encode()).hexdigest())


if __name__=='__main__':main()
