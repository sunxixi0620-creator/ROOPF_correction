"""Isolated execution policy for the R13 CEC-only remeasurement.

Frozen source files are not rewritten. Each experiment worker installs this
policy on the current model and strict-ablation modules only. The independently
loaded archived baseline keeps its own structural predicate and CPU execution.
Problem names, implementations, bounds, weights, cases and gates are unchanged.
"""
from __future__ import annotations

POLICY_ID = 'r13_cec_same_execution_path_as_bbob_v1'


def install_execution_policy():
    import roopf.model as model
    import roopf.strict_component_ablations as strict
    if not hasattr(model, '_r13_frozen_structured_predicate'):
        model._r13_frozen_structured_predicate = model._is_structured_system_problem
    frozen = model._r13_frozen_structured_predicate

    def uniform_predicate(problem):
        name = str(problem.getfunname()).lower() if hasattr(problem, 'getfunname') else ''
        if name.startswith('cecf'):
            return False
        return frozen(problem)

    model._is_structured_system_problem = uniform_predicate
    strict._is_structured_system_problem = uniform_predicate
    return dict(policy_id=POLICY_ID, patched_modules=['roopf.model','roopf.strict_component_ablations'],
                problem_names_preserved=True, objective_implementation_preserved=True,
                baseline_module_patched=False,
                unified_routes=['generation_activation','structured_selector_lock','agent_full_pool',
                                'residual_eligibility','bbob_guarded_proxy','strict_ablation_routes'])
