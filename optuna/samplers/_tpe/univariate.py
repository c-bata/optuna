import math
from typing import Any
from typing import Callable
from typing import Container
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import warnings

from optuna.logging import get_logger
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import TrialState


EPS = 1e-12
_logger = get_logger(__name__)


def _get_observation_pairs_univariate(
    study: Study,
    param_name: str,
    constant_liar: bool = False,  # TODO(hvy): Remove default value and fix unit tests.
) -> Tuple[List[Optional[float]], List[Tuple[float, List[float]]],]:
    signs = []
    for d in study.directions:
        if d == StudyDirection.MINIMIZE:
            signs.append(1)
        else:
            signs.append(-1)

    states = [TrialState.COMPLETE, TrialState.PRUNED]
    if constant_liar:
        states.append(TrialState.RUNNING)

    scores = []
    values: List[Optional[float]] = []
    for trial in study.get_trials(deepcopy=False, states=states):
        # We extract score from the trial.
        if trial.state is TrialState.COMPLETE:
            if trial.values is None:
                continue
            score = (-float("inf"), [sign * v for sign, v in zip(signs, trial.values)])
        elif trial.state is TrialState.PRUNED:
            if study._is_multi_objective():
                continue

            if len(trial.intermediate_values) > 0:
                step, intermediate_value = max(trial.intermediate_values.items())
                if math.isnan(intermediate_value):
                    score = (-step, [float("inf")])
                else:
                    score = (-step, [signs[0] * intermediate_value])
            else:
                score = (float("inf"), [0.0])
        elif trial.state is TrialState.RUNNING:
            if study._is_multi_objective():
                continue

            assert constant_liar
            score = (-float("inf"), [signs[0] * float("inf")])
        else:
            assert False
        scores.append(score)

        param_value: Optional[float]
        if param_name in trial.params:
            distribution = trial.distributions[param_name]
            param_value = distribution.to_internal_repr(trial.params[param_name])
        else:
            param_value = None
        values.append(param_value)
    return values, scores


def _get_observation_pairs_univariate_constraints(
    study: Study,
    param_name: str,
    constant_liar: bool = False,  # TODO(hvy): Remove default value and fix unit tests.
    constraints_enabled: bool = False,
) -> Tuple[List[Optional[float]], List[Tuple[float, List[float]]], Optional[List[float]],]:
    signs = []
    for d in study.directions:
        if d == StudyDirection.MINIMIZE:
            signs.append(1)
        else:
            signs.append(-1)

    states: Container[TrialState]
    if constant_liar:
        states = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING)
    else:
        states = (TrialState.COMPLETE, TrialState.PRUNED)

    scores = []
    values: List[Optional[float]] = []
    violations: Optional[List[float]] = [] if constraints_enabled else None
    for trial in study.get_trials(deepcopy=False, states=states):
        # We extract score from the trial.
        if trial.state is TrialState.COMPLETE:
            if trial.values is None:
                continue
            score = (-float("inf"), [sign * v for sign, v in zip(signs, trial.values)])
        elif trial.state is TrialState.PRUNED:
            if study._is_multi_objective():
                continue

            if len(trial.intermediate_values) > 0:
                step, intermediate_value = max(trial.intermediate_values.items())
                if math.isnan(intermediate_value):
                    score = (-step, [float("inf")])
                else:
                    score = (-step, [signs[0] * intermediate_value])
            else:
                score = (float("inf"), [0.0])
        elif trial.state is TrialState.RUNNING:
            if study._is_multi_objective():
                continue

            assert constant_liar
            score = (-float("inf"), [signs[0] * float("inf")])
        else:
            assert False
        scores.append(score)

        param_value: Optional[float]
        if param_name in trial.params:
            distribution = trial.distributions[param_name]
            param_value = distribution.to_internal_repr(trial.params[param_name])
        else:
            param_value = None
        values.append(param_value)

        if constraints_enabled:
            assert violations is not None
            constraint = trial.system_attrs.get(_CONSTRAINTS_KEY)
            if constraint is None:
                warnings.warn(
                    f"Trial {trial.number} does not have constraint values."
                    " It will be treated as a lower priority than other trials."
                )
                violation = float("inf")
            else:
                # Violation values of infeasible dimensions are summed up.
                violation = sum(v for v in constraint if v > 0)
            violations.append(violation)

    return values, scores, violations
