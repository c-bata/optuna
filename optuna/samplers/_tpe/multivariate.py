import math
from typing import Any
from typing import Callable
from typing import Container
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import warnings

import numpy as np

from optuna._hypervolume import WFG
from optuna.distributions import BaseDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.logging import get_logger
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _process_constraints_after_trial
from optuna.samplers._base import BaseSampler
from optuna.samplers._random import RandomSampler
from optuna.samplers._search_space import IntersectionSearchSpace
from optuna.samplers._search_space.group_decomposed import _GroupDecomposedSearchSpace
from optuna.samplers._search_space.group_decomposed import _SearchSpaceGroup
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


EPS = 1e-12
_logger = get_logger(__name__)


def _get_observation_pairs_multivariate(
    study: Study,
    param_names: List[str],
    multivariate: bool,
    constant_liar: bool = False,  # TODO(hvy): Remove default value and fix unit tests.
    constraints_enabled: bool = False,
) -> Tuple[
    Dict[str, List[Optional[float]]],
    List[Tuple[float, List[float]]],
    Optional[List[float]],
]:
    """Get observation pairs from the study.

    This function collects observation pairs from the complete or pruned trials of the study.
    In addition, if ``constant_liar`` is :obj:`True`, the running trials are considered.
    The values for trials that don't contain the parameter in the ``param_names`` are skipped.

    An observation pair fundamentally consists of a parameter value and an objective value.
    However, due to the pruning mechanism of Optuna, final objective values are not always
    available. Therefore, this function uses intermediate values in addition to the final
    ones, and reports the value with its step count as ``(-step, value)``.
    Consequently, the structure of the observation pair is as follows:
    ``(param_value, (-step, value))``.

    The second element of an observation pair is used to rank observations in
    ``_split_observation_pairs`` method (i.e., observations are sorted lexicographically by
    ``(-step, value)``).

    When ``constraints_enabled`` is :obj:`True`, 1-dimensional violation values are returned
    as the third element (:obj:`None` otherwise). Each value is a float of 0 or greater and a
    trial is feasible if and only if its violation score is 0.
    """

    assert len(param_names) > 1
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
    values: Dict[str, List[Optional[float]]] = {param_name: [] for param_name in param_names}
    violations: Optional[List[float]] = [] if constraints_enabled else None
    for trial in study.get_trials(deepcopy=False, states=states):
        # If ``multivariate`` = True and ``group`` = True, we ignore the trials that are not
        # included in each subspace.
        # If ``multivariate`` = False, we skip the check.
        if any([param_name not in trial.params for param_name in param_names]):
            continue

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

        # We extract param_value from the trial.
        for param_name in param_names:
            param_value: Optional[float]
            if param_name in trial.params:
                distribution = trial.distributions[param_name]
                param_value = distribution.to_internal_repr(trial.params[param_name])
            else:
                param_value = None
            values[param_name].append(param_value)

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
