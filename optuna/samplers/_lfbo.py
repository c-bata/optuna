from __future__ import annotations

import copy
from typing import Any
from typing import Literal
from typing import Optional
import warnings

import numpy as np

import optuna
from optuna._imports import try_import
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


with try_import() as _imports:
    # TODO(c-bata): Remove ConfigSpace dependency.
    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as csh
    from scipy.optimize import Bounds
    from scipy.optimize import OptimizeResult
    from scipy.stats import truncnorm
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils import check_random_state


class LfboSampler(BaseSampler):
    """A sampler based on `LFBO (Likelihood-free Bayesian Optimization) <https://arxiv.org/abs/2206.13035>`_.

    The implementation were derived from https://github.com/lfbo-ml/lfbo, an official LFBO implementation.
    """

    def __init__(
        self,
        *,
        num_random_init: int = 10,
        consider_pruned_trials: bool = False,
        seed: Optional[int] = None,
        gamma: float = 0.33,
        weight_type: Literal["ei", "pi"] = "ei",
        nasbench: bool = False,
    ):
        _imports.check()
        self._independent_sampler = optuna.samplers.RandomSampler(seed=seed)
        self._lfbo: Optional[LFBO] = None
        self._num_random_init = num_random_init
        self._gamma = gamma
        self._weight_type = weight_type
        self._nasbench = nasbench
        self._consider_pruned_trials = consider_pruned_trials
        self._search_space = optuna.samplers.IntersectionSearchSpace()
        self._rng = np.random.RandomState(seed)

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()
        self._rng.seed()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        search_space: dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution
        return search_space

    def _sample(
        self, study: Study, seed: int, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        observation_pairs = self._get_observation_pairs(study, list(search_space.keys()))
        if len(observation_pairs) < self._num_random_init:
            return {}

        config_space = search_space_to_config_space(search_space)
        lfbo = LFBO(
            config_space=config_space,
            num_random_init=self._num_random_init,
            seed=seed,
            gamma=self._gamma,
            weight_type=self._weight_type,
            nasbench=self._nasbench,
        )
        for config_dict, value in observation_pairs:
            lfbo.add_new_observation(config_dict, value)
        return lfbo.step()

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if len(search_space) == 0:
            return {}
        seed = self._rng.randint(1, 2**16) + trial.number
        params = self._sample(study, seed, search_space)
        return params

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        seed = self._rng.randint(1, 2**16) + trial.number
        search_space = {param_name: param_distribution}
        params = self._sample(study, seed, search_space)

        if not params:
            return self._independent_sampler.sample_independent(
                study,
                trial,
                param_name,
                param_distribution,
            )

        return params

    def _get_trials(self, study: Study) -> list[FrozenTrial]:
        complete_trials = []
        for t in study.get_trials(deepcopy=False):
            if t.state == TrialState.COMPLETE:
                complete_trials.append(t)
            elif (
                t.state == TrialState.PRUNED
                and len(t.intermediate_values) > 0
                and self._consider_pruned_trials
            ):
                _, value = max(t.intermediate_values.items())
                if value is None:
                    continue
                # We rewrite the value of the trial `t` for sampling, so we need a deepcopy.
                copied_t = copy.deepcopy(t)
                copied_t.value = value
                complete_trials.append(copied_t)
        return complete_trials

    def _get_observation_pairs(
        self,
        study: Study,
        target_params: list[str],
    ) -> list[tuple[dict[str, Any], float]]:
        observation_pairs = []
        for trial in self._get_trials(study):
            if not all(t in trial.params for t in target_params):
                continue
            params = {name: trial.params[name] for name in target_params}
            value = trial.value
            assert value is not None
            if study.direction == StudyDirection.MAXIMIZE:
                value *= -1
            observation_pairs.append((params, value))
        return observation_pairs


def search_space_to_config_space(
    search_space: dict[str, BaseDistribution]
) -> CS.ConfigurationSpace:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        cs = CS.ConfigurationSpace()
        for name, distribution in search_space.items():
            cs_hp = _distribution_to_config_space(name, distribution)
            cs.add_hyperparameter(cs_hp)
    return cs


def _distribution_to_config_space(name: str, distribution: BaseDistribution) -> csh.Hyperparameter:
    if isinstance(distribution, IntDistribution):
        return csh.UniformIntegerHyperparameter(
            name, lower=distribution.low, upper=distribution.high, log=distribution.log
        )
    if isinstance(distribution, FloatDistribution):
        return csh.UniformFloatHyperparameter(
            name, lower=distribution.low, upper=distribution.high, log=distribution.log
        )
    if isinstance(distribution, CategoricalDistribution):
        return csh.CategoricalHyperparameter(name, choices=distribution.choices)


def ceil_divide(a, b, *args, **kwargs):
    return -np.floor_divide(-a, b, *args, **kwargs)


def steps_per_epoch(dataset_size, batch_size):
    return int(ceil_divide(dataset_size, batch_size))


def truncated_normal(loc, scale, lower, upper):
    a = (lower - loc) / scale
    b = (upper - loc) / scale
    return truncnorm(a=a, b=b, loc=loc, scale=scale)


def maybe_distort(loc, distortion=None, bounds=None, random_state=None, print_fn=print):
    if distortion is None:
        return loc

    assert bounds is not None, "must specify bounds!"
    ret = truncated_normal(loc=loc, scale=distortion, lower=bounds.lb, upper=bounds.ub).rvs(
        random_state=random_state
    )
    print_fn(f"Suggesting x={ret} (after applying distortion={distortion:.3E})")

    return ret


class DenseConfigurationSpace(CS.ConfigurationSpace):
    def __init__(self, other, *args, **kwargs):

        super(DenseConfigurationSpace, self).__init__(*args, **kwargs)
        # deep-copy only the hyperparameters. conditions, clauses, seed,
        # and other metadata ignored
        self.add_hyperparameters(other.get_hyperparameters())

        nums, cats, size_sparse, size_dense = self._get_mappings()

        if nums:
            self.num_src, self.num_trg = map(np.uintp, zip(*nums))

        if cats:
            self.cat_src, self.cat_trg, self.cat_sizes = map(np.uintp, zip(*cats))

        self.nums = nums
        self.cats = cats
        self.size_sparse = size_sparse
        self.size_dense = size_dense

    def get_dimensions(self, sparse=False):
        return self.size_sparse if sparse else self.size_dense

    def sample_configuration(self, size=1):

        config_sparse = super(DenseConfigurationSpace, self).sample_configuration(size=size)

        configs_sparse_list = config_sparse if size > 1 else [config_sparse]

        configs = []
        for config in configs_sparse_list:
            configs.append(DenseConfiguration(self, values=config.get_dictionary()))

        return configs if size > 1 else configs.pop()

    def get_bounds(self):
        lowers = np.zeros(self.size_dense)
        uppers = np.ones(self.size_dense)

        # return list(zip(lowers, uppers))
        return Bounds(lowers, uppers)

    def _get_mappings(self):

        nums = []
        cats = []

        src_ind = trg_ind = 0
        for src_ind, hp in enumerate(self.get_hyperparameters()):
            if isinstance(hp, CS.CategoricalHyperparameter):
                cat_size = hp.num_choices
                cats.append((src_ind, trg_ind, cat_size))
                trg_ind += cat_size
            elif isinstance(hp, (CS.UniformIntegerHyperparameter, CS.UniformFloatHyperparameter)):
                nums.append((src_ind, trg_ind))
                trg_ind += 1
            else:
                raise NotImplementedError(
                    "Only hyperparameters of types "
                    "`CategoricalHyperparameter`, "
                    "`UniformIntegerHyperparameter`, "
                    "`UniformFloatHyperparameter` are supported!"
                )

        size_sparse = src_ind + 1
        size_dense = trg_ind

        return nums, cats, size_sparse, size_dense


class DenseConfiguration(CS.Configuration):
    def __init__(self, configuration_space, *args, **kwargs):

        assert isinstance(configuration_space, DenseConfigurationSpace)
        super(DenseConfiguration, self).__init__(configuration_space, *args, **kwargs)

    @classmethod
    def from_array(cls, configuration_space, array_dense, dtype="float64"):

        assert isinstance(configuration_space, DenseConfigurationSpace)
        cs = configuration_space
        # initialize output array
        array_sparse = np.empty(cs.size_sparse, dtype=dtype)

        # process numerical hyperparameters
        if cs.nums:
            array_sparse[cs.num_src] = array_dense[cs.num_trg]

        # process categorical hyperparameters
        for src_ind, trg_ind, size in cs.cats:
            ind_max = np.argmax(array_dense[trg_ind : trg_ind + size])
            array_sparse[src_ind] = ind_max

        return cls(configuration_space=configuration_space, vector=array_sparse)

    def to_array(self, dtype="float64"):

        cs = self.configuration_space
        array_sparse = super(DenseConfiguration, self).get_array()

        # initialize output array
        array_dense = np.zeros(cs.size_dense, dtype=dtype)

        # process numerical hyperparameters
        if cs.nums:
            array_dense[cs.num_trg] = array_sparse[cs.num_src]

        # process categorical hyperparameters
        if cs.cats:
            cat_trg_offset = np.uintp(array_sparse[cs.cat_src])
            array_dense[cs.cat_trg + cat_trg_offset] = 1

        return array_dense


def dict_from_array(config_space, array):
    config = DenseConfiguration.from_array(config_space, array_dense=array)
    return config.get_dictionary()


def array_from_dict(config_space, dct):
    config = DenseConfiguration(config_space, values=dct)
    return config.to_array()


def from_bounds(bounds):
    if isinstance(bounds, Bounds):
        low = bounds.lb
        high = bounds.ub
        dim = len(low)
        assert dim == len(high), "lower and upper bounds sizes do not match!"
    else:
        # assumes `bounds` is a list of tuples
        low, high = zip(*bounds)
        dim = len(bounds)

    return (low, high), dim


class MaximizableRFClassifier(RandomForestClassifier):
    def maxima(
        self,
        bounds,
        filter_fn=lambda res: True,
        num_samples=1024,
        random_state=None,
        nasbench=False,
    ):
        self._func_min = lambda u: -self.predict_proba(u)[:, 1]
        random_state = check_random_state(random_state)
        (low, high), dim = from_bounds(bounds)

        if not nasbench:
            X_init = random_state.uniform(low=low, high=high, size=(num_samples, dim))
        else:
            # Sample one-hot initializations for nasbenc201
            X_init = []
            for _ in range(num_samples):
                x = np.zeros((6, 5))
                idx = random_state.choice(5, size=6)
                for i in range(6):
                    x[i][idx[i]] = 1
                X_init.append(x.reshape(-1))
            X_init = np.array(X_init)

        z_init = self.predict_proba(X_init)[:, 1]
        # the function to minimize is negative of the classifier output
        f_init = -z_init

        i = np.argmin(f_init, axis=None)
        result = OptimizeResult(x=X_init[i], fun=f_init[i], success=True)
        if filter_fn(result):
            return result


class Record:
    def __init__(self):
        self.features = []
        self.targets = []
        self.budgets = []

    def size(self):
        return len(self.targets)

    def append(self, x, y, b=None):
        self.features.append(x)
        self.targets.append(y)
        if b is not None:
            self.budgets.append(b)

    def load_classification_data(self, gamma, weight_type):
        assert weight_type in ["pi", "ei"]

        if weight_type == "ei":
            X, y = np.vstack(self.features), np.hstack(self.targets)
            tau = np.quantile(y, q=gamma)
            z = np.less(y, tau)
            x1, z1 = X[z], z[z]
            x0, z0 = X, np.zeros_like(z)
            w1 = (tau - y)[z]
            w1 = w1 / np.mean(w1)
            w0 = 1 - z0

            x = np.concatenate([x1, x0], axis=0)
            z = np.concatenate([z1, z0], axis=0)
            s1 = x1.shape[0]
            s0 = x0.shape[0]

            w = np.concatenate([w1 * (s1 + s0) / s1, w0 * (s1 + s0) / s0], axis=0)
            w = w / np.mean(w)
            return x, z, w

        if weight_type == "pi":
            x, y = np.vstack(self.features), np.hstack(self.targets)
            tau = np.quantile(y, q=gamma)
            z = np.less(y, tau)
            return x, z, np.ones_like(z)

    def is_duplicate(self, x, rtol=1e-5, atol=1e-8):
        return any(np.allclose(x_prev, x, rtol=rtol, atol=atol) for x_prev in self.features)


class LFBO:
    def __init__(
        self,
        config_space,
        num_random_init,
        gamma,
        weight_type,
        nasbench,
        num_samples=5000,
        method="L-BFGS-B",
        seed=None,
    ):
        self.config_space = DenseConfigurationSpace(config_space, seed=seed)
        self.num_random_init = num_random_init
        self.nasbench = nasbench
        self.weight_type = weight_type
        self.gamma = gamma
        self.record = Record()
        self.bounds = self.config_space.get_bounds()
        self.num_samples = num_samples
        self.method = method
        self.random_state = np.random.RandomState(seed)
        self.model = MaximizableRFClassifier(n_estimators=1000, min_samples_split=2)

    def _is_unique(self, res):
        is_duplicate = self.record.is_duplicate(res.x)
        return not is_duplicate

    def step(self):
        config_random = self.config_space.sample_configuration()
        config_random_dict = config_random.get_dictionary()

        if self.record.size() < self.num_random_init:
            return config_random_dict

        # Update the classifier
        x, z, w = self.record.load_classification_data(self.gamma, self.weight_type)
        self.model.fit(x, z, sample_weight=w)

        opt = self.model.maxima(
            bounds=self.bounds,
            filter_fn=self._is_unique,
            num_samples=self.num_samples,
            random_state=self.random_state,
            nasbench=self.nasbench,
        )

        if opt is None:
            return config_random_dict

        loc = opt.x
        config_opt_arr = maybe_distort(loc, None, self.bounds, self.random_state)
        config_opt_dict = dict_from_array(self.config_space, config_opt_arr)

        return config_opt_dict

    def add_new_observation(self, config_dict, y):
        config_arr = array_from_dict(self.config_space, config_dict)
        self.record.append(x=config_arr, y=y)
