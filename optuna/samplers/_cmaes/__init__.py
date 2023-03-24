from __future__ import annotations

from typing import Any
from typing import Optional
import warnings

from optuna import Study
from optuna.distributions import BaseDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import BaseSampler
from optuna.samplers._cmaes.sampler_inmemory import InMemoryCmaEsSampler
from optuna.samplers._cmaes.sampler_resumable import ResumableCmaEsSampler
from optuna.trial import FrozenTrial


class CmaEsSampler(BaseSampler):
    """A sampler using `cmaes <https://github.com/CyberAgentAILab/cmaes>`_ as the backend.

    Example:

        Optimize a simple quadratic function by using :class:`~optuna.samplers.CmaEsSampler`.

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -1, 1)
                y = trial.suggest_int("y", -1, 1)
                return x**2 + y


            sampler = optuna.samplers.CmaEsSampler()
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=20)

    Please note that this sampler does not support CategoricalDistribution.
    However, :class:`~optuna.distributions.FloatDistribution` with ``step``,
    (:func:`~optuna.trial.Trial.suggest_float`) and
    :class:`~optuna.distributions.IntDistribution` (:func:`~optuna.trial.Trial.suggest_int`)
    are supported.

    If your search space contains categorical parameters, I recommend you
    to use :class:`~optuna.samplers.TPESampler` instead.
    Furthermore, there is room for performance improvements in parallel
    optimization settings. This sampler cannot use some trials for updating
    the parameters of multivariate normal distribution.

    For further information about CMA-ES algorithm, please refer to the following papers:

    - `N. Hansen, The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772, 2016.
      <https://arxiv.org/abs/1604.00772>`_
    - `A. Auger and N. Hansen. A restart CMA evolution strategy with increasing population
      size. In Proceedings of the IEEE Congress on Evolutionary Computation (CEC 2005),
      pages 1769–1776. IEEE Press, 2005.
      <http://www.cmap.polytechnique.fr/~nikolaus.hansen/cec2005ipopcmaes.pdf>`_
    - `N. Hansen. Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed.
      GECCO Workshop, 2009.
      <https://dl.acm.org/doi/10.1145/1570256.1570333>`_
    - `Raymond Ros, Nikolaus Hansen. A Simple Modification in CMA-ES Achieving Linear Time and
      Space Complexity. 10th International Conference on Parallel Problem Solving From Nature,
      Sep 2008, Dortmund, Germany. inria-00287367.
      <https://hal.inria.fr/inria-00287367/document>`_
    - `Masahiro Nomura, Shuhei Watanabe, Youhei Akimoto, Yoshihiko Ozaki, Masaki Onishi.
      Warm Starting CMA-ES for Hyperparameter Optimization, AAAI. 2021.
      <https://arxiv.org/abs/2012.06932>`_
    - `R. Hamano, S. Saito, M. Nomura, S. Shirakawa. CMA-ES with Margin: Lower-Bounding Marginal
      Probability for Mixed-Integer Black-Box Optimization, GECCO. 2022.
      <https://arxiv.org/abs/2205.13482>`_

    .. seealso::
        You can also use :class:`optuna.integration.PyCmaSampler` which is a sampler using cma
        library as the backend.

    Args:

        x0:
            A dictionary of an initial parameter values for CMA-ES. By default, the mean of ``low``
            and ``high`` for each distribution is used. Note that ``x0`` is sampled uniformly
            within the search space domain for each restart if you specify ``restart_strategy``
            argument.

        sigma0:
            Initial standard deviation of CMA-ES. By default, ``sigma0`` is set to
            ``min_range / 6``, where ``min_range`` denotes the minimum range of the distributions
            in the search space.

        seed:
            A random seed for CMA-ES.

        n_startup_trials:
            The independent sampling is used instead of the CMA-ES algorithm until the given number
            of trials finish in the same study.

        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler.
            The search space for :class:`~optuna.samplers.CmaEsSampler` is determined by
            :func:`~optuna.search_space.intersection_search_space()`.

            If :obj:`None` is specified, :class:`~optuna.samplers.RandomSampler` is used
            as the default.

            .. seealso::
                :class:`optuna.samplers` module provides built-in independent samplers
                such as :class:`~optuna.samplers.RandomSampler` and
                :class:`~optuna.samplers.TPESampler`.

        warn_independent_sampling:
            If this is :obj:`True`, a warning message is emitted when
            the value of a parameter is sampled by using an independent sampler.

            Note that the parameters of the first trial in a study are always sampled
            via an independent sampler, so no warning messages are emitted in this case.

        restart_strategy:
            Strategy for restarting CMA-ES optimization when converges to a local minimum.
            If :obj:`None` is given, CMA-ES will not restart (default).
            If 'ipop' is given, CMA-ES will restart with increasing population size.
            if 'bipop' is given, CMA-ES will restart with the population size
            increased or decreased.
            Please see also ``inc_popsize`` parameter.

            .. note::
                Added in v2.1.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.1.0.

        popsize:
            A population size of CMA-ES. When ``restart_strategy = 'ipop'``
            or ``restart_strategy = 'bipop'`` is specified,
            this is used as the initial population size.

        inc_popsize:
            Multiplier for increasing population size before each restart.
            This argument will be used when ``restart_strategy = 'ipop'``
            or ``restart_strategy = 'bipop'`` is specified.

        consider_pruned_trials:
            If this is :obj:`True`, the PRUNED trials are considered for sampling.

            .. note::
                Added in v2.0.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.0.0.

            .. note::
                It is suggested to set this flag :obj:`False` when the
                :class:`~optuna.pruners.MedianPruner` is used. On the other hand, it is suggested
                to set this flag :obj:`True` when the :class:`~optuna.pruners.HyperbandPruner` is
                used. Please see `the benchmark result
                <https://github.com/optuna/optuna/pull/1229>`_ for the details.

        use_separable_cma:
            If this is :obj:`True`, the covariance matrix is constrained to be diagonal.
            Due to reduce the model complexity, the learning rate for the covariance matrix
            is increased. Consequently, this algorithm outperforms CMA-ES on separable functions.

            .. note::
                Added in v2.6.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.6.0.

        with_margin:
            If this is :obj:`True`, CMA-ES with margin is used. This algorithm prevents samples in
            each discrete distribution (:class:`~optuna.distributions.FloatDistribution` with
            `step` and :class:`~optuna.distributions.IntDistribution`) from being fixed to a single
            point.
            Currently, this option cannot be used with ``use_separable_cma=True``.

            .. note::
                Added in v3.1.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v3.1.0.

        source_trials:
            This option is for Warm Starting CMA-ES, a method to transfer prior knowledge on
            similar HPO tasks through the initialization of CMA-ES. This method estimates a
            promising distribution from ``source_trials`` and generates the parameter of
            multivariate gaussian distribution. Please note that it is prohibited to use
            ``x0``, ``sigma0``, or ``use_separable_cma`` argument together.

            .. note::
                Added in v2.6.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.6.0.

        keep_optimizer_inmemory:
            If this is :obj:`True`, the CMA-ES optimizer object is stored in memory, which makes
            the program run faster and the internal implementation much simpler.

            .. note::
                Added in v3.2.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v3.2.0.

    """

    def __init__(
        self,
        x0: Optional[dict[str, Any]] = None,
        sigma0: Optional[float] = None,
        n_startup_trials: int = 1,
        independent_sampler: Optional[BaseSampler] = None,
        warn_independent_sampling: bool = True,
        seed: Optional[int] = None,
        *,
        consider_pruned_trials: bool = False,
        restart_strategy: Optional[str] = None,
        popsize: Optional[int] = None,
        inc_popsize: int = 2,
        use_separable_cma: bool = False,
        with_margin: bool = False,
        source_trials: Optional[list[FrozenTrial]] = None,
        keep_optimizer_inmemory: bool = False,
    ) -> None:
        if keep_optimizer_inmemory:
            warnings.warn(
                "`keep_optimizer_inmemory` option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )
            self._backend_sampler = InMemoryCmaEsSampler(
                x0,
                sigma0,
                n_startup_trials,
                independent_sampler,
                warn_independent_sampling,
                seed,
                consider_pruned_trials=consider_pruned_trials,
                restart_strategy=restart_strategy,
                popsize=popsize,
                inc_popsize=inc_popsize,
                use_separable_cma=use_separable_cma,
                with_margin=with_margin,
                source_trials=source_trials,
            )
        else:
            self._backend_sampler = ResumableCmaEsSampler(
                x0,
                sigma0,
                n_startup_trials,
                independent_sampler,
                warn_independent_sampling,
                seed,
                consider_pruned_trials=consider_pruned_trials,
                restart_strategy=restart_strategy,
                popsize=popsize,
                inc_popsize=inc_popsize,
                use_separable_cma=use_separable_cma,
                with_margin=with_margin,
                source_trials=source_trials,
            )

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return self._backend_sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return self._backend_sampler.sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._backend_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )
