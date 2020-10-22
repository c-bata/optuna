import abc
import numpy as np


class BaseAlpha(abc.ABC):
    @property
    @abc.abstractmethod
    def value(self) -> float:
        pass

    @abc.abstractmethod
    def update(
        self,
        first_model_pred: np.ndarray,
        second_model_pred: np.ndarray
    ) -> None:
        pass


class ConstantAlpha(BaseAlpha):
    def __init__(self, alpha: float) -> None:
        self._alpha = alpha

    @property
    def value(self) -> float:
        return self._alpha

    def update(
        self,
        first_model_pred: np.ndarray,
        second_model_pred: np.ndarray
    ) -> None:
        pass


class AdaptiveAlpha(BaseAlpha):
    def __init__(
        self,
        population_size: int,
        alpha_min: float = 0.04,
        alpha_max: float = 1.0,
        rde_min: float = 0.0,
        rde_max: float = 1.0,
        lr: float = 0.3
    ) -> None:
        self._mu = population_size // 2
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max

        self._rde = 1.0
        self._rde_min = rde_min
        self._rde_max = rde_max
        self._lr = lr

        self._alpha_cache = alpha_max

    @property
    def value(self) -> float:
        return self._alpha_cache

    def update(
        self,
        first_model_pred: np.ndarray,
        second_model_pred: np.ndarray
    ) -> None:
        ranking_first_model = np.argsort(first_model_pred)
        ranking_second_model = np.argsort(second_model_pred)

        rde = np.sum(
            np.abs(ranking_second_model[:self._mu] - ranking_first_model[:self._mu])
        ) / np.sum(
            np.abs(np.arange(self._mu) - np.argsort(-np.arange(self._mu)))
        )
        self._rde = (1 - self._lr) * self._rde + self._lr * rde

        r = (self._rde - self._rde_min) / (self._rde_max - self._rde_min)
        alpha = self._alpha_min + max(0, min(1, r)) * (self._alpha_max - self._alpha_min)
        self._alpha_cache = alpha
