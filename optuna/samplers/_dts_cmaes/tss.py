import abc

from optuna.trial import FrozenTrial
from typing import List


class BaseTSS(abc.ABC):
    @abc.abstractmethod
    def select(
        self,
        completed_trials: List[FrozenTrial],
    ) -> List[FrozenTrial]:
        pass


class AllTSS(BaseTSS):
    def select(
        self,
        completed_trials: List[FrozenTrial],
    ) -> List[FrozenTrial]:
        return completed_trials
