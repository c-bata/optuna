from typing import Union  # NOQA

from optuna.storages._base import BaseStorage  # NOQA
from optuna.storages._in_memory import InMemoryStorage


def get_storage(storage):
    # type: (Union[None, str, BaseStorage]) -> BaseStorage

    if storage is None:
        return InMemoryStorage()
    else:
        return storage
