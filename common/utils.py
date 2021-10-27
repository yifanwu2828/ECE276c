import random
import uuid
from datetime import datetime
from time import sleep
from itertools import zip_longest
from typing import Tuple, Dict, Any, Iterable, Union, Optional

import numpy as np
import torch as th


def make_unique_timestamp() -> str:
    """Timestamp, with random uuid added to avoid collisions."""
    ISO_TIMESTAMP = "%Y%m%d_%H%M_%S"
    timestamp = datetime.now().strftime(ISO_TIMESTAMP)
    random_uuid = uuid.uuid4().hex[:3]
    return f"{timestamp}_{random_uuid}"


def set_random_seed(seed: int) -> None:
    """Set random seed to both numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)


def countdown(t_sec: int) -> None:
    """Countdown t seconds."""
    while t_sec:
        mins, secs = divmod(t_sec, 60)
        time_format = f"{mins: 02d}:{secs: 02d}"
        print(time_format, end="\r")
        sleep(1)
        t_sec -= 1
    print("Done!!")


def get_stats(x: np.ndarray) -> Tuple[np.ndarray, ...]:
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    return x.mean(), x.std(), x.min(), x.max()  # noqa


def combined_shape(length: int, shape: Optional[Tuple[int, ...]] = None):
    if shape is None:
        return (length,)  # noqa
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def zip_strict(*iterables: Iterable) -> Iterable:
    """
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.
    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    # ! Slow
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo
