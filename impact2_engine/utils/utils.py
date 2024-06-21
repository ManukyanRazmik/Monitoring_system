from typing import Any
from numpy import quantile, min, max
from pandas import Series

class MissingDataException(Exception):
    def __init__(self, message):
        super().__init__(message)


def quantile_025(vec: Series) -> Any:
    """2.5 percentile"""
    return quantile(vec, .025)


def quantile_975(vec: Series) -> Any:
    """97,5 percentile"""
    return quantile(vec, .975)


def raw_data(vec: Series) -> tuple[Any, ...]:
    """Vector-column of grouped raw values"""
    return tuple(vec)

def diff(vec: Series) -> float:
    """Range (aka np.ptp), NaN-safe and conveniently named"""
    return max(vec) - min(vec)
