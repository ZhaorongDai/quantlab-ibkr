"""Plain-Python views of numpy and pandas values, for JSON files and dict keys.

Private support code shared by the run directory (``outputs.py``), the parity
report (``parity.py``), the run metrics (``metrics.py``) and the decision core
(``decision.py``). It imports numpy and pandas only, so the nautilus-free
decision core can use it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def jsonable(value):
    """Return ``value`` as strict JSON values, as quantlab's ``to_jsonable`` does.

    Dicts and lists are converted item by item (dict keys become strings, tuples
    lists). NaN, infinities and NaT become ``None``, timestamps ISO strings,
    timedeltas their ``str``, numpy scalars Python ones.

    Parameters
    ----------
    value : object
        A scalar, or a dict, list or tuple of them.

    Returns
    -------
    object
        ``value`` with every leaf a JSON-serialisable Python value.

    Examples
    --------
    >>> jsonable({"sharpe": np.float64("nan"), "start": pd.Timestamp("2024-01-02")})
    {'sharpe': None, 'start': '2024-01-02T00:00:00'}
    >>> jsonable([np.int64(3), np.timedelta64("NaT", "ns"), pd.Timedelta("1D"), np.bool_(True)])
    [3, None, '1 days 00:00:00', True]
    """
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (np.datetime64, np.timedelta64)):
        if np.isnat(value):
            return None
        value = pd.Timestamp(value) if isinstance(value, np.datetime64) else pd.Timedelta(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def python_scalar(value):
    """Return a numpy scalar as the equal Python scalar; anything else as is.

    A symbol label read from an array's ``.values`` is a numpy scalar
    (``np.int64(10001)``); unwrapped, it compares, hashes and serialises as
    the Python label quantlab's run directory holds.

    Parameters
    ----------
    value : object
        Any value.

    Returns
    -------
    object
        ``value.item()`` for a numpy scalar, else ``value``.

    Examples
    --------
    >>> python_scalar(np.int64(10001)), python_scalar(np.str_("BUY")), python_scalar(10001)
    (10001, 'BUY', 10001)
    """
    return value.item() if isinstance(value, np.generic) else value
