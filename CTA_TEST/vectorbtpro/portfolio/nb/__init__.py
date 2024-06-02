"""Numba-compiled functions for working with portfolio.

Provides an arsenal of Numba-compiled functions that are used for portfolio
simulation, such as generating and filling orders. These only accept NumPy arrays and
other Numba-compatible types.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.

    All functions passed as argument must be Numba-compiled.

    Records must retain the order they were created in.

!!! warning
    Accumulation of roundoff error possible.
    See [here](https://en.wikipedia.org/wiki/Round-off_error#Accumulation_of_roundoff_error) for explanation.

    Rounding errors can cause trades and positions to not close properly:

    ```pycon
    >>> print('%.50f' % 0.1)  # has positive error
    0.10000000000000000555111512312578270211815834045410

    >>> # many buy transactions with positive error -> cannot close position
    >>> sum([0.1 for _ in range(1000000)]) - 100000
    1.3328826753422618e-06

    >>> print('%.50f' % 0.3)  # has negative error
    0.29999999999999998889776975374843459576368331909180

    >>> # many sell transactions with negative error -> cannot close position
    >>> 300000 - sum([0.3 for _ in range(1000000)])
    5.657668225467205e-06
    ```

    While vectorbt has implemented tolerance checks when comparing floats for equality,
    adding/subtracting small amounts large number of times may still introduce a noticable
    error that cannot be corrected post factum.

    To mitigate this issue, avoid repeating lots of micro-transactions of the same sign.
    For example, reduce by `np.inf` or `position_now` to close a long/short position.

    See `vectorbtpro.utils.math_` for current tolerance values.

!!! warning
    Make sure to use `parallel=True` only if your columns are independent.
"""

from vectorbtpro.portfolio.nb.analysis import *
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.portfolio.nb.from_order_func import *
from vectorbtpro.portfolio.nb.from_orders import *
from vectorbtpro.portfolio.nb.from_signals import *
from vectorbtpro.portfolio.nb.iter_ import *
from vectorbtpro.portfolio.nb.records import *

__all__ = []
