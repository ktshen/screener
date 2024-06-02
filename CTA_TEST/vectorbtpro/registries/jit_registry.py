# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Global registry for jittables.

Jitting is a process of just-in-time compiling functions to make their execution faster.
A jitter is a decorator that wraps a regular Python function and returns the decorated function.
Depending upon a jitter, this decorated function has the same or at least a similar signature
to the function that has been decorated. Jitters take various jitter-specific options
to change the behavior of execution; that is, a single regular Python function can be
decorated by multiple jitter instances (for example, one jitter for decorating a function
with `numba.jit` and another jitter for doing the same with `parallel=True` flag).

In addition to jitters, vectorbt introduces the concept of tasks. One task can be
executed by multiple jitter types (such as NumPy, Numba, and JAX). For example, one
can create a task that converts price into returns and implements it using NumPy and Numba.
Those implementations are registered by `JITRegistry` as `JitableSetup` instances, are stored
in `JITRegistry.jitable_setups`, and can be uniquely identified by the task id and jitter type.
Note that `JitableSetup` instances contain only information on how to decorate a function.

The decorated function itself and the jitter that has been used are registered as a `JittedSetup`
instance and stored in `JITRegistry.jitted_setups`. It acts as a cache to quickly retrieve an
already decorated function and to avoid recompilation.

Let's implement a task that takes a sum over an array using both NumPy and Numba:

```pycon
>>> import vectorbtpro as vbt
>>> import numpy as np
>>> import pandas as pd

>>> @vbt.register_jitted(task_id_or_func='sum')
... def sum_np(a):
...     return a.sum()

>>> @vbt.register_jitted(task_id_or_func='sum')
... def sum_nb(a):
...     out = 0.
...     for i in range(a.shape[0]):
...         out += a[i]
...     return out
```

We can see that two new jitable setups were registered:

```pycon
>>> vbt.jit_reg.jitable_setups['sum']
{'np': JitableSetup(task_id='sum', jitter_id='np', py_func=<function sum_np at 0x7fea215b1e18>, jitter_kwargs={}, tags=None),
 'nb': JitableSetup(task_id='sum', jitter_id='nb', py_func=<function sum_nb at 0x7fea273d41e0>, jitter_kwargs={}, tags=None)}
```

Moreover, two jitted setups were registered for our decorated functions:

```pycon
>>> from vectorbtpro.registries.jit_registry import JitableSetup

>>> hash_np = JitableSetup.get_hash('sum', 'np')
>>> vbt.jit_reg.jitted_setups[hash_np]
{3527539: JittedSetup(jitter=<vectorbtpro.utils.jitting.NumPyJitter object at 0x7fea21506080>, jitted_func=<function sum_np at 0x7fea215b1e18>)}

>>> hash_nb = JitableSetup.get_hash('sum', 'nb')
>>> vbt.jit_reg.jitted_setups[hash_nb]
{6326224984503844995: JittedSetup(jitter=<vectorbtpro.utils.jitting.NumbaJitter object at 0x7fea214d0ba8>, jitted_func=CPUDispatcher(<function sum_nb at 0x7fea273d41e0>))}
```

These setups contain decorated functions with the options passed during the registration.
When we call `JITRegistry.resolve` without any additional keyword arguments,
`JITRegistry` returns exactly these functions:

```pycon
>>> jitted_func = vbt.jit_reg.resolve('sum', jitter='nb')
>>> jitted_func
CPUDispatcher(<function sum_nb at 0x7fea273d41e0>)

>>> jitted_func.targetoptions
{'nopython': True, 'nogil': True, 'parallel': False, 'boundscheck': False}
```

Once we pass any other option, the Python function will be redecorated, and another `JittedOption`
instance will be registered:

```pycon
>>> jitted_func = vbt.jit_reg.resolve('sum', jitter='nb', nopython=False)
>>> jitted_func
CPUDispatcher(<function sum_nb at 0x7fea273d41e0>)

>>> jitted_func.targetoptions
{'nopython': False, 'nogil': True, 'parallel': False, 'boundscheck': False}

>>> vbt.jit_reg.jitted_setups[hash_nb]
{6326224984503844995: JittedSetup(jitter=<vectorbtpro.utils.jitting.NumbaJitter object at 0x7fea214d0ba8>, jitted_func=CPUDispatcher(<function sum_nb at 0x7fea273d41e0>)),
 -2979374923679407948: JittedSetup(jitter=<vectorbtpro.utils.jitting.NumbaJitter object at 0x7fea00bf94e0>, jitted_func=CPUDispatcher(<function sum_nb at 0x7fea273d41e0>))}
```

## Templates

Templates can be used to, based on the current context, dynamically select the jitter or
keyword arguments for jitting. For example, let's pick the NumPy jitter over any other
jitter if there are more than two of them for a given task:

```pycon
>>> vbt.jit_reg.resolve('sum', jitter=vbt.RepEval("'nb' if 'nb' in task_setups else None"))
CPUDispatcher(<function sum_nb at 0x7fea273d41e0>)
```

## Disabling

In the case we want to disable jitting, we can simply pass `disable=True` to `JITRegistry.resolve`:

```pycon
>>> py_func = vbt.jit_reg.resolve('sum', jitter='nb', disable=True)
>>> py_func
<function __main__.sum_nb(a)>
```

We can also disable jitting globally:

```pycon
>>> vbt.settings.jitting['disable'] = True

>>> vbt.jit_reg.resolve('sum', jitter='nb')
<function __main__.sum_nb(a)>
```

!!! hint
    If we don't plan to use any additional options and we have only one jitter registered per task,
    we can also disable resolution to increase performance.

!!! warning
    Disabling jitting globally only applies to functions resolved using `JITRegistry.resolve`.
    Any decorated function that is being called directly will be executed as usual.

## Jitted option

Since most functions that call other jitted functions in vectorbt have a `jitted` argument,
you can pass `jitted` as a dictionary with options, as a string denoting the jitter, or False
to disable jitting (see `vectorbtpro.utils.jitting.resolve_jitted_option`):

```pycon
>>> def sum_arr(arr, jitted=None):
...     func = vbt.jit_reg.resolve_option('sum', jitted)
...     return func(arr)

>>> arr = np.random.uniform(size=1000000)

>>> %timeit sum_arr(arr, jitted='np')
319 µs ± 3.35 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

>>> %timeit sum_arr(arr, jitted='nb')
1.09 ms ± 4.13 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

>>> %timeit sum_arr(arr, jitted=dict(jitter='nb', disable=True))
133 ms ± 2.32 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

!!! hint
    A good rule of thumb is: whenever a caller function accepts a `jitted` argument,
    the jitted functions it calls are most probably resolved using `JITRegistry.resolve_option`.

## Changing options upon registration

Options are usually specified upon registration using `register_jitted`:

```pycon
>>> from numba import prange

>>> @vbt.register_jitted(parallel=True, tags={'can_parallel'})
... def sum_parallel_nb(a):
...     out = np.empty(a.shape[1])
...     for col in prange(a.shape[1]):
...         total = 0.
...         for i in range(a.shape[0]):
...             total += a[i, col]
...         out[col] = total
...     return out

>>> sum_parallel_nb.targetoptions
{'nopython': True, 'nogil': True, 'parallel': True, 'boundscheck': False}
```

But what if we wanted to change the registration options of vectorbt's own jitable functions,
such as `vectorbtpro.generic.nb.base.diff_nb`? For example, let's disable caching for all Numba functions.

```pycon
>>> vbt.settings.jitting.jitters['nb']['override_options'] = dict(cache=False)
```

Since all functions have already been registered, the above statement has no effect:

```pycon
>>> vbt.jit_reg.jitable_setups['vectorbtpro.generic.nb.base.diff_nb']['nb'].jitter_kwargs
{'cache': True}
```

In order for them to be applied, we need to save the settings to a file and
load them before all functions are imported:

```pycon
>>> vbt.settings.save('my_settings')
```

Let's restart the runtime and instruct vectorbt to load the file with settings before anything else:

```pycon
>>> import os
>>> os.environ['VBT_SETTINGS_PATH'] = "my_settings"

>>> import vectorbtpro as vbt
>>> vbt.jit_reg.jitable_setups['vectorbtpro.generic.nb.base.diff_nb']['nb'].jitter_kwargs
{'cache': False}
```

We can also change the registration options for some specific tasks, and even replace Python functions.
For example, we can change the implementation in the deepest places of the core.
Let's change the default `ddof` from 0 to 1 in `vectorbtpro.generic.nb.base.nanstd_1d_nb` and disable caching with Numba:

```pycon
>>> from vectorbtpro.generic.nb import nanstd_1d_nb, nanvar_1d_nb

>>> nanstd_1d_nb(np.array([1, 2, 3]))
0.816496580927726

>>> def new_nanstd_1d_nb(arr, ddof=1):
...     return np.sqrt(nanvar_1d_nb(arr, ddof=ddof))

>>> vbt.settings.jitting.jitters['nb']['tasks']['vectorbtpro.generic.nb.base.nanstd_1d_nb'] = dict(
...     replace_py_func=new_nanstd_1d_nb,
...     override_options=dict(
...         cache=False
...     )
... )

>>> vbt.settings.save('my_settings')
```

After restarting the runtime:

```pycon
>>> import os
>>> os.environ['VBT_SETTINGS_PATH'] = "my_settings"

>>> import numpy as np
>>> from vectorbtpro.generic.nb import nanstd_1d_nb, nanvar_1d_nb

>>> nanstd_1d_nb(np.array([1, 2, 3]))
1.0
```

!!! note
    All of the above examples require saving the setting to a file, restarting the runtime,
    setting the path to the file to an environment variable, and only then importing vectorbtpro.

## Changing options upon resolution

Another approach but without the need to restart the runtime is by changing the options
upon resolution using `JITRegistry.resolve_option`:

```pycon
>>> # On specific Numba function
>>> vbt.settings.jitting.jitters['nb']['tasks']['vectorbtpro.generic.nb.base.diff_nb'] = dict(
...     resolve_kwargs=dict(
...         nogil=False
...     )
... )

>>> # disabled
>>> vbt.jit_reg.resolve('vectorbtpro.generic.nb.base.diff_nb', jitter='nb').targetoptions
{'nopython': True, 'nogil': False, 'parallel': False, 'boundscheck': False}

>>> # still enabled
>>> vbt.jit_reg.resolve('sum', jitter='nb').targetoptions
{'nopython': True, 'nogil': True, 'parallel': False, 'boundscheck': False}

>>> # On each Numba function
>>> vbt.settings.jitting.jitters['nb']['resolve_kwargs'] = dict(nogil=False)

>>> # disabled
>>> vbt.jit_reg.resolve('vectorbtpro.generic.nb.base.diff_nb', jitter='nb').targetoptions
{'nopython': True, 'nogil': False, 'parallel': False, 'boundscheck': False}

>>> # disabled
>>> vbt.jit_reg.resolve('sum', jitter='nb').targetoptions
{'nopython': True, 'nogil': False, 'parallel': False, 'boundscheck': False}
```

## Building custom jitters

Let's build a custom jitter on top of `vectorbtpro.utils.jitting.NumbaJitter` that converts
any argument that contains a Pandas object to a 2-dimensional NumPy array prior to decoration:

```pycon
>>> from functools import wraps
>>> from vectorbtpro.utils.jitting import NumbaJitter
>>> import pandas as pd

>>> class SafeNumbaJitter(NumbaJitter):
...     def decorate(self, py_func, tags=None):
...         if self.wrapping_disabled:
...             return py_func
...
...         @wraps(py_func)
...         def wrapper(*args, **kwargs):
...             new_args = ()
...             for arg in args:
...                 if isinstance(arg, pd.Series):
...                     arg = np.expand_dims(arg.values, 1)
...                 elif isinstance(arg, pd.DataFrame):
...                     arg = arg.values
...                 new_args += (arg,)
...             new_kwargs = dict()
...             for k, v in kwargs.items():
...                 if isinstance(v, pd.Series):
...                     v = np.expand_dims(v.values, 1)
...                 elif isinstance(v, pd.DataFrame):
...                     v = v.values
...                 new_kwargs[k] = v
...             return NumbaJitter.decorate(self, py_func, tags=tags)(*new_args, **new_kwargs)
...         return wrapper
```

After we have defined our jitter class, we need to register it globally:

```pycon
>>> vbt.settings.jitting.jitters['safe_nb'] = dict(cls=SafeNumbaJitter)
```

Finally, we can execute any Numba function by specifying our new jitter:

```pycon
>>> func = vbt.jit_reg.resolve(
...     task_id_or_func=vbt.generic.nb.diff_nb,
...     jitter='safe_nb',
...     allow_new=True
... )
>>> func(pd.DataFrame([[1, 2], [3, 4]]))
array([[nan, nan],
       [ 2.,  2.]])
```

Whereas executing the same func using the vanilla Numba jitter causes an error:

```pycon
>>> func = vbt.jit_reg.resolve(task_id_or_func=vbt.generic.nb.diff_nb)
>>> func(pd.DataFrame([[1, 2], [3, 4]]))
Failed in nopython mode pipeline (step: nopython frontend)
non-precise type pyobject
```

!!! note
    Make sure to pass a function as `task_id_or_func` if the jitted function hasn't been registered yet.

    This jitter cannot be used for decorating Numba functions that should be called
    from other Numba functions since the convertion operation is done using Python.
"""

import attr

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts, atomic_dict
from vectorbtpro.utils.hashing import Hashable
from vectorbtpro.utils.jitting import (
    Jitter,
    resolve_jitted_kwargs,
    resolve_jitter_type,
    resolve_jitter,
    get_id_of_jitter_type,
    get_func_suffix,
)
from vectorbtpro.utils.template import RepEval, substitute_templates, CustomTemplate

__all__ = [
    "jit_reg",
    "register_jitted",
]


def get_func_full_name(func: tp.Callable) -> str:
    """Get full name of the func to be used as task id."""
    return func.__module__ + "." + func.__name__


@attr.s(frozen=True, eq=False)
class JitableSetup(Hashable):
    """Class that represents a jitable setup.

    !!! note
        Hashed solely by `task_id` and `jitter_id`."""

    task_id: tp.Hashable = attr.ib()
    """Task id."""

    jitter_id: tp.Hashable = attr.ib()
    """Jitter id."""

    py_func: tp.Callable = attr.ib()
    """Python function to be jitted."""

    jitter_kwargs: tp.KwargsLike = attr.ib(default=None)
    """Keyword arguments passed to `vectorbtpro.utils.jitting.resolve_jitter`."""

    tags: tp.SetLike = attr.ib(default=None)
    """Set of tags."""

    @staticmethod
    def get_hash(task_id: tp.Hashable, jitter_id: tp.Hashable) -> int:
        return hash((task_id, jitter_id))

    @property
    def hash_key(self) -> tuple:
        return (self.task_id, self.jitter_id)


@attr.s(frozen=True, eq=False)
class JittedSetup(Hashable):
    """Class that represents a jitted setup.

    !!! note
        Hashed solely by sorted config of `jitter`. That is, two jitters with the same config
        will yield the same hash and the function won't be re-decorated."""

    jitter: Jitter = attr.ib()
    """Jitter that decorated the function."""

    jitted_func: tp.Callable = attr.ib()
    """Decorated function."""

    @staticmethod
    def get_hash(jitter: Jitter) -> int:
        return hash(tuple(sorted(jitter.config.items())))

    @property
    def hash_key(self) -> tuple:
        return tuple(sorted(self.jitter.config.items()))


class JITRegistry:
    """Class that registers jitted functions."""

    def __init__(self) -> None:
        self._jitable_setups = {}
        self._jitted_setups = {}

    @property
    def jitable_setups(self) -> tp.Dict[tp.Hashable, tp.Dict[tp.Hashable, JitableSetup]]:
        """Dict of registered `JitableSetup` instances by `task_id` and `jitter_id`."""
        return self._jitable_setups

    @property
    def jitted_setups(self) -> tp.Dict[int, tp.Dict[int, JittedSetup]]:
        """Nested dict of registered `JittedSetup` instances by hash of their `JitableSetup` instance."""
        return self._jitted_setups

    def register_jitable_setup(
        self,
        task_id: tp.Hashable,
        jitter_id: tp.Hashable,
        py_func: tp.Callable,
        jitter_kwargs: tp.KwargsLike = None,
        tags: tp.Optional[set] = None,
    ) -> JitableSetup:
        """Register a jitable setup."""
        jitable_setup = JitableSetup(
            task_id=task_id,
            jitter_id=jitter_id,
            py_func=py_func,
            jitter_kwargs=jitter_kwargs,
            tags=tags,
        )
        if task_id not in self.jitable_setups:
            self.jitable_setups[task_id] = dict()
        if jitter_id not in self.jitable_setups[task_id]:
            self.jitable_setups[task_id][jitter_id] = jitable_setup
        return jitable_setup

    def register_jitted_setup(
        self,
        jitable_setup: JitableSetup,
        jitter: Jitter,
        jitted_func: tp.Callable,
    ) -> JittedSetup:
        """Register a jitted setup."""
        jitable_setup_hash = hash(jitable_setup)
        jitted_setup = JittedSetup(jitter=jitter, jitted_func=jitted_func)
        jitted_setup_hash = hash(jitted_setup)
        if jitable_setup_hash not in self.jitted_setups:
            self.jitted_setups[jitable_setup_hash] = dict()
        if jitted_setup_hash not in self.jitted_setups[jitable_setup_hash]:
            self.jitted_setups[jitable_setup_hash][jitted_setup_hash] = jitted_setup
        return jitted_setup

    def decorate_and_register(
        self,
        task_id: tp.Hashable,
        py_func: tp.Callable,
        jitter: tp.Optional[tp.JitterLike] = None,
        jitter_kwargs: tp.KwargsLike = None,
        tags: tp.Optional[set] = None,
    ):
        """Decorate a jitable function and register both jitable and jitted setups."""
        if jitter_kwargs is None:
            jitter_kwargs = {}
        jitter = resolve_jitter(jitter=jitter, py_func=py_func, **jitter_kwargs)
        jitter_id = get_id_of_jitter_type(type(jitter))
        if jitter_id is None:
            raise ValueError("Jitter id cannot be None: is jitter registered globally?")
        jitable_setup = self.register_jitable_setup(task_id, jitter_id, py_func, jitter_kwargs=jitter_kwargs, tags=tags)
        jitted_func = jitter.decorate(py_func, tags=tags)
        self.register_jitted_setup(jitable_setup, jitter, jitted_func)
        return jitted_func

    def match_jitable_setups(
        self,
        expression: tp.Optional[str] = None,
        context: tp.KwargsLike = None,
    ) -> tp.Set[JitableSetup]:
        """Match jitable setups against an expression with each setup being a context."""
        matched_setups = set()
        for setups_by_jitter_id in self.jitable_setups.values():
            for setup in setups_by_jitter_id.values():
                if expression is None:
                    result = True
                else:
                    result = RepEval(expression).substitute(context=merge_dicts(attr.asdict(setup), context))
                    checks.assert_instance_of(result, bool)

                if result:
                    matched_setups.add(setup)
        return matched_setups

    def match_jitted_setups(
        self,
        jitable_setup: JitableSetup,
        expression: tp.Optional[str] = None,
        context: tp.KwargsLike = None,
    ) -> tp.Set[JittedSetup]:
        """Match jitted setups of a jitable setup against an expression with each setup a context."""
        matched_setups = set()
        for setup in self.jitted_setups[hash(jitable_setup)].values():
            if expression is None:
                result = True
            else:
                result = RepEval(expression).substitute(context=merge_dicts(attr.asdict(setup), context))
                checks.assert_instance_of(result, bool)

            if result:
                matched_setups.add(setup)
        return matched_setups

    def resolve(
        self,
        task_id_or_func: tp.Union[tp.Hashable, tp.Callable],
        jitter: tp.Optional[tp.Union[tp.JitterLike, CustomTemplate]] = None,
        disable: tp.Optional[tp.Union[bool, CustomTemplate]] = None,
        disable_resolution: tp.Optional[bool] = None,
        allow_new: tp.Optional[bool] = None,
        register_new: tp.Optional[bool] = None,
        return_missing_task: bool = False,
        template_context: tp.Optional[tp.Mapping] = None,
        tags: tp.Optional[set] = None,
        **jitter_kwargs,
    ) -> tp.Union[tp.Hashable, tp.Callable]:
        """Resolve jitted function for the given task id.

        For details on the format of `task_id_or_func`, see `register_jitted`.

        Jitter keyword arguments are merged in the following order:

        * `jitable_setup.jitter_kwargs`
        * `jitter.your_jitter.resolve_kwargs` in `vectorbtpro._settings.jitting`
        * `jitter.your_jitter.tasks.your_task.resolve_kwargs` in `vectorbtpro._settings.jitting`
        * `jitter_kwargs`

        Templates are substituted in `jitter`, `disable`, and `jitter_kwargs`.

        Set `disable` to True to return the Python function without decoration.
        If `disable_resolution` is enabled globally, `task_id_or_func` is returned unchanged.

        !!! note
            `disable` is only being used by `JITRegistry`, not `vectorbtpro.utils.jitting`.

        !!! note
            If there are more than one jitted setups registered for a single task id,
            make sure to provide a jitter.

        If no jitted setup of type `JittedSetup` was found and `allow_new` is True,
        decorates and returns the function supplied as `task_id_or_func` (otherwise throws an error).

        Set `return_missing_task` to True to return `task_id_or_func` if it cannot be found
        in `JITRegistry.jitable_setups`.
        """
        from vectorbtpro._settings import settings

        jitting_cfg = settings["jitting"]

        if disable_resolution is None:
            disable_resolution = jitting_cfg["disable_resolution"]
        if disable_resolution:
            return task_id_or_func

        if allow_new is None:
            allow_new = jitting_cfg["allow_new"]
        if register_new is None:
            register_new = jitting_cfg["register_new"]

        if hasattr(task_id_or_func, "py_func"):
            py_func = task_id_or_func.py_func
            task_id = get_func_full_name(py_func)
        elif callable(task_id_or_func):
            py_func = task_id_or_func
            task_id = get_func_full_name(py_func)
        else:
            py_func = None
            task_id = task_id_or_func

        if task_id not in self.jitable_setups:
            if not allow_new:
                if return_missing_task:
                    return task_id_or_func
                raise KeyError(f"Task id '{task_id}' not registered")
        task_setups = self.jitable_setups.get(task_id, dict())

        template_context = merge_dicts(
            jitting_cfg["template_context"],
            template_context,
            dict(task_id=task_id, py_func=py_func, task_setups=atomic_dict(task_setups)),
        )
        jitter = substitute_templates(jitter, template_context, sub_id="jitter")

        if jitter is None and py_func is not None:
            jitter = get_func_suffix(py_func)

        if jitter is None:
            if len(task_setups) > 1:
                raise ValueError(
                    f"There are multiple registered setups for task id '{task_id}'. Please specify the jitter."
                )
            elif len(task_setups) == 0:
                raise ValueError(f"There are no registered setups for task id '{task_id}'")
            jitable_setup = list(task_setups.values())[0]
            jitter = jitable_setup.jitter_id
            jitter_id = jitable_setup.jitter_id
        else:
            jitter_type = resolve_jitter_type(jitter=jitter)
            jitter_id = get_id_of_jitter_type(jitter_type)
            if jitter_id not in task_setups:
                if not allow_new:
                    raise KeyError(f"Jitable setup with task id '{task_id}' and jitter id '{jitter_id}' not registered")
                jitable_setup = None
            else:
                jitable_setup = task_setups[jitter_id]
        if jitter_id is None:
            raise ValueError("Jitter id cannot be None: is jitter registered globally?")
        if jitable_setup is None and py_func is None:
            raise ValueError(f"Unable to find Python function for task id '{task_id}' and jitter id '{jitter_id}'")

        template_context = merge_dicts(
            template_context,
            dict(jitter_id=jitter_id, jitter=jitter, jitable_setup=jitable_setup),
        )
        disable = substitute_templates(disable, template_context, sub_id="disable")
        if disable is None:
            disable = jitting_cfg["disable"]
        if disable:
            if jitable_setup is None:
                return py_func
            return jitable_setup.py_func

        if not isinstance(jitter, Jitter):
            jitter_cfg = jitting_cfg["jitters"].get(jitter_id, {})
            setup_cfg = jitter_cfg.get("tasks", {}).get(task_id, {})

            jitter_kwargs = merge_dicts(
                jitable_setup.jitter_kwargs if jitable_setup is not None else None,
                jitter_cfg.get("resolve_kwargs", None),
                setup_cfg.get("resolve_kwargs", None),
                jitter_kwargs,
            )
            jitter_kwargs = substitute_templates(jitter_kwargs, template_context, sub_id="jitter_kwargs")
            jitter = resolve_jitter(jitter=jitter, **jitter_kwargs)

        if jitable_setup is not None:
            jitable_hash = hash(jitable_setup)
            jitted_hash = JittedSetup.get_hash(jitter)
            if jitable_hash in self.jitted_setups and jitted_hash in self.jitted_setups[jitable_hash]:
                return self.jitted_setups[jitable_hash][jitted_hash].jitted_func
        else:
            if register_new:
                return self.decorate_and_register(
                    task_id=task_id,
                    py_func=py_func,
                    jitter=jitter,
                    jitter_kwargs=jitter_kwargs,
                    tags=tags,
                )
            return jitter.decorate(py_func, tags=tags)

        jitted_func = jitter.decorate(jitable_setup.py_func, tags=jitable_setup.tags)
        self.register_jitted_setup(jitable_setup, jitter, jitted_func)

        return jitted_func

    def resolve_option(
        self,
        task_id: tp.Union[tp.Hashable, tp.Callable],
        option: tp.JittedOption,
        **kwargs,
    ) -> tp.Union[tp.Hashable, tp.Callable]:
        """Resolve `option` using `vectorbtpro.utils.jitting.resolve_jitted_option` and call `JITRegistry.resolve`."""
        kwargs = resolve_jitted_kwargs(option=option, **kwargs)
        if kwargs is None:
            kwargs = dict(disable=True)
        return self.resolve(task_id, **kwargs)


jit_reg = JITRegistry()
"""Default registry of type `JITRegistry`."""


def register_jitted(
    py_func: tp.Optional[tp.Callable] = None,
    task_id_or_func: tp.Optional[tp.Union[tp.Hashable, tp.Callable]] = None,
    registry: JITRegistry = jit_reg,
    tags: tp.Optional[set] = None,
    **options,
) -> tp.Callable:
    """Decorate and register a jitable function using `JITRegistry.decorate_and_register`.

    If `task_id_or_func` is a callable, gets replaced by the callable's module name and function name.
    Additionally, the function name may contain a suffix pointing at the jitter (such as `_nb`).

    Options are merged in the following order:

    * `jitters.{jitter_id}.options` in `vectorbtpro._settings.jitting`
    * `jitters.{jitter_id}.tasks.{task_id}.options` in `vectorbtpro._settings.jitting`
    * `options`
    * `jitters.{jitter_id}.override_options` in `vectorbtpro._settings.jitting`
    * `jitters.{jitter_id}.tasks.{task_id}.override_options` in `vectorbtpro._settings.jitting`

    `py_func` can also be overridden using `jitters.your_jitter.tasks.your_task.replace_py_func`
    in `vectorbtpro._settings.jitting`."""

    def decorator(_py_func: tp.Callable) -> tp.Callable:
        nonlocal options

        from vectorbtpro._settings import settings

        jitting_cfg = settings["jitting"]

        if task_id_or_func is None:
            task_id = get_func_full_name(_py_func)
        elif hasattr(task_id_or_func, "py_func"):
            task_id = get_func_full_name(task_id_or_func.py_func)
        elif callable(task_id_or_func):
            task_id = get_func_full_name(task_id_or_func)
        else:
            task_id = task_id_or_func

        jitter = options.pop("jitter", None)
        jitter_type = resolve_jitter_type(jitter=jitter, py_func=_py_func)
        jitter_id = get_id_of_jitter_type(jitter_type)

        jitter_cfg = jitting_cfg["jitters"].get(jitter_id, {})
        setup_cfg = jitter_cfg.get("tasks", {}).get(task_id, {})
        options = merge_dicts(
            jitter_cfg.get("options", None),
            setup_cfg.get("options", None),
            options,
            jitter_cfg.get("override_options", None),
            setup_cfg.get("override_options", None),
        )
        if setup_cfg.get("replace_py_func", None) is not None:
            _py_func = setup_cfg["replace_py_func"]
            if task_id_or_func is None:
                task_id = get_func_full_name(_py_func)

        return registry.decorate_and_register(
            task_id=task_id,
            py_func=_py_func,
            jitter=jitter,
            jitter_kwargs=options,
            tags=tags,
        )

    if py_func is None:
        return decorator
    return decorator(py_func)
