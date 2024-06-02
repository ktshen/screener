# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Engines for executing functions."""

import time
import multiprocessing
import concurrent.futures
import gc

from numba.core.registry import CPUDispatcher

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import merge_dicts, Configured
from vectorbtpro.utils.pbar import get_pbar
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.template import CustomTemplate, substitute_templates

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from ray.remote_function import RemoteFunction as RemoteFunctionT
    from ray import ObjectRef as ObjectRefT
except ImportError:
    RemoteFunctionT = tp.Any
    ObjectRefT = tp.Any

__all__ = [
    "SerialEngine",
    "ThreadPoolEngine",
    "ProcessPoolEngine",
    "PathosEngine",
    "DaskEngine",
    "RayEngine",
    "execute",
]


class ExecutionEngine(Configured):
    """Abstract class for executing functions."""

    def __init__(self, **kwargs) -> None:
        Configured.__init__(self, **kwargs)

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        """Run an iterable of tuples out of a function, arguments, and keyword arguments.

        Provide `n_calls` in case `funcs_args` is a generator and the underlying engine needs it."""
        raise NotImplementedError


class SerialEngine(ExecutionEngine):
    """Class for executing functions sequentially.

    For defaults, see `engines.serial` in `vectorbtpro._settings.execution`."""

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (ExecutionEngine._expected_keys or set()) | {
        "show_progress",
        "progress_desc",
        "pbar_kwargs",
        "clear_cache",
        "collect_garbage",
        "cooldown",
    }

    def __init__(
        self,
        show_progress: tp.Optional[bool] = None,
        progress_desc: tp.Optional[tp.Sequence] = None,
        pbar_kwargs: tp.KwargsLike = None,
        clear_cache: tp.Union[None, bool, int] = None,
        collect_garbage: tp.Union[None, bool, int] = None,
        cooldown: tp.Optional[int] = None,
        **kwargs,
    ) -> None:
        from vectorbtpro._settings import settings

        serial_cfg = settings["execution"]["engines"]["serial"]

        if show_progress is None:
            show_progress = serial_cfg["show_progress"]
        pbar_kwargs = merge_dicts(pbar_kwargs, serial_cfg["pbar_kwargs"])
        if clear_cache is None:
            clear_cache = serial_cfg["clear_cache"]
        if collect_garbage is None:
            collect_garbage = serial_cfg["collect_garbage"]
        if cooldown is None:
            cooldown = serial_cfg["cooldown"]

        self._show_progress = show_progress
        self._progress_desc = progress_desc
        self._pbar_kwargs = pbar_kwargs
        self._clear_cache = clear_cache
        self._collect_garbage = collect_garbage
        self._cooldown = cooldown

        ExecutionEngine.__init__(
            self,
            show_progress=show_progress,
            progress_desc=progress_desc,
            pbar_kwargs=pbar_kwargs,
            clear_cache=clear_cache,
            collect_garbage=collect_garbage,
            cooldown=cooldown,
            **kwargs,
        )

    @property
    def show_progress(self) -> bool:
        """Whether to show the progress bar using `vectorbtpro.utils.pbar.get_pbar`."""
        return self._show_progress

    @property
    def progress_desc(self) -> tp.Optional[tp.Sequence]:
        """Sequence used to describe each iteration of the progress bar."""
        return self._progress_desc

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`."""
        return self._pbar_kwargs

    @property
    def clear_cache(self) -> tp.Union[bool, int]:
        """Whether to clear vectorbt's cache after each iteration.

        If integer, do it once a number of calls."""
        return self._clear_cache

    @property
    def collect_garbage(self) -> tp.Union[bool, int]:
        """Whether to clear garbage after each iteration.

        If integer, do it once a number of calls."""
        return self._collect_garbage

    @property
    def cooldown(self) -> tp.Optional[int]:
        """Number of seconds to sleep after each call."""
        return self._cooldown

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        from vectorbtpro.registries.ca_registry import CAQueryDelegator
        import time

        results = []
        if n_calls is None and hasattr(funcs_args, "__len__"):
            n_calls = len(funcs_args)
        with get_pbar(total=n_calls, show_progress=self.show_progress, **self.pbar_kwargs) as pbar:
            for i, (func, args, kwargs) in enumerate(funcs_args):
                if self.progress_desc is not None:
                    pbar.set_description(str(self.progress_desc[i]))
                results.append(func(*args, **kwargs))
                pbar.update(1)
                if isinstance(self.clear_cache, bool):
                    if self.clear_cache:
                        CAQueryDelegator().clear_cache()
                elif i > 0 and (i + 1) % self.clear_cache == 0:
                    CAQueryDelegator().clear_cache()
                if isinstance(self.collect_garbage, bool):
                    if self.collect_garbage:
                        gc.collect()
                elif i > 0 and (i + 1) % self.collect_garbage == 0:
                    gc.collect()
                if self.cooldown is not None:
                    time.sleep(self.cooldown)

        return results


class ThreadPoolEngine(ExecutionEngine):
    """Class for executing functions using `ThreadPoolExecutor` from `concurrent.futures`.

    For defaults, see `engines.threadpool` in `vectorbtpro._settings.execution`."""

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (ExecutionEngine._expected_keys or set()) | {
        "init_kwargs",
    }

    def __init__(self, init_kwargs: tp.KwargsLike = None, **kwargs) -> None:
        from vectorbtpro._settings import settings

        threadpool_cfg = settings["execution"]["engines"]["threadpool"]

        init_kwargs = merge_dicts(init_kwargs, threadpool_cfg["init_kwargs"])

        self._init_kwargs = init_kwargs

        ExecutionEngine.__init__(self, init_kwargs=init_kwargs, **kwargs)

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used to initialize `ThreadPoolExecutor`."""
        return self._init_kwargs

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        with concurrent.futures.ThreadPoolExecutor(**self.init_kwargs) as executor:
            futures = {executor.submit(func, *args, **kwargs): i for i, (func, args, kwargs) in enumerate(funcs_args)}
            results = [None] * len(futures)
            for fut in concurrent.futures.as_completed(futures):
                results[futures[fut]] = fut.result()
            return results


class ProcessPoolEngine(ExecutionEngine):
    """Class for executing functions using `ProcessPoolExecutor` from `concurrent.futures`.

    For defaults, see `engines.processpool` in `vectorbtpro._settings.execution`."""

    def __init__(self, init_kwargs: tp.KwargsLike = None) -> None:
        from vectorbtpro._settings import settings

        processpool_cfg = settings["execution"]["engines"]["processpool"]

        init_kwargs = merge_dicts(init_kwargs, processpool_cfg["init_kwargs"])

        self._init_kwargs = init_kwargs

        ExecutionEngine.__init__(self, init_kwargs=init_kwargs)

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used to initialize `ProcessPoolExecutor`."""
        return self._init_kwargs

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        with concurrent.futures.ProcessPoolExecutor(**self.init_kwargs) as executor:
            futures = {executor.submit(func, *args, **kwargs): i for i, (func, args, kwargs) in enumerate(funcs_args)}
            results = [None] * len(futures)
            for fut in concurrent.futures.as_completed(futures):
                results[futures[fut]] = fut.result()
            return results


def pass_kwargs_as_args(func, args, kwargs):
    """Helper function for `pathos.pools.ParallelPool`."""
    return func(*args, **kwargs)


class PathosEngine(ExecutionEngine):
    """Class for executing functions using `pathos`.

    For defaults, see `engines.pathos` in `vectorbtpro._settings.execution`."""

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (ExecutionEngine._expected_keys or set()) | {
        "pool_type",
        "sleep",
        "init_kwargs",
    }

    def __init__(
        self,
        pool_type: tp.Optional[str] = None,
        sleep: tp.Optional[int] = None,
        init_kwargs: tp.KwargsLike = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        from vectorbtpro._settings import settings

        pathos_cfg = settings["execution"]["engines"]["pathos"]

        if pool_type is None:
            pool_type = pathos_cfg["pool_type"]
        if sleep is None:
            sleep = pathos_cfg["sleep"]
        init_kwargs = merge_dicts(init_kwargs, pathos_cfg["init_kwargs"])
        if show_progress is None:
            show_progress = pathos_cfg["show_progress"]
        pbar_kwargs = merge_dicts(pbar_kwargs, pathos_cfg["pbar_kwargs"])

        self._pool_type = pool_type
        self._sleep = sleep
        self._init_kwargs = init_kwargs
        self._show_progress = show_progress
        self._pbar_kwargs = pbar_kwargs

        ExecutionEngine.__init__(self, init_kwargs=init_kwargs, **kwargs)

    @property
    def pool_type(self) -> str:
        """Pool type."""
        return self._pool_type

    @property
    def sleep(self) -> tp.Optional[int]:
        """Number of seconds between task checks.

        The higher, the less CPU it uses but also the more time it takes to gather the results.
        Thus, should be in a millisecond range."""
        return self._sleep

    @property
    def show_progress(self) -> bool:
        """Whether to show the progress bar using `vectorbtpro.utils.pbar.get_pbar`."""
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`."""
        return self._pbar_kwargs

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used to initialize the pool."""
        return self._init_kwargs

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("pathos")

        if self.pool_type.lower() in ("thread", "threadpool"):
            from pathos.pools import ThreadPool as Pool
        elif self.pool_type.lower() in ("process", "processpool"):
            from pathos.pools import ProcessPool as Pool
        elif self.pool_type.lower() in ("parallel", "parallelpool"):
            from pathos.pools import ParallelPool as Pool

            funcs_args = [(pass_kwargs_as_args, x, {}) for x in funcs_args]
        else:
            raise ValueError(f"Invalid option pool_type='{self.pool_type}'")

        if n_calls is None and hasattr(funcs_args, "__len__"):
            n_calls = len(funcs_args)

        with get_pbar(total=n_calls, show_progress=self.show_progress, **self.pbar_kwargs) as pbar:
            with Pool(**self.init_kwargs) as pool:
                futures = [pool.apipe(func, *args, **kwargs) for (func, args, kwargs) in funcs_args]
                tasks = set(futures)
                while tasks:
                    ready_tasks = {task for task in tasks if task.ready()}
                    if ready_tasks:
                        pbar.update(len(ready_tasks))
                        tasks -= ready_tasks
                    if self.sleep is not None:
                        time.sleep(self.sleep)
                return [f.get() for f in futures]


class DaskEngine(ExecutionEngine):
    """Class for executing functions in parallel using Dask.

    For defaults, see `engines.dask` in `vectorbtpro._settings.execution`.

    !!! note
        Use multi-threading mainly on numeric code that releases the GIL
        (like NumPy, Pandas, Scikit-Learn, Numba)."""

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (ExecutionEngine._expected_keys or set()) | {
        "compute_kwargs",
    }

    def __init__(self, compute_kwargs: tp.KwargsLike = None, **kwargs) -> None:
        from vectorbtpro._settings import settings

        dask_cfg = settings["execution"]["engines"]["dask"]

        compute_kwargs = merge_dicts(compute_kwargs, dask_cfg["compute_kwargs"])

        self._compute_kwargs = compute_kwargs

        ExecutionEngine.__init__(self, compute_kwargs=compute_kwargs, **kwargs)

    @property
    def compute_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `dask.compute`."""
        return self._compute_kwargs

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("dask")
        import dask

        results_delayed = []
        for func, args, kwargs in funcs_args:
            results_delayed.append(dask.delayed(func)(*args, **kwargs))
        return list(dask.compute(*results_delayed, **self.compute_kwargs))


class RayEngine(ExecutionEngine):
    """Class for executing functions in parallel using Ray.

    For defaults, see `engines.ray` in `vectorbtpro._settings.execution`.

    !!! note
        Ray spawns multiple processes as opposed to threads, so any argument and keyword argument must first
        be put into an object store to be shared. Make sure that the computation with `func` takes
        a considerable amount of time compared to this copying operation, otherwise there will be
        a little to no speedup."""

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (ExecutionEngine._expected_keys or set()) | {
        "restart",
        "reuse_refs",
        "del_refs",
        "shutdown",
        "init_kwargs",
        "remote_kwargs",
    }

    def __init__(
        self,
        restart: tp.Optional[bool] = None,
        reuse_refs: tp.Optional[bool] = None,
        del_refs: tp.Optional[bool] = None,
        shutdown: tp.Optional[bool] = None,
        init_kwargs: tp.KwargsLike = None,
        remote_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        from vectorbtpro._settings import settings

        ray_cfg = settings["execution"]["engines"]["ray"]

        if restart is None:
            restart = ray_cfg["restart"]
        if reuse_refs is None:
            reuse_refs = ray_cfg["reuse_refs"]
        if del_refs is None:
            del_refs = ray_cfg["del_refs"]
        if shutdown is None:
            shutdown = ray_cfg["shutdown"]
        init_kwargs = merge_dicts(init_kwargs, ray_cfg["init_kwargs"])
        remote_kwargs = merge_dicts(remote_kwargs, ray_cfg["remote_kwargs"])

        self._restart = restart
        self._reuse_refs = reuse_refs
        self._del_refs = del_refs
        self._shutdown = shutdown
        self._init_kwargs = init_kwargs
        self._remote_kwargs = remote_kwargs

        ExecutionEngine.__init__(
            self,
            restart=restart,
            reuse_refs=reuse_refs,
            del_refs=del_refs,
            shutdown=shutdown,
            init_kwargs=init_kwargs,
            remote_kwargs=remote_kwargs,
            **kwargs,
        )

    @property
    def restart(self) -> bool:
        """Whether to terminate the Ray runtime and initialize a new one."""
        return self._restart

    @property
    def reuse_refs(self) -> bool:
        """Whether to re-use function and object references, such that each unique object
        will be copied only once."""
        return self._reuse_refs

    @property
    def del_refs(self) -> bool:
        """Whether to explicitly delete the result object references."""
        return self._del_refs

    @property
    def shutdown(self) -> bool:
        """Whether to True to terminate the Ray runtime upon the job end."""
        return self._shutdown

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `ray.init`."""
        return self._init_kwargs

    @property
    def remote_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `ray.remote`."""
        return self._remote_kwargs

    @staticmethod
    def get_ray_refs(
        funcs_args: tp.FuncsArgs,
        reuse_refs: bool = True,
        remote_kwargs: tp.KwargsLike = None,
    ) -> tp.List[tp.Tuple[RemoteFunctionT, tp.Tuple[ObjectRefT, ...], tp.Dict[str, ObjectRefT]]]:
        """Get result references by putting each argument and keyword argument into the object store
        and invoking the remote decorator on each function using Ray.

        If `reuse_refs` is True, will generate one reference per unique object id."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ray")
        import ray
        from ray.remote_function import RemoteFunction
        from ray import ObjectRef

        if remote_kwargs is None:
            remote_kwargs = {}

        func_id_remotes = {}
        obj_id_refs = {}
        funcs_args_refs = []
        for func, args, kwargs in funcs_args:
            # Get remote function
            if isinstance(func, RemoteFunction):
                func_remote = func
            else:
                if not reuse_refs or id(func) not in func_id_remotes:
                    if isinstance(func, CPUDispatcher):
                        # Numba-wrapped function is not recognized by ray as a function
                        _func = lambda *_args, **_kwargs: func(*_args, **_kwargs)
                    else:
                        _func = func
                    if len(remote_kwargs) > 0:
                        func_remote = ray.remote(**remote_kwargs)(_func)
                    else:
                        func_remote = ray.remote(_func)
                    if reuse_refs:
                        func_id_remotes[id(func)] = func_remote
                else:
                    func_remote = func_id_remotes[id(func)]

            # Get id of each (unique) arg
            arg_refs = ()
            for arg in args:
                if isinstance(arg, ObjectRef):
                    arg_ref = arg
                else:
                    if not reuse_refs or id(arg) not in obj_id_refs:
                        arg_ref = ray.put(arg)
                        obj_id_refs[id(arg)] = arg_ref
                    else:
                        arg_ref = obj_id_refs[id(arg)]
                arg_refs += (arg_ref,)

            # Get id of each (unique) kwarg
            kwarg_refs = {}
            for kwarg_name, kwarg in kwargs.items():
                if isinstance(kwarg, ObjectRef):
                    kwarg_ref = kwarg
                else:
                    if not reuse_refs or id(kwarg) not in obj_id_refs:
                        kwarg_ref = ray.put(kwarg)
                        obj_id_refs[id(kwarg)] = kwarg_ref
                    else:
                        kwarg_ref = obj_id_refs[id(kwarg)]
                kwarg_refs[kwarg_name] = kwarg_ref

            funcs_args_refs.append((func_remote, arg_refs, kwarg_refs))
        return funcs_args_refs

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ray")
        import ray

        if self.restart:
            if ray.is_initialized():
                ray.shutdown()
        if not ray.is_initialized():
            ray.init(**self.init_kwargs)
        funcs_args_refs = self.get_ray_refs(funcs_args, reuse_refs=self.reuse_refs, remote_kwargs=self.remote_kwargs)
        result_refs = []
        for func_remote, arg_refs, kwarg_refs in funcs_args_refs:
            result_refs.append(func_remote.remote(*arg_refs, **kwarg_refs))
        try:
            results = ray.get(result_refs)
        finally:
            if self.del_refs:
                # clear object store
                del result_refs
            if self.shutdown:
                ray.shutdown()
        return results


def execute_serially(funcs_args: tp.FuncsArgs, id_objs: tp.Dict[int, tp.Any]) -> list:
    """Execute serially."""
    results = []
    for func, args, kwargs in funcs_args:
        new_func = id_objs[func]
        new_args = tuple(id_objs[arg] for arg in args)
        new_kwargs = {k: id_objs[v] for k, v in kwargs.items()}
        results.append(new_func(*new_args, **new_kwargs))
    return results


def build_serial_chunk(funcs_args: tp.FuncsArgs) -> tp.FuncArgs:
    """Build a serial chunk."""
    ref_ids = dict()
    id_objs = dict()

    def _prepare(x):
        if id(x) in ref_ids:
            return ref_ids[id(x)]
        new_id = len(id_objs)
        ref_ids[id(x)] = new_id
        id_objs[new_id] = x
        return new_id

    new_funcs_args = []
    for func, args, kwargs in funcs_args:
        new_func = _prepare(func)
        new_args = tuple(_prepare(arg) for arg in args)
        new_kwargs = {k: _prepare(v) for k, v in kwargs.items()}
        new_funcs_args.append((new_func, new_args, new_kwargs))
    return execute_serially, (new_funcs_args, id_objs), {}


def execute(
    funcs_args: tp.FuncsArgs,
    engine: tp.EngineLike = "serial",
    n_calls: tp.Optional[int] = None,
    n_chunks: tp.Optional[tp.Union[str, int]] = None,
    chunk_len: tp.Optional[tp.Union[str, int]] = None,
    chunk_meta: tp.Optional[tp.Iterable[tp.ChunkMeta]] = None,
    distribute: tp.Optional[str] = None,
    in_chunk_order: bool = False,
    show_progress: tp.Optional[bool] = None,
    progress_desc: tp.Optional[tp.Sequence] = None,
    pbar_kwargs: tp.KwargsLike = None,
    template_context: tp.KwargsLike = None,
    engine_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> list:
    """Execute using an engine.

    Supported values for `engine`:

    * Name of the engine (see supported engines)
    * Subclass of `ExecutionEngine` - initializes with `kwargs` and `engine_kwargs`
    * Instance of `ExecutionEngine` - calls `ExecutionEngine.execute` with `n_calls`
    * Callable - passes `funcs_args`, `n_calls` (if not None), and `kwargs` and `engine_kwargs`

    Can execute per chunk if `chunk_meta` is provided. Otherwise, if any of `n_chunks` and `chunk_len`
    are set, passes them to `vectorbtpro.utils.chunking.yield_chunk_meta` to generate `chunk_meta`.
    Arguments `n_chunks` and `chunk_len` can be set globally in the engine-specific settings.
    Set `n_chunks` and `chunk_len` to 'auto' to set them to the number of cores.

    If `distribute` is "calls", distributes calls within each chunk.
    If indices in `chunk_meta` are perfectly sorted and `funcs_args` is an iterable, iterates
    over `funcs_args` to avoid converting it into a list. Otherwise, iterates over `chunk_meta`.
    If `in_chunk_order` is True, returns the outputs in the order they appear in `chunk_meta`.
    Otherwise, always returns them in the same order as in `funcs_args`.

    If `distribute` is "chunks", distributes chunks. For this, executes calls
    within each chunk serially using `execute_serially`. Also, compresses each chunk such that
    each unique function, positional argument, and keyword argument is serialized only once.

    If `funcs_args` is a custom template, substitutes it once `chunk_meta` is established.
    Use `template_context` as an additional context. All the resolved functions and arguments
    will be immediately passed to the executor.

    !!! info
        Chunks are processed sequentially, while functions within each chunk can be processed distributively.

    Supported engines can be found in `engines` in `vectorbtpro._settings.execution`."""
    from vectorbtpro._settings import settings

    execution_cfg = settings["execution"]
    engines_cfg = execution_cfg["engines"]

    engine_kwargs = merge_dicts(kwargs, engine_kwargs)

    engine_cfg = dict()
    if isinstance(engine, str):
        if engine.lower() in engines_cfg:
            engine_cfg = engines_cfg[engine]
            engine = engines_cfg[engine]["cls"]
        else:
            raise ValueError(f"Invalid engine name '{engine}'")
    if isinstance(engine, type) and issubclass(engine, ExecutionEngine):
        for k, v in engines_cfg.items():
            if v["cls"] is engine:
                engine_cfg = v
        func_arg_names = get_func_arg_names(engine.__init__)
        if "show_progress" in func_arg_names and "show_progress" not in engine_kwargs:
            engine_kwargs["show_progress"] = show_progress
        if "progress_desc" in func_arg_names and "progress_desc" not in engine_kwargs:
            engine_kwargs["progress_desc"] = progress_desc
        if "pbar_kwargs" in func_arg_names and "pbar_kwargs" not in engine_kwargs:
            engine_kwargs["pbar_kwargs"] = pbar_kwargs
        engine = engine(**engine_kwargs)
    elif isinstance(engine, ExecutionEngine):
        for k, v in engines_cfg.items():
            if v["cls"] is type(engine):
                engine_cfg = v
    if callable(engine):
        func_arg_names = get_func_arg_names(engine)
        if "show_progress" in func_arg_names and "show_progress" not in engine_kwargs:
            engine_kwargs["show_progress"] = show_progress
        if "progress_desc" in func_arg_names and "progress_desc" not in engine_kwargs:
            engine_kwargs["progress_desc"] = progress_desc
        if "pbar_kwargs" in func_arg_names and "pbar_kwargs" not in engine_kwargs:
            engine_kwargs["pbar_kwargs"] = pbar_kwargs

    if n_chunks is None:
        n_chunks = engine_cfg.get("n_chunks", execution_cfg["n_chunks"])
    if chunk_len is None:
        chunk_len = engine_cfg.get("chunk_len", execution_cfg["chunk_len"])
    if distribute is None:
        distribute = engine_cfg.get("distribute", execution_cfg["distribute"])
    if show_progress is None:
        show_progress = engine_cfg.get("show_progress", execution_cfg["show_progress"])
    pbar_kwargs = merge_dicts(execution_cfg["pbar_kwargs"], engine_cfg.get("pbar_kwargs", None), pbar_kwargs)

    def _execute(_funcs_args, _n_calls):
        if isinstance(engine, ExecutionEngine):
            return engine.execute(_funcs_args, n_calls=_n_calls)
        if callable(engine):
            if "n_calls" in func_arg_names:
                return engine(_funcs_args, n_calls=_n_calls, **engine_kwargs)
            return engine(_funcs_args, **engine_kwargs)
        raise TypeError(f"Engine of type {type(engine)} is not supported")

    if n_chunks is None and chunk_len is None and chunk_meta is None:
        n_chunks = 1
    if n_chunks == 1 and not isinstance(funcs_args, CustomTemplate):
        return _execute(funcs_args, n_calls)

    if chunk_meta is None:
        # Generate chunk metadata
        from vectorbtpro.utils.chunking import yield_chunk_meta

        if not isinstance(funcs_args, CustomTemplate) and hasattr(funcs_args, "__len__"):
            _n_calls = len(funcs_args)
        elif n_calls is not None:
            _n_calls = n_calls
        else:
            if isinstance(funcs_args, CustomTemplate):
                raise ValueError("When funcs_args is a template, n_calls must be provided")
            funcs_args = list(funcs_args)
            _n_calls = len(funcs_args)
        if isinstance(n_chunks, str) and n_chunks.lower() == "auto":
            n_chunks = multiprocessing.cpu_count()
        if isinstance(chunk_len, str) and chunk_len.lower() == "auto":
            chunk_len = multiprocessing.cpu_count()
        chunk_meta = yield_chunk_meta(n_chunks=n_chunks, size=_n_calls, chunk_len=chunk_len)

    # Substitute templates
    if isinstance(funcs_args, CustomTemplate):
        template_context = merge_dicts(dict(chunk_meta=chunk_meta), template_context)
        funcs_args = substitute_templates(funcs_args, template_context, sub_id="funcs_args")
        if hasattr(funcs_args, "__len__"):
            n_calls = len(funcs_args)
        else:
            n_calls = None
        return _execute(funcs_args, n_calls)

    # Get indices of each chunk and whether they are sorted
    last_idx = -1
    indices_sorted = True
    all_chunk_indices = []
    for _chunk_meta in chunk_meta:
        if _chunk_meta.indices is not None:
            chunk_indices = list(_chunk_meta.indices)
        else:
            if _chunk_meta.start is None or _chunk_meta.end is None:
                raise ValueError("Each chunk must have a start and an end index")
            chunk_indices = list(range(_chunk_meta.start, _chunk_meta.end))
        if indices_sorted:
            for idx in chunk_indices:
                if idx != last_idx + 1:
                    indices_sorted = False
                    break
                last_idx = idx
        all_chunk_indices.append(chunk_indices)

    if distribute.lower() == "calls":
        if indices_sorted and not hasattr(funcs_args, "__len__"):
            # Iterate over funcs_args
            outputs = []
            chunk_idx = 0
            _funcs_args = []

            with get_pbar(total=len(all_chunk_indices), show_progress=show_progress, **pbar_kwargs) as pbar:
                for i, func_args in enumerate(funcs_args):
                    if i > all_chunk_indices[chunk_idx][-1]:
                        chunk_indices = all_chunk_indices[chunk_idx]
                        outputs.extend(_execute(_funcs_args, len(chunk_indices)))
                        chunk_idx += 1
                        _funcs_args = []
                        pbar.update(1)
                    _funcs_args.append(func_args)
                if len(_funcs_args) > 0:
                    chunk_indices = all_chunk_indices[chunk_idx]
                    outputs.extend(_execute(_funcs_args, len(chunk_indices)))
                    pbar.update(1)
            return outputs
        else:
            # Iterate over chunks
            funcs_args = list(funcs_args)
            outputs = []

            with get_pbar(total=len(all_chunk_indices), show_progress=show_progress, **pbar_kwargs) as pbar:
                for chunk_indices in all_chunk_indices:
                    _funcs_args = []
                    for idx in chunk_indices:
                        _funcs_args.append(funcs_args[idx])
                    chunk_output = _execute(_funcs_args, len(chunk_indices))
                    if in_chunk_order or indices_sorted:
                        outputs.extend(chunk_output)
                    else:
                        outputs.extend(zip(chunk_indices, chunk_output))
                    pbar.update(1)
            if in_chunk_order or indices_sorted:
                return outputs
            return list(list(zip(*sorted(outputs, key=lambda x: x[0])))[1])
    elif distribute.lower() == "chunks":
        if indices_sorted and not hasattr(funcs_args, "__len__"):
            # Iterate over funcs_args
            chunk_idx = 0
            _funcs_args = []
            funcs_args_chunks = []

            for i, func_args in enumerate(funcs_args):
                if i > all_chunk_indices[chunk_idx][-1]:
                    funcs_args_chunks.append(build_serial_chunk(_funcs_args))
                    chunk_idx += 1
                    _funcs_args = []
                _funcs_args.append(func_args)
            if len(_funcs_args) > 0:
                funcs_args_chunks.append(build_serial_chunk(_funcs_args))
            outputs = _execute(funcs_args_chunks, len(funcs_args_chunks))
            return [x for o in outputs for x in o]
        else:
            # Iterate over chunks
            funcs_args = list(funcs_args)
            funcs_args_chunks = []
            output_indices = []

            for chunk_indices in all_chunk_indices:
                _funcs_args = []
                for idx in chunk_indices:
                    _funcs_args.append(funcs_args[idx])
                funcs_args_chunks.append(build_serial_chunk(_funcs_args))
                output_indices.extend(chunk_indices)
            outputs = _execute(funcs_args_chunks, len(funcs_args_chunks))
            outputs = [x for o in outputs for x in o]
            if in_chunk_order or indices_sorted:
                return outputs
            return [x for _, x in sorted(zip(output_indices, outputs))]
    else:
        raise ValueError(f"Invalid option distribute='{distribute}'")
