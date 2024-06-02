# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Mixin for building statistics out of performance metrics."""

import inspect
import string
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.wrapping import Wrapping
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import get_dict_attr, AttrResolverMixin
from vectorbtpro.utils.config import merge_dicts, Config, HybridConfig
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.tagging import match_tags
from vectorbtpro.utils.template import substitute_templates, CustomTemplate

__all__ = []


class MetaStatsBuilderMixin(type):
    """Meta class that exposes a read-only class property `StatsBuilderMixin.metrics`."""

    @property
    def metrics(cls) -> Config:
        """Metrics supported by `StatsBuilderMixin.stats`."""
        return cls._metrics


class StatsBuilderMixin(metaclass=MetaStatsBuilderMixin):
    """Mixin that implements `StatsBuilderMixin.stats`.

    Required to be a subclass of `vectorbtpro.base.wrapping.Wrapping`."""

    _writeable_attrs: tp.ClassVar[tp.Optional[tp.Set[str]]] = {"_metrics"}

    def __init__(self) -> None:
        checks.assert_instance_of(self, Wrapping)

        # Copy writeable attrs
        self._metrics = type(self)._metrics.copy()

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `StatsBuilderMixin.stats`.

        See `vectorbtpro._settings.stats_builder`."""
        from vectorbtpro._settings import settings

        stats_builder_cfg = settings["stats_builder"]

        return merge_dicts(stats_builder_cfg, dict(settings=dict(freq=self.wrapper.freq)))

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start=dict(
                title="Start",
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None,
                tags="wrapper",
            ),
            end=dict(
                title="End",
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None,
                tags="wrapper",
            ),
            period=dict(
                title="Period",
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags="wrapper",
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        """Metrics supported by `${cls_name}`.

        ```python
        ${metrics}
        ```

        Returns `${cls_name}._metrics`, which gets (hybrid-) copied upon creation of each instance.
        Thus, changing this config won't affect the class.

        To change metrics, you can either change the config in-place, override this property,
        or overwrite the instance variable `${cls_name}._metrics`."""
        return self._metrics

    def stats(
        self,
        metrics: tp.Optional[tp.MaybeIterable[tp.Union[str, tp.Tuple[str, tp.Kwargs]]]] = None,
        tags: tp.Optional[tp.MaybeIterable[str]] = None,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        agg_func: tp.Optional[tp.Callable] = np.mean,
        dropna: tp.Optional[bool] = None,
        silence_warnings: tp.Optional[bool] = None,
        template_context: tp.Optional[tp.Mapping] = None,
        settings: tp.KwargsLike = None,
        filters: tp.KwargsLike = None,
        metric_settings: tp.KwargsLike = None,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Compute various metrics on this object.

        Args:
            metrics (str, tuple, iterable, or dict): Metrics to calculate.

                Each element can be either:

                * Metric name (see keys in `StatsBuilderMixin.metrics`)
                * Tuple of a metric name and a settings dict as in `StatsBuilderMixin.metrics`
                * Tuple of a metric name and a template of instance `vectorbtpro.utils.template.CustomTemplate`
                * Tuple of a metric name and a list of settings dicts to be expanded into multiple metrics

                The settings dict can contain the following keys:

                * `title`: Title of the metric. Defaults to the name.
                * `tags`: Single or multiple tags to associate this metric with.
                    If any of these tags is in `tags`, keeps this metric.
                * `check_{filter}` and `inv_check_{filter}`: Whether to check this metric against a
                    filter defined in `filters`. True (or False for inverse) means to keep this metric.
                * `calc_func` (required): Calculation function for custom metrics.
                    Must return either a scalar for one column/group, pd.Series for multiple columns/groups,
                    or a dict of such for multiple sub-metrics.
                * `resolve_calc_func`: whether to resolve `calc_func`. If the function can be accessed
                    by traversing attributes of this object, you can specify the path to this function
                    as a string (see `vectorbtpro.utils.attr_.deep_getattr` for the path format).
                    If `calc_func` is a function, arguments from merged metric settings are matched with
                    arguments in the signature (see below). If `resolve_calc_func` is False, `calc_func`
                    must accept (resolved) self and dictionary of merged metric settings.
                    Defaults to True.
                * `use_shortcuts`: Whether to use shortcut properties whenever possible when
                    resolving `calc_func`. Defaults to True.
                * `post_calc_func`: Function to post-process the result of `calc_func`.
                    Must accept (resolved) self, output of `calc_func`, and dictionary of merged metric settings,
                    and return whatever is acceptable to be returned by `calc_func`. Defaults to None.
                * `fill_wrap_kwargs`: Whether to fill `wrap_kwargs` with `to_timedelta` and `silence_warnings`.
                    Defaults to False.
                * `apply_to_timedelta`: Whether to apply `vectorbtpro.base.wrapping.ArrayWrapper.arr_to_timedelta`
                    on the result. To disable this globally, pass `to_timedelta=False` in `settings`.
                    Defaults to False.
                * `pass_{arg}`: Whether to pass any argument from the settings (see below). Defaults to True if
                    this argument was found in the function's signature. Set to False to not pass.
                    If argument to be passed was not found, `pass_{arg}` is removed.
                * `resolve_path_{arg}`: Whether to resolve an argument that is meant to be an attribute of
                    this object and is the first part of the path of `calc_func`. Passes only optional arguments.
                    Defaults to True. See `vectorbtpro.utils.attr_.AttrResolverMixin.resolve_attr`.
                * `resolve_{arg}`: Whether to resolve an argument that is meant to be an attribute of
                    this object and is present in the function's signature. Defaults to False.
                    See `vectorbtpro.utils.attr_.AttrResolverMixin.resolve_attr`.
                * `use_shortcuts_{arg}`: Whether to use shortcut properties whenever possible when resolving
                    an argument. Defaults to True.
                * `select_col_{arg}`: Whether to select the column from an argument that is meant to be
                    an attribute of this object. Defaults to False.
                * `template_context`: Mapping to replace templates in metric settings. Used across all settings.
                * Any other keyword argument that overrides the settings or is passed directly to `calc_func`.

                If `resolve_calc_func` is True, the calculation function may "request" any of the
                following arguments by accepting them or if `pass_{arg}` was found in the settings dict:

                * Each of `vectorbtpro.utils.attr_.AttrResolverMixin.self_aliases`: original object
                    (ungrouped, with no column selected)
                * `group_by`: won't be passed if it was used in resolving the first attribute of `calc_func`
                    specified as a path, use `pass_group_by=True` to pass anyway
                * `column`
                * `metric_name`
                * `agg_func`
                * `silence_warnings`
                * `to_timedelta`: replaced by True if None and frequency is set
                * Any argument from `settings`
                * Any attribute of this object if it meant to be resolved
                    (see `vectorbtpro.utils.attr_.AttrResolverMixin.resolve_attr`)

                Pass `metrics='all'` to calculate all supported metrics.
            tags (str or iterable): Tags to select.

                See `vectorbtpro.utils.tagging.match_tags`.
            column (str): Name of the column/group.

                !!! hint
                    There are two ways to select a column: `obj['a'].stats()` and `obj.stats(column='a')`.
                    They both accomplish the same thing but in different ways: `obj['a'].stats()` computes
                    statistics of the column 'a' only, while `obj.stats(column='a')` computes statistics of
                    all columns first and only then selects the column 'a'. The first method is preferred
                    when you have a lot of data or caching is disabled. The second method is preferred when
                    most attributes have already been cached.
            group_by (any): Group or ungroup columns. See `vectorbtpro.base.grouping.base.Grouper`.
            agg_func (callable): Aggregation function to aggregate statistics across all columns.
                By default, takes the mean of all columns. If None, returns all columns as a DataFrame.

                Must take `pd.Series` and return a const.

                Takes effect if `column` was specified or this object contains only one column of data.

                If `agg_func` has been overridden by a metric:

                * Takes effect if global `agg_func` is not None
                * Raises a warning if it's None but the result of calculation has multiple values
            dropna (bool): Whether to hide metrics that are all NaN.
            silence_warnings (bool): Whether to silence all warnings.
            template_context (mapping): Global context to replace templates.

                Gets merged over `template_context` from `StatsBuilderMixin.stats_defaults`.

                Applied on `settings` and then on each metric settings.
            filters (dict): Filters to apply.

                Each item consists of the filter name and settings dict.

                The settings dict can contain the following keys:

                * `filter_func`: Filter function that must accept resolved self and
                    merged settings for a metric, and return either True or False.
                * `warning_message`: Warning message to be shown when skipping a metric.
                    Can be a template that will be substituted using merged metric settings as context.
                    Defaults to None.
                * `inv_warning_message`: Same as `warning_message` but for inverse checks.

                Gets merged over `filters` from `StatsBuilderMixin.stats_defaults`.
            settings (dict): Global settings and resolution arguments.

                Extends/overrides `settings` from `StatsBuilderMixin.stats_defaults`.
                Gets extended/overridden by metric settings.
            metric_settings (dict): Keyword arguments for each metric.

                Extends/overrides all global and metric settings.

        For template logic, see `vectorbtpro.utils.template`.

        For defaults, see `StatsBuilderMixin.stats_defaults`.

        !!! hint
            There are two types of arguments: optional (or resolution) and mandatory arguments.
            Optional arguments are only passed if they are found in the function's signature.
            Mandatory arguments are passed regardless of this. Optional arguments can only be defined
            using `settings` (that is, globally), while mandatory arguments can be defined both using
            default metric settings and `{metric_name}_kwargs`. Overriding optional arguments using default
            metric settings or `{metric_name}_kwargs` won't turn them into mandatory. For this, pass `pass_{arg}=True`.

        !!! hint
            Make sure to resolve and then to re-use as many object attributes as possible to
            utilize built-in caching (even if global caching is disabled).

        Usage:
            See `vectorbtpro.portfolio.base`.
        """
        # Resolve defaults
        if dropna is None:
            dropna = self.stats_defaults.get("dropna", False)
        if silence_warnings is None:
            silence_warnings = self.stats_defaults.get("silence_warnings", False)
        template_context = merge_dicts(
            self.stats_defaults.get("template_context", {}),
            template_context,
        )
        filters = merge_dicts(self.stats_defaults.get("filters", {}), filters)
        settings = merge_dicts(self.stats_defaults.get("settings", {}), settings)
        metric_settings = merge_dicts(
            self.stats_defaults.get("metric_settings", {}),
            metric_settings,
        )

        # Replace templates globally (not used at metric level)
        if len(template_context) > 0:
            sub_settings = substitute_templates(
                settings,
                context=template_context,
                sub_id="sub_settings",
                strict=False,
            )
        else:
            sub_settings = settings

        # Resolve self
        reself = self.resolve_self(
            cond_kwargs=sub_settings,
            impacts_caching=False,
            silence_warnings=silence_warnings,
        )

        # Prepare metrics
        if metrics is None:
            metrics = reself.stats_defaults.get("metrics", "all")
        if metrics == "all":
            metrics = reself.metrics
        if isinstance(metrics, dict):
            metrics = list(metrics.items())
        if isinstance(metrics, (str, tuple)):
            metrics = [metrics]

        # Prepare tags
        if tags is None:
            tags = reself.stats_defaults.get("tags", "all")
        if isinstance(tags, str) and tags == "all":
            tags = None
        if isinstance(tags, (str, tuple)):
            tags = [tags]

        # Bring to the same shape
        new_metrics = []
        for i, metric in enumerate(metrics):
            if isinstance(metric, str):
                metric = (metric, reself.metrics[metric])
            if not isinstance(metric, tuple):
                raise TypeError(f"Metric at index {i} must be either a string or a tuple")
            new_metrics.append(metric)
        metrics = new_metrics

        # Expand metrics
        new_metrics = []
        for i, (metric_name, _metric_settings) in enumerate(metrics):
            if isinstance(_metric_settings, CustomTemplate):
                metric_context = merge_dicts(
                    template_context,
                    {name: reself for name in reself.self_aliases},
                    dict(
                        column=column,
                        group_by=group_by,
                        metric_name=metric_name,
                        agg_func=agg_func,
                        silence_warnings=silence_warnings,
                        to_timedelta=None,
                    ),
                    settings,
                )
                metric_context = substitute_templates(
                    metric_context,
                    context=metric_context,
                    sub_id="metric_context",
                )
                _metric_settings = _metric_settings.substitute(
                    context=metric_context,
                    strict=True,
                    sub_id="metric",
                )
            if isinstance(_metric_settings, list):
                for __metric_settings in _metric_settings:
                    new_metrics.append((metric_name, __metric_settings))
            else:
                new_metrics.append((metric_name, _metric_settings))
        metrics = new_metrics

        # Handle duplicate names
        metric_counts = Counter(list(map(lambda x: x[0], metrics)))
        metric_i = {k: -1 for k in metric_counts.keys()}
        metrics_dct = {}
        for i, (metric_name, _metric_settings) in enumerate(metrics):
            if metric_counts[metric_name] > 1:
                metric_i[metric_name] += 1
                metric_name = metric_name + "_" + str(metric_i[metric_name])
            metrics_dct[metric_name] = _metric_settings

        # Check metric_settings
        missed_keys = set(metric_settings.keys()).difference(set(metrics_dct.keys()))
        if len(missed_keys) > 0:
            raise ValueError(
                f"Keys {missed_keys} in metric_settings could not be matched with any metric"
            )

        # Merge settings
        opt_arg_names_dct = {}
        custom_arg_names_dct = {}
        resolved_self_dct = {}
        context_dct = {}
        for metric_name, _metric_settings in list(metrics_dct.items()):
            opt_settings = merge_dicts(
                {name: reself for name in reself.self_aliases},
                dict(
                    column=column,
                    group_by=group_by,
                    metric_name=metric_name,
                    agg_func=agg_func,
                    silence_warnings=silence_warnings,
                    to_timedelta=None,
                ),
                settings,
            )
            _metric_settings = _metric_settings.copy()
            passed_metric_settings = metric_settings.get(metric_name, {})
            merged_settings = merge_dicts(opt_settings, _metric_settings, passed_metric_settings)
            metric_template_context = merged_settings.pop("template_context", {})
            template_context_merged = merge_dicts(template_context, metric_template_context)
            template_context_merged = substitute_templates(
                template_context_merged,
                context=merged_settings,
                sub_id="template_context_merged",
            )
            context = merge_dicts(template_context_merged, merged_settings)
            merged_settings = substitute_templates(
                merged_settings,
                context=context,
                sub_id="merged_settings",
            )

            # Filter by tag
            if tags is not None:
                in_tags = merged_settings.get("tags", None)
                if in_tags is None or not match_tags(tags, in_tags):
                    metrics_dct.pop(metric_name, None)
                    continue

            custom_arg_names = set(_metric_settings.keys()).union(
                set(passed_metric_settings.keys())
            )
            opt_arg_names = set(opt_settings.keys())
            custom_reself = reself.resolve_self(
                cond_kwargs=merged_settings,
                custom_arg_names=custom_arg_names,
                impacts_caching=True,
                silence_warnings=merged_settings["silence_warnings"],
            )

            metrics_dct[metric_name] = merged_settings
            custom_arg_names_dct[metric_name] = custom_arg_names
            opt_arg_names_dct[metric_name] = opt_arg_names
            resolved_self_dct[metric_name] = custom_reself
            context_dct[metric_name] = context

        # Filter metrics
        for metric_name, _metric_settings in list(metrics_dct.items()):
            custom_reself = resolved_self_dct[metric_name]
            context = context_dct[metric_name]
            _silence_warnings = _metric_settings.get("silence_warnings")

            metric_filters = set()
            for k in _metric_settings.keys():
                filter_name = None
                if k.startswith("check_"):
                    filter_name = k[len("check_") :]
                elif k.startswith("inv_check_"):
                    filter_name = k[len("inv_check_") :]
                if filter_name is not None:
                    if filter_name not in filters:
                        raise ValueError(f"Metric '{metric_name}' requires filter '{filter_name}'")
                    metric_filters.add(filter_name)

            for filter_name in metric_filters:
                filter_settings = filters[filter_name]
                _filter_settings = substitute_templates(
                    filter_settings,
                    context=context,
                    sub_id="filter_settings",
                )
                filter_func = _filter_settings["filter_func"]
                warning_message = _filter_settings.get("warning_message", None)
                inv_warning_message = _filter_settings.get("inv_warning_message", None)
                to_check = _metric_settings.get("check_" + filter_name, False)
                inv_to_check = _metric_settings.get("inv_check_" + filter_name, False)

                if to_check or inv_to_check:
                    whether_true = filter_func(custom_reself, _metric_settings)
                    to_remove = (to_check and not whether_true) or (inv_to_check and whether_true)
                    if to_remove:
                        if to_check and warning_message is not None and not _silence_warnings:
                            warnings.warn(warning_message)
                        if (
                            inv_to_check
                            and inv_warning_message is not None
                            and not _silence_warnings
                        ):
                            warnings.warn(inv_warning_message)

                        metrics_dct.pop(metric_name, None)
                        custom_arg_names_dct.pop(metric_name, None)
                        opt_arg_names_dct.pop(metric_name, None)
                        resolved_self_dct.pop(metric_name, None)
                        context_dct.pop(metric_name, None)
                        break

        # Any metrics left?
        if len(metrics_dct) == 0:
            if not silence_warnings:
                warnings.warn("No metrics to calculate", stacklevel=2)
            return None

        # Compute stats
        arg_cache_dct = {}
        stats_dct = {}
        used_agg_func = False
        for i, (metric_name, _metric_settings) in enumerate(metrics_dct.items()):
            try:
                final_kwargs = _metric_settings.copy()
                opt_arg_names = opt_arg_names_dct[metric_name]
                custom_arg_names = custom_arg_names_dct[metric_name]
                custom_reself = resolved_self_dct[metric_name]

                # Clean up keys
                for k, v in list(final_kwargs.items()):
                    if k.startswith("check_") or k.startswith("inv_check_") or k in ("tags",):
                        final_kwargs.pop(k, None)

                # Get metric-specific values
                _column = final_kwargs.get("column")
                _group_by = final_kwargs.get("group_by")
                _agg_func = final_kwargs.get("agg_func")
                _silence_warnings = final_kwargs.get("silence_warnings")
                if final_kwargs["to_timedelta"] is None:
                    final_kwargs["to_timedelta"] = custom_reself.wrapper.freq is not None
                to_timedelta = final_kwargs.get("to_timedelta")
                title = final_kwargs.pop("title", metric_name)
                calc_func = final_kwargs.pop("calc_func")
                resolve_calc_func = final_kwargs.pop("resolve_calc_func", True)
                post_calc_func = final_kwargs.pop("post_calc_func", None)
                use_shortcuts = final_kwargs.pop("use_shortcuts", True)
                use_caching = final_kwargs.pop("use_caching", True)
                fill_wrap_kwargs = final_kwargs.pop("fill_wrap_kwargs", False)
                if fill_wrap_kwargs:
                    final_kwargs["wrap_kwargs"] = merge_dicts(
                        dict(to_timedelta=to_timedelta, silence_warnings=_silence_warnings),
                        final_kwargs.get("wrap_kwargs", None),
                    )
                apply_to_timedelta = final_kwargs.pop("apply_to_timedelta", False)

                # Resolve calc_func
                if resolve_calc_func:
                    if not callable(calc_func):
                        passed_kwargs_out = {}

                        def _getattr_func(
                            obj: tp.Any,
                            attr: str,
                            args: tp.ArgsLike = None,
                            kwargs: tp.KwargsLike = None,
                            call_attr: bool = True,
                            _final_kwargs: tp.Kwargs = final_kwargs,
                            _opt_arg_names: tp.Set[str] = opt_arg_names,
                            _custom_arg_names: tp.Set[str] = custom_arg_names,
                            _arg_cache_dct: tp.Kwargs = arg_cache_dct,
                            _use_shortcuts: bool = use_shortcuts,
                            _use_caching: bool = use_caching,
                        ) -> tp.Any:
                            if attr in _final_kwargs:
                                return _final_kwargs[attr]
                            if args is None:
                                args = ()
                            if kwargs is None:
                                kwargs = {}

                            if obj is custom_reself:
                                resolve_path_arg = _final_kwargs.pop("resolve_path_" + attr, True)
                                if resolve_path_arg:
                                    if call_attr:
                                        cond_kwargs = {
                                            k: v
                                            for k, v in _final_kwargs.items()
                                            if k in _opt_arg_names
                                        }
                                        out = custom_reself.resolve_attr(
                                            attr,  # do not pass _attr, important for caching
                                            args=args,
                                            cond_kwargs=cond_kwargs,
                                            kwargs=kwargs,
                                            custom_arg_names=_custom_arg_names,
                                            cache_dct=_arg_cache_dct,
                                            use_caching=_use_caching,
                                            passed_kwargs_out=passed_kwargs_out,
                                            use_shortcuts=_use_shortcuts,
                                        )
                                    else:
                                        if isinstance(obj, AttrResolverMixin):
                                            cls_dir = obj.cls_dir
                                        else:
                                            cls_dir = dir(type(obj))
                                        if "get_" + attr in cls_dir:
                                            _attr = "get_" + attr
                                        else:
                                            _attr = attr
                                        out = getattr(obj, _attr)
                                    _select_col_arg = _final_kwargs.pop("select_col_" + attr, False)
                                    if _select_col_arg and _column is not None:
                                        out = custom_reself.select_col_from_obj(
                                            out,
                                            _column,
                                            wrapper=custom_reself.wrapper.regroup(_group_by),
                                        )
                                        passed_kwargs_out["group_by"] = _group_by
                                        passed_kwargs_out["column"] = _column
                                    return out

                            out = getattr(obj, attr)
                            if callable(out) and call_attr:
                                return out(*args, **kwargs)
                            return out

                        calc_func = custom_reself.deep_getattr(
                            calc_func,
                            getattr_func=_getattr_func,
                            call_last_attr=False,
                        )

                        if "group_by" in passed_kwargs_out:
                            if "pass_group_by" not in final_kwargs:
                                final_kwargs.pop("group_by", None)
                        if "column" in passed_kwargs_out:
                            if "pass_column" not in final_kwargs:
                                final_kwargs.pop("column", None)

                    # Resolve arguments
                    if callable(calc_func):
                        func_arg_names = get_func_arg_names(calc_func)
                        for k in func_arg_names:
                            if k not in final_kwargs:
                                resolve_arg = final_kwargs.pop("resolve_" + k, False)
                                use_shortcuts_arg = final_kwargs.pop("use_shortcuts_" + k, True)
                                select_col_arg = final_kwargs.pop("select_col_" + k, False)
                                if resolve_arg:
                                    try:
                                        arg_out = custom_reself.resolve_attr(
                                            k,
                                            cond_kwargs=final_kwargs,
                                            custom_arg_names=custom_arg_names,
                                            cache_dct=arg_cache_dct,
                                            use_caching=use_caching,
                                            use_shortcuts=use_shortcuts_arg,
                                        )
                                    except AttributeError:
                                        continue

                                    if select_col_arg and _column is not None:
                                        arg_out = custom_reself.select_col_from_obj(
                                            arg_out,
                                            _column,
                                            wrapper=custom_reself.wrapper.regroup(_group_by),
                                        )
                                    final_kwargs[k] = arg_out
                        for k in list(final_kwargs.keys()):
                            if k in opt_arg_names:
                                if "pass_" + k in final_kwargs:
                                    if not final_kwargs.get("pass_" + k):  # first priority
                                        final_kwargs.pop(k, None)
                                elif k not in func_arg_names:  # second priority
                                    final_kwargs.pop(k, None)
                        for k in list(final_kwargs.keys()):
                            if k.startswith("pass_") or k.startswith("resolve_"):
                                final_kwargs.pop(k, None)  # cleanup

                        # Call calc_func
                        out = calc_func(**final_kwargs)
                    else:
                        # calc_func is already a result
                        out = calc_func
                else:
                    # Do not resolve calc_func
                    out = calc_func(custom_reself, _metric_settings)

                # Call post_calc_func
                if post_calc_func is not None:
                    out = post_calc_func(custom_reself, out, _metric_settings)

                # Post-process and store the metric
                multiple = True
                if not isinstance(out, dict):
                    multiple = False
                    out = {None: out}
                for k, v in out.items():
                    # Resolve title
                    if multiple:
                        if title is None:
                            t = str(k)
                        else:
                            t = title + ": " + str(k)
                    else:
                        t = title

                    # Check result type
                    if checks.is_any_array(v) and not checks.is_series(v):
                        raise TypeError(
                            "calc_func must return either a scalar for one column/group, "
                            "pd.Series for multiple columns/groups, or a dict of such. "
                            f"Not {type(v)}."
                        )

                    # Handle apply_to_timedelta
                    if apply_to_timedelta and to_timedelta:
                        v = custom_reself.wrapper.arr_to_timedelta(
                            v,
                            silence_warnings=_silence_warnings,
                        )

                    # Select column or aggregate
                    if checks.is_series(v):
                        if _column is None and v.shape[0] == 1:
                            v = v.iloc[0]
                        elif _column is not None:
                            v = custom_reself.select_col_from_obj(
                                v,
                                _column,
                                wrapper=custom_reself.wrapper.regroup(_group_by),
                            )
                        elif _agg_func is not None and agg_func is not None:
                            v = _agg_func(v)
                            if _agg_func is agg_func:
                                used_agg_func = True
                        elif _agg_func is None and agg_func is not None:
                            if not _silence_warnings:
                                warnings.warn(
                                    f"Metric '{metric_name}' returned multiple values "
                                    "despite having no aggregation function",
                                    stacklevel=2,
                                )
                            continue

                    # Store metric
                    if t in stats_dct:
                        if not _silence_warnings:
                            warnings.warn(f"Duplicate metric title '{t}'", stacklevel=2)
                    stats_dct[t] = v
            except Exception as e:
                warnings.warn(f"Metric '{metric_name}' raised an exception", stacklevel=2)
                raise e

        # Return the stats
        if reself.wrapper.get_ndim(group_by=group_by) == 1:
            sr = pd.Series(
                stats_dct,
                name=reself.wrapper.get_name(group_by=group_by),
                dtype=object,
            )
            if dropna:
                sr.replace([np.inf, -np.inf], np.nan, inplace=True)
                return sr.dropna()
            return sr
        if column is not None:
            sr = pd.Series(stats_dct, name=column, dtype=object)
            if dropna:
                sr.replace([np.inf, -np.inf], np.nan, inplace=True)
                return sr.dropna()
            return sr
        if agg_func is not None:
            if used_agg_func and not silence_warnings:
                warnings.warn(
                    f"Object has multiple columns. Aggregated some metrics using {agg_func}. "
                    "Pass column to select a single column/group.",
                    stacklevel=2,
                )
            sr = pd.Series(stats_dct, name="agg_stats", dtype=object)
            if dropna:
                sr.replace([np.inf, -np.inf], np.nan, inplace=True)
                return sr.dropna()
            return sr
        new_index = reself.wrapper.grouper.get_index(group_by=group_by)
        df = pd.DataFrame(stats_dct, index=new_index)
        if dropna:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df.dropna(axis=1, how="all")
        return df

    # ############# Docs ############# #

    @classmethod
    def build_metrics_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """Build metrics documentation."""
        if source_cls is None:
            source_cls = StatsBuilderMixin
        return string.Template(
            inspect.cleandoc(get_dict_attr(source_cls, "metrics").__doc__),
        ).substitute(
            {"metrics": cls.metrics.prettify(), "cls_name": cls.__name__},
        )

    @classmethod
    def override_metrics_doc(cls, __pdoc__: dict, source_cls: tp.Optional[type] = None) -> None:
        """Call this method on each subclass that overrides `StatsBuilderMixin.metrics`."""
        __pdoc__[cls.__name__ + ".metrics"] = cls.build_metrics_doc(source_cls=source_cls)


__pdoc__ = dict()
StatsBuilderMixin.override_metrics_doc(__pdoc__)
