# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Global settings of vectorbtpro.

`settings` config is also accessible via `vectorbtpro.settings`.

!!! note
    All places in vectorbt import `vectorbtpro._settings.settings`, not `vectorbtpro.settings`.
    Overwriting `vectorbtpro.settings` only overwrites the reference created for the user.
    Consider updating the settings config instead of replacing it.

Here are the main properties of the `settings` config:

* It's a nested config, that is, a config that consists of multiple sub-configs.
    one per sub-package (e.g., 'data'), module (e.g., 'wrapping'), or even class (e.g., 'configured').
    Each sub-config may consist of other sub-configs.
* It has frozen keys - you cannot add other sub-configs or remove the existing ones, but you can modify them.
* Each sub-config can either inherit the properties of the parent one by being an instance of
    `vectorbtpro.utils.config.child_dict` or overwrite them by being an instance of
    `vectorbtpro.utils.config.Config` or a regular `dict`. The main reason for defining an own config
    is to allow adding new keys (e.g., 'plotting.layout').

For example, you can change default width and height of each plot:

```pycon
>>> import vectorbtpro as vbt

>>> vbt.settings['plotting']['layout']['width'] = 800
>>> vbt.settings['plotting']['layout']['height'] = 400
```

The main sub-configs such as for plotting can be also accessed/modified using the dot notation:

```
>>> vbt.settings.plotting['layout']['width'] = 800
```

Some sub-configs allow the dot notation too but this depends whether they inherit the rules of the root config.

```plaintext
>>> vbt.settings.data - ok
>>> vbt.settings.data.binance - ok
>>> vbt.settings.data.binance.api_key - error
>>> vbt.settings.data.binance['api_key'] - ok
```

Since this is only visible when looking at the source code, the advice is to always use the bracket notation.

!!! note
    Whether the change takes effect immediately depends upon the place that accesses the settings.
    For example, changing 'wrapping.freq` has an immediate effect because the value is resolved
    every time `vectorbtpro.base.wrapping.ArrayWrapper.freq` is called. On the other hand, changing
    'portfolio.fillna_close' has only effect on `vectorbtpro.portfolio.base.Portfolio` instances created
    in the future, not the existing ones, because the value is resolved upon the object's construction.
    Moreover, some settings are only accessed when importing the package for the first time,
    such as 'jitting.jit_decorator'. In any case, make sure to check whether the update actually took place.

## Saving and loading

Like any other class subclassing `vectorbtpro.utils.config.Config`, we can persist settings to the disk,
load it back, and replace in-place. There are several ways of how to update the settings.

### Binary file

Pickling will dump the entire settings object into a byte stream and save as a binary file.
Supported file extensions are "pickle" (default) and "pkl".

```pycon
>>> vbt.settings.save('my_settings')
>>> vbt.settings['caching']['disable'] = True
>>> vbt.settings['caching']['disable']
True

>>> vbt.settings.load_update('my_settings', clear=True)  # replace in-place
>>> vbt.settings['caching']['disable']
False
```

!!! note
    Argument `clear=True` will replace the entire settings object. Disable it to apply
    only a subset of settings (default).

### Config file

We can also encode the settings object into a config and save as a text file that can be edited
easily. Supported file extensions are "config" (default), "cfg", and "ini".

```pycon
>>> vbt.settings.save('my_settings', file_format="config")
>>> vbt.settings['caching']['disable'] = True
>>> vbt.settings['caching']['disable']
True

>>> vbt.settings.load_update('my_settings', file_format="config", clear=True)  # replace in-place
>>> vbt.settings['caching']['disable']
False
```

### On import

Some settings (such as Numba-related ones) are applied only on import, so changing them during the runtime
will have no effect. In this case, change the settings, save them to the disk, and then either
rename the file to "vbt" (with extension) and place it in the working directory for it to be
recognized automatically, or create an environment variable "VBT_SETTINGS_PATH" that holds the full path
to the file - vectorbt will load it before any other module. You can also change the recognized file
name using an environment variable "VBT_SETTINGS_NAME", which defaults to "vbt".

!!! note
    Environment variables must be set before importing vectorbtpro.

For example, to set the default theme to dark, create the following "vbt.ini" file:

```ini
[plotting]
default_theme = dark
```
"""

import json
import os
import pkgutil

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.utils.checks import is_instance_of
from vectorbtpro.utils.config import child_dict, Config
from vectorbtpro.utils.execution import (
    SerialEngine,
    ThreadPoolEngine,
    ProcessPoolEngine,
    PathosEngine,
    DaskEngine,
    RayEngine
)
from vectorbtpro.utils.jitting import NumPyJitter, NumbaJitter
from vectorbtpro.utils.template import Sub, RepEval, substitute_templates

__all__ = [
    "settings",
]

__pdoc__: dict = {}

# ############# Settings sub-configs ############# #

_settings = {}

importing = child_dict(
    auto_import=True,
    plotly=True,
    telegram=True,
    quantstats=True,
    sklearn=True,
)
"""_"""

__pdoc__["importing"] = Sub(
    """Sub-config with settings applied on importing.
    
Disabling these options will make vectorbt load faster, but will limit the flexibility of accessing
various features of the package.
    
!!! note
    If `auto_import` is False, you won't be able to access most important modules and objects 
    such as via `vbt.Portfolio`, only by explicitly importing them such as via 
    `from vectorbtpro.portfolio.base import Portfolio`.

```python
${config_doc}
```"""
)

_settings["importing"] = importing

caching = child_dict(
    disable=False,
    disable_whitelist=False,
    disable_machinery=False,
    silence_warnings=False,
    register_lazily=True,
    ignore_args=["jitted", "chunked"],
    use_cached_accessors=True,
)
"""_"""

__pdoc__["caching"] = Sub(
    """Sub-config with settings applied across `vectorbtpro.registries.ca_registry`, 
`vectorbtpro.utils.caching`, and cacheable decorators in `vectorbtpro.utils.decorators`.

!!! hint
    Apply setting `register_lazily` on startup to register all unbound cacheables.
    
    Setting `use_cached_accessors` is applied only on import.

```python
${config_doc}
```"""
)

_settings["caching"] = caching

jitting = child_dict(
    disable=False,
    disable_wrapping=False,
    disable_resolution=False,
    option=True,
    allow_new=False,
    register_new=False,
    jitters=Config(
        nb=Config(
            cls=NumbaJitter,
            aliases={"numba"},
            options=dict(),
            override_options=dict(),
            resolve_kwargs=dict(),
            tasks=dict(),
        ),
        np=Config(
            cls=NumPyJitter,
            aliases={"numpy"},
            options=dict(),
            override_options=dict(),
            resolve_kwargs=dict(),
            tasks=dict(),
        ),
    ),
    template_context=Config(),
)
"""_"""

__pdoc__["jitting"] = Sub(
    """Sub-config with settings applied across `vectorbtpro.registries.jit_registry` and 
`vectorbtpro.utils.jitting`.

!!! note
    Options (with `_options` suffix) are applied only on import. 
    
    Keyword arguments (with `_kwargs` suffix) are applied right away.

```python
${config_doc}
```"""
)

_settings["jitting"] = jitting

numba = child_dict(
    parallel=None,
    silence_warnings=False,
    check_func_type=True,
    check_func_suffix=False,
)
"""_"""

__pdoc__["numba"] = Sub(
    """Sub-config with Numba-related settings.

```python
${config_doc}
```"""
)

_settings["numba"] = numba

math = child_dict(
    use_tol=True,
    rel_tol=1e-9,  # 1,000,000,000 == 1,000,000,001
    abs_tol=1e-12,  # 0.000000000001 == 0.000000000002
    use_round=True,
    decimals=12,  # 0.0000000000004 -> 0.0, # 0.0000000000006 -> 0.000000000001
)
"""_"""

__pdoc__["math"] = Sub(
    """Sub-config with settings applied across `vectorbtpro.utils.math_`.

!!! note
    All math settings are applied only on import.

```python
${config_doc}
```"""
)

_settings["math"] = math

execution = child_dict(
    n_chunks=None,
    chunk_len=None,
    distribute="calls",
    show_progress=True,
    pbar_kwargs=Config(),
    engines=Config(
        serial=Config(
            cls=SerialEngine,
            show_progress=False,
            pbar_kwargs=Config(),
            clear_cache=False,
            collect_garbage=False,
            cooldown=None,
        ),
        threadpool=Config(
            cls=ThreadPoolEngine,
            init_kwargs=Config(),
        ),
        processpool=Config(
            cls=ProcessPoolEngine,
            init_kwargs=Config(),
        ),
        pathos=Config(
            cls=PathosEngine,
            pool_type="process",
            sleep=0.001,
            init_kwargs=Config(),
            show_progress=False,
            pbar_kwargs=Config(),
        ),
        dask=Config(
            cls=DaskEngine,
            compute_kwargs=Config(),
        ),
        ray=Config(
            cls=RayEngine,
            restart=False,
            reuse_refs=True,
            del_refs=True,
            shutdown=False,
            init_kwargs=Config(),
            remote_kwargs=Config(),
        ),
    ),
)
"""_"""

__pdoc__["execution"] = Sub(
    """Sub-config with settings applied across `vectorbtpro.utils.execution`.

```python
${config_doc}
```"""
)

_settings["execution"] = execution

chunking = child_dict(
    disable=False,
    disable_wrapping=False,
    option=False,
    n_chunks=None,
    min_size=None,
    chunk_len=None,
    skip_one_chunk=True,
    silence_warnings=False,
    template_context=Config(),
    options=Config(),
    override_setup_options=Config(),
    override_options=Config(),
)
"""_"""

__pdoc__["chunking"] = Sub(
    """Sub-config with settings applied across `vectorbtpro.registries.ch_registry` 
and `vectorbtpro.utils.chunking`.

!!! note
    Options (with `_options` suffix) and setting `disable_machinery` are applied only on import.

```python
${config_doc}
```"""
)

_settings["chunking"] = chunking

params = child_dict(
    search_except_types=None,
    search_max_len=None,
    search_max_depth=None,
    skip_single_param=True,
    template_context=Config(),
    random_subset=None,
    seed=None,
    index_stack_kwargs=Config(),
    name_tuple_to_str=True,
    execute_kwargs=Config(),
)
"""_"""

__pdoc__["params"] = Sub(
    """Sub-config with settings applied across `vectorbtpro.utils.params`.

```python
${config_doc}
```"""
)

_settings["params"] = params

template = child_dict(
    strict=True,
    except_types=(list, set, frozenset),
    max_len=None,
    max_depth=None,
    context=Config(),
)
"""_"""

__pdoc__["template"] = Sub(
    """Sub-config with settings applied across `vectorbtpro.utils.template`.

```python
${config_doc}
```"""
)

_settings["template"] = template

pickling = child_dict(
    pickle_classes=None,
    file_format="pickle",
    compression=None,
    extensions=child_dict(
        serialization=child_dict(
            pickle={"pickle", "pkl", "p"},
            config={"config", "cfg", "ini"},
        ),
        compression=child_dict(
            bz2={"bzip2", "bz2", "bz"},
            gzip={"gzip", "gz"},
            lzma={"lzma", "xz"},
            lz4={"lz4"},
            blosc={"blosc"},
        ),
    )
)
"""_"""

__pdoc__["pickling"] = Sub(
    """Sub-config with settings applied to `vectorbtpro.utils.pickling`.

```python
${config_doc}
```"""
)

_settings["pickling"] = pickling

config = child_dict(
    options=Config(),
)
"""_"""

__pdoc__["config"] = Sub(
    """Sub-config with settings applied to `vectorbtpro.utils.config.Config`.

```python
${config_doc}
```"""
)

_settings["config"] = config

configured = child_dict(
    check_expected_keys_=True,
    config=child_dict(
        options=dict(
            readonly=True,
            nested=False,
        )
    ),
)
"""_"""

__pdoc__["configured"] = Sub(
    """Sub-config with settings applied to `vectorbtpro.utils.config.Configured`.

```python
${config_doc}
```"""
)

_settings["configured"] = configured

broadcasting = child_dict(
    align_index=True,
    align_columns=True,
    index_from="strict",
    columns_from="stack",
    ignore_sr_names=True,
    check_index_names=True,
    drop_duplicates=True,
    keep="last",
    drop_redundant=True,
    ignore_ranges=True,
    keep_wrap_default=False,
    keep_flex=False,
    min_ndim=None,
    expand_axis=1,
    index_to_param=True,
)
"""_"""

__pdoc__["broadcasting"] = Sub(
    """Sub-config with settings applied to broadcasting functions across `vectorbtpro.base`.

```python
${config_doc}
```"""
)

_settings["broadcasting"] = broadcasting

indexing = child_dict(
    rotate_rows=False,
    rotate_cols=False,
)
"""_"""

__pdoc__["indexing"] = Sub(
    """Sub-config with settings applied to indexing functions across `vectorbtpro.base`.
    
!!! note
    Options `rotate_rows` and `rotate_cols` are applied only on import. 

```python
${config_doc}
```"""
)

_settings["indexing"] = indexing

wrapping = child_dict(
    column_only_select=False,
    range_only_select=False,
    group_select=True,
    freq=None,
    silence_warnings=False,
    zero_to_none=True,
    min_precision=None,
    max_precision=None,
    prec_float_only=True,
    prec_check_bounds=True,
    prec_strict=True,
)
"""_"""

__pdoc__["wrapping"] = Sub(
    """Sub-config with settings applied across `vectorbtpro.base.wrapping`.

```python
${config_doc}
```

When enabling `max_precision` and running your code for the first time, make sure to enable 
`prec_check_bounds`. After that, you can safely disable it to slightly increase performance."""
)

_settings["wrapping"] = wrapping

resampling = child_dict(
    silence_warnings=False,
)
"""_"""

__pdoc__["resampling"] = Sub(
    """Sub-config with settings applied across `vectorbtpro.base.resampling`.

```python
${config_doc}
```"""
)

_settings["resampling"] = resampling

datetime = child_dict(
    naive_tz="tzlocal()",
    to_fixed_offset=None,
    parse_index=False,
    parser_kwargs=Config(),
)
"""_"""

__pdoc__["datetime"] = Sub(
    """Sub-config with settings applied across `vectorbtpro.utils.datetime_`.

```python
${config_doc}
```"""
)

_settings["datetime"] = datetime

data = child_dict(
    wrapper_kwargs=Config(),
    skip_on_error=False,
    silence_warnings=False,
    execute_kwargs=Config(),
    tz_localize="UTC",
    tz_convert="UTC",
    missing_index="nan",
    missing_columns="raise",
    custom=Config(
        # Synthetic
        synthetic=Config(
            start=None,
            end=None,
            freq=None,
            tz=None,
            normalize=False,
            inclusive="left",
        ),
        random=Config(
            start_value=100.0,
            mean=0.0,
            std=0.01,
            symmetric=False,
            seed=None,
        ),
        random_ohlc=Config(
            std=0.001,
            n_ticks=50,
        ),
        gbm=Config(
            start_value=100.0,
            mean=0.0,
            std=0.01,
            dt=1.0,
            seed=None,
        ),
        gbm_ohlc=Config(
            std=0.001,
            n_ticks=50,
        ),
        # Local
        local=Config(),
        # File
        file=Config(
            match_paths=True,
            match_regex=None,
            sort_paths=True,
        ),
        csv=Config(
            start=None,
            end=None,
            tz=None,
            start_row=None,
            end_row=None,
            header=0,
            index_col=0,
            parse_dates=True,
            squeeze=True,
            read_csv_kwargs=dict(),
        ),
        hdf=Config(
            start=None,
            end=None,
            tz=None,
            start_row=None,
            end_row=None,
            read_hdf_kwargs=dict(),
        ),
        # Remote
        remote=Config(),
        yf=Config(
            period="max",
            start=None,
            end=None,
            timeframe="1d",
            tz=None,
            history_kwargs=dict(),
        ),
        binance=Config(
            client=None,
            client_config=dict(
                api_key=None,
                api_secret=None,
            ),
            start=0,
            end="now",
            timeframe="1d",
            tz="UTC",
            klines_type="spot",
            limit=1000,
            delay=500,
            show_progress=True,
            pbar_kwargs=dict(),
            silence_warnings=False,
            get_klines_kwargs=dict(),
        ),
        ccxt=Config(
            exchange="binance",
            exchange_config=dict(
                enableRateLimit=True,
            ),
            start=None,
            end=None,
            timeframe="1d",
            tz="UTC",
            find_earliest_date=False,
            limit=1000,
            delay=None,
            retries=3,
            show_progress=True,
            pbar_kwargs=dict(),
            fetch_params=dict(),
            exchanges=dict(),
            silence_warnings=False,
        ),
        alpaca=Config(
            client=None,
            client_type="stocks",
            client_config=dict(
                api_key=None,
                secret_key=None,
                oauth_token=None,
                paper=False,
            ),
            start=0,
            end="now",
            timeframe="1d",
            tz="UTC",
            adjustment="raw",
            feed=None,
            limit=None,
        ),
        polygon=Config(
            client=None,
            client_config=dict(
                api_key=None,
            ),
            start=0,
            end="now",
            timeframe="1d",
            tz="UTC",
            adjusted=True,
            limit=50000,
            params=dict(),
            delay=500,
            retries=3,
            show_progress=True,
            pbar_kwargs=dict(),
            silence_warnings=False,
        ),
        av=Config(
            apikey=None,
            api_meta=None,
            category=None,
            function=None,
            timeframe=None,
            tz="UTC",
            adjusted=False,
            extended=False,
            slice="year1month1",
            series_type="close",
            time_period=10,
            outputsize="full",
            read_csv_kwargs=dict(
                index_col=0,
                parse_dates=True,
                infer_datetime_format=True,
            ),
            match_params=True,
            params=dict(),
            silence_warnings=False,
        ),
        ndl=Config(
            api_key=None,
            start=None,
            end=None,
            tz="UTC",
            column_indices=None,
            collapse=None,
            transform=None,
            params=dict(),
        ),
        tv=Config(
            client=None,
            client_config=dict(
                username=None,
                password=None,
                user_agent=None,
                token=None,
            ),
            exchange=None,
            timeframe="D",
            tz="UTC",
            fut_contract=None,
            adjustment="splits",
            extended_session=False,
            pro_data=True,
            limit=20000,
            delay=None,
            show_progress=True,
            pbar_kwargs=Config(),
        ),
    ),
    stats=Config(),
    plots=Config(),
)
"""_"""

__pdoc__["data"] = Sub(
    """Sub-config with settings applied across `vectorbtpro.data`.

```python
${config_doc}
```

Binance:
    See `binance.client.Client`.

CCXT:
    See [Configuring API Keys](https://ccxt.readthedocs.io/en/latest/manual.html#configuring-api-keys).
    Keys can be defined per exchange. If a key is defined at the root, it applies to all exchanges.
    
Alpaca:
    Sign up for Alpaca API keys under https://app.alpaca.markets/signup.
"""
)

_settings["data"] = data

plotting = child_dict(
    use_widgets=True,
    use_resampler=False,
    show_kwargs=Config(),
    use_gl=False,
    color_schema=Config(
        increasing="#26a69a",
        decreasing="#ee534f",
        lightblue="#6ca6cd",
        lightpurple="#6c76cd",
        lightpink="#cd6ca6",
    ),
    contrast_color_schema=Config(
        blue="#4285F4",
        orange="#FFAA00",
        green="#37B13F",
        red="#EA4335",
        gray="#E2E2E2",
        purple="#A661D5",
        pink="#DD59AA"
    ),
    themes=child_dict(
        light=child_dict(
            color_schema=Config(
                blue="#1f77b4",
                orange="#ff7f0e",
                green="#2ca02c",
                red="#dc3912",
                purple="#9467bd",
                brown="#8c564b",
                pink="#e377c2",
                gray="#7f7f7f",
                yellow="#bcbd22",
                cyan="#17becf",
            ),
            path="__name__/templates/light.json",
        ),
        dark=child_dict(
            color_schema=Config(
                blue="#1f77b4",
                orange="#ff7f0e",
                green="#2ca02c",
                red="#dc3912",
                purple="#9467bd",
                brown="#8c564b",
                pink="#e377c2",
                gray="#7f7f7f",
                yellow="#bcbd22",
                cyan="#17becf",
            ),
            path="__name__/templates/dark.json",
        ),
        seaborn=child_dict(
            color_schema=Config(
                blue="rgb(76,114,176)",
                orange="rgb(221,132,82)",
                green="rgb(85,168,104)",
                red="rgb(196,78,82)",
                purple="rgb(129,114,179)",
                brown="rgb(147,120,96)",
                pink="rgb(218,139,195)",
                gray="rgb(140,140,140)",
                yellow="rgb(204,185,116)",
                cyan="rgb(100,181,205)",
            ),
            path="__name__/templates/seaborn.json",
        ),
    ),
    default_theme="light",
    layout=Config(
        width=700,
        height=350,
        margin=dict(
            t=30,
            b=30,
            l=30,
            r=30,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            traceorder="normal",
        ),
    ),
)
"""_"""

__pdoc__["plotting"] = Sub(
    """Sub-config with settings applied to Plotly figures 
created from `vectorbtpro.utils.figure`.

```python
${config_doc}
```
"""
)

_settings["plotting"] = plotting

stats_builder = child_dict(
    metrics="all",
    tags="all",
    dropna=False,
    silence_warnings=False,
    template_context=Config(),
    filters=Config(
        is_not_grouped=dict(
            filter_func=lambda self, metric_settings: not self.wrapper.grouper.is_grouped(
                group_by=metric_settings["group_by"]
            ),
            warning_message=Sub("Metric '$metric_name' does not support grouped data"),
        ),
        has_freq=dict(
            filter_func=lambda self, metric_settings: self.wrapper.freq is not None,
            warning_message=Sub("Metric '$metric_name' requires frequency to be set"),
        ),
    ),
    settings=Config(
        to_timedelta=None,
        use_caching=True,
    ),
    metric_settings=Config(),
)
"""_"""

__pdoc__["stats_builder"] = Sub(
    """Sub-config with settings applied to 
`vectorbtpro.generic.stats_builder.StatsBuilderMixin`.

```python
${config_doc}
```"""
)

_settings["stats_builder"] = stats_builder

plots_builder = child_dict(
    subplots="all",
    tags="all",
    silence_warnings=False,
    template_context=Config(),
    filters=Config(
        is_not_grouped=dict(
            filter_func=lambda self, subplot_settings: not self.wrapper.grouper.is_grouped(
                group_by=subplot_settings["group_by"]
            ),
            warning_message=Sub("Subplot '$subplot_name' does not support grouped data"),
        ),
        has_freq=dict(
            filter_func=lambda self, subplot_settings: self.wrapper.freq is not None,
            warning_message=Sub("Subplot '$subplot_name' requires frequency to be set"),
        ),
    ),
    settings=Config(
        use_caching=True,
        hline_shape_kwargs=dict(
            type="line",
            line=dict(
                color="gray",
                dash="dash",
            ),
        ),
    ),
    subplot_settings=Config(),
    show_titles=True,
    hide_id_labels=True,
    group_id_labels=True,
    make_subplots_kwargs=Config(),
    layout_kwargs=Config(),
)
"""_"""

__pdoc__["plots_builder"] = Sub(
    """Sub-config with settings applied to 
`vectorbtpro.generic.plots_builder.PlotsBuilderMixin`.

```python
${config_doc}
```"""
)

_settings["plots_builder"] = plots_builder

generic = child_dict(
    use_jitted=False,
    stats=Config(
        filters=dict(
            has_mapping=dict(
                filter_func=lambda self, metric_settings: metric_settings.get(
                    "mapping",
                    self.mapping,
                )
                is not None,
            )
        ),
        settings=dict(
            incl_all_keys=False,
        ),
    ),
    plots=Config(),
)
"""_"""

__pdoc__["generic"] = Sub(
    """Sub-config with settings applied to `vectorbtpro.generic.accessors.GenericAccessor`.

```python
${config_doc}
```"""
)

_settings["generic"] = generic

ranges = child_dict(
    stats=Config(),
    plots=Config(),
)
"""_"""

__pdoc__["ranges"] = Sub(
    """Sub-config with settings applied to `vectorbtpro.generic.ranges.Ranges`.

```python
${config_doc}
```"""
)

_settings["ranges"] = ranges

splitter = child_dict(
    stats=Config(
        settings=dict(normalize=True),
        filters=dict(
            has_multiple_sets=dict(
                filter_func=lambda self, metric_settings: self.get_n_sets(
                    set_group_by=metric_settings.get("set_group_by", None)
                ) > 1,
            ),
            normalize=dict(
                filter_func=lambda self, metric_settings: metric_settings["normalize"],
            ),
        ),
    ),
    plots=Config(),
)
"""_"""

__pdoc__["splitter"] = Sub(
    """Sub-config with settings applied to `vectorbtpro.generic.splitting.base.Splitter`.

```python
${config_doc}
```"""
)

_settings["splitter"] = splitter

drawdowns = child_dict(
    stats=Config(
        settings=dict(
            incl_active=False,
        ),
    ),
    plots=Config(),
)
"""_"""

__pdoc__["drawdowns"] = Sub(
    """Sub-config with settings applied to `vectorbtpro.generic.drawdowns.Drawdowns`.

```python
${config_doc}
```"""
)

_settings["drawdowns"] = drawdowns

ohlcv = child_dict(
    ohlc_type="candlestick",
    column_names=child_dict(
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
    ),
    stats=Config(),
    plots=Config(),
)
"""_"""

__pdoc__["ohlcv"] = Sub(
    """Sub-config with settings applied across `vectorbtpro.ohlcv`.

```python
${config_doc}
```"""
)

_settings["ohlcv"] = ohlcv

signals = child_dict(
    stats=Config(
        filters=dict(
            silent_has_other=dict(
                filter_func=lambda self, metric_settings: metric_settings.get("other", None) is not None,
            ),
        ),
        settings=dict(
            other=None,
            other_name="Other",
            from_other=False,
        ),
    ),
    plots=Config(),
)
"""_"""

__pdoc__["signals"] = Sub(
    """Sub-config with settings applied to `vectorbtpro.signals.accessors.SignalsAccessor`.

```python
${config_doc}
```"""
)

_settings["signals"] = signals

returns = child_dict(
    year_freq="365 days",
    bm_returns=None,
    defaults=Config(
        start_value=1.0,
        window=10,
        minp=None,
        ddof=1,
        risk_free=0.0,
        levy_alpha=2.0,
        required_return=0.0,
        cutoff=0.05,
    ),
    stats=Config(
        filters=dict(
            has_year_freq=dict(
                filter_func=lambda self, metric_settings: self.year_freq is not None,
                warning_message=Sub("Metric '$metric_name' requires year frequency to be set"),
            ),
            has_bm_returns=dict(
                filter_func=lambda self, metric_settings: metric_settings.get(
                    "bm_returns",
                    self.bm_returns,
                )
                is not None,
                warning_message=Sub("Metric '$metric_name' requires bm_returns to be set"),
            ),
        ),
        settings=dict(
            check_is_not_grouped=True,
        ),
    ),
    plots=Config(),
)
"""_"""

__pdoc__["returns"] = Sub(
    """Sub-config with settings applied to `vectorbtpro.returns.accessors.ReturnsAccessor`.

```python
${config_doc}
```"""
)

_settings["returns"] = returns

qs_adapter = child_dict(
    defaults=Config(),
)
"""_"""

__pdoc__["qs_adapter"] = Sub(
    """Sub-config with settings applied to `vectorbtpro.returns.qs_adapter.QSAdapter`.

```python
${config_doc}
```"""
)

_settings["qs_adapter"] = qs_adapter

records = child_dict(
    stats=Config(),
    plots=Config(),
)
"""_"""

__pdoc__["records"] = Sub(
    """Sub-config with settings applied to `vectorbtpro.records.base.Records`.

```python
${config_doc}
```"""
)

_settings["records"] = records

mapped_array = child_dict(
    stats=Config(
        filters=dict(
            has_mapping=dict(
                filter_func=lambda self, metric_settings: metric_settings.get(
                    "mapping",
                    self.mapping,
                )
                is not None,
            )
        ),
        settings=dict(
            incl_all_keys=False,
        ),
    ),
    plots=Config(),
)
"""_"""

__pdoc__["mapped_array"] = Sub(
    """Sub-config with settings applied to `vectorbtpro.records.mapped_array.MappedArray`.

```python
${config_doc}
```"""
)

_settings["mapped_array"] = mapped_array

orders = child_dict(
    stats=Config(),
    plots=Config(),
)
"""_"""

__pdoc__["orders"] = Sub(
    """Sub-config with settings applied to `vectorbtpro.portfolio.orders.Orders`.

```python
${config_doc}
```"""
)

_settings["orders"] = orders

trades = child_dict(
    stats=Config(
        settings=dict(
            incl_open=False,
        ),
        template_context=dict(incl_open_tags=RepEval("['open', 'closed'] if incl_open else ['closed']")),
    ),
    plots=Config(),
)
"""_"""

__pdoc__["trades"] = Sub(
    """Sub-config with settings applied to `vectorbtpro.portfolio.trades.Trades`.

```python
${config_doc}
```"""
)

_settings["trades"] = trades

logs = child_dict(
    stats=Config(),
)
"""_"""

__pdoc__["logs"] = Sub(
    """Sub-config with settings applied to `vectorbtpro.portfolio.logs.Logs`.

```python
${config_doc}
```"""
)

_settings["logs"] = logs

portfolio = child_dict(
    # Setup
    data=None,
    open=None,
    high=None,
    low=None,
    close=None,
    bm_close=None,
    val_price="price",
    init_cash=100.0,
    init_position=0.0,
    init_price=np.nan,
    cash_deposits=0.0,
    cash_earnings=0.0,
    cash_dividends=0.0,
    cash_sharing=False,
    ffill_val_price=True,
    update_value=False,
    save_state=False,
    save_value=False,
    save_returns=False,
    fill_pos_info=True,
    track_value=True,
    row_wise=False,
    seed=None,
    group_by=None,
    broadcast_named_args=None,
    broadcast_kwargs=Config(
        require_kwargs=dict(requirements="W"),
    ),
    template_context=Config(),
    keep_inout_flex=True,
    from_ago=None,
    call_seq=None,
    attach_call_seq=False,
    max_orders=None,
    max_logs=None,
    jitted=None,
    chunked=None,
    staticized=False,
    records=None,
    # Orders
    size=np.inf,
    size_type="amount",
    direction="both",
    price="close",
    fees=0.0,
    fixed_fees=0.0,
    slippage=0.0,
    min_size=np.nan,
    max_size=np.nan,
    size_granularity=np.nan,
    leverage=1.0,
    leverage_mode="lazy",
    reject_prob=0.0,
    price_area_vio_mode="ignore",
    allow_partial=True,
    raise_reject=False,
    log=False,
    from_orders=Config(),
    # Signals
    from_signals=Config(
        direction="longonly",
        adjust_func_nb=None,
        adjust_args=(),
        signal_func_nb=None,
        signal_args=None,
        post_segment_func_nb=None,
        post_segment_args=(),
        order_mode=False,
        accumulate=False,
        upon_long_conflict="ignore",
        upon_short_conflict="ignore",
        upon_dir_conflict="ignore",
        upon_opposite_entry="reversereduce",
        order_type="market",
        limit_reverse=False,
        limit_delta=np.nan,
        limit_tif=-1,
        limit_expiry=-1,
        upon_adj_limit_conflict="keepignore",
        upon_opp_limit_conflict="cancelexecute",
        use_stops=None,
        stop_ladder="disabled",
        sl_stop=np.nan,
        tsl_th=np.nan,
        tsl_stop=np.nan,
        tp_stop=np.nan,
        td_stop=-1,
        dt_stop=-1,
        stop_entry_price="close",
        stop_exit_price="stop",
        stop_order_type="market",
        stop_limit_delta=np.nan,
        stop_exit_type="close",
        upon_stop_update="override",
        upon_adj_stop_conflict="keepexecute",
        upon_opp_stop_conflict="keepexecute",
        delta_format="percent",
        time_delta_format="index",
    ),
    # Holding
    hold_direction="longonly",
    close_at_end=False,
    # Order function
    from_order_func=Config(
        segment_mask=True,
        call_pre_segment=False,
        call_post_segment=False,
        pre_sim_func_nb=None,
        pre_sim_args=(),
        post_sim_func_nb=None,
        post_sim_args=(),
        pre_group_func_nb=None,
        pre_group_args=(),
        post_group_func_nb=None,
        post_group_args=(),
        pre_row_func_nb=None,
        pre_row_args=(),
        post_row_func_nb=None,
        post_row_args=(),
        pre_segment_func_nb=None,
        pre_segment_args=(),
        post_segment_func_nb=None,
        post_segment_args=(),
        order_func_nb=None,
        order_args=(),
        flex_order_func_nb=None,
        flex_order_args=(),
        post_order_func_nb=None,
        post_order_args=(),
        row_wise=False,
    ),
    from_def_order_func=Config(
        flexible=False,
    ),
    # Portfolio
    freq=None,
    use_in_outputs=True,
    fillna_close=True,
    trades_type="exittrades",
    stats=Config(
        filters=dict(
            has_year_freq=dict(
                filter_func=lambda self, metric_settings: metric_settings.get("year_freq", None) is not None,
                warning_message=Sub("Metric '$metric_name' requires year frequency to be set"),
            ),
            has_bm_returns=dict(
                filter_func=lambda self, metric_settings: metric_settings.get(
                    "bm_returns",
                    self.bm_returns,
                )
                is not None,
                warning_message=Sub("Metric '$metric_name' requires bm_returns to be set"),
            ),
            has_cash_deposits=dict(
                filter_func=lambda self, metric_settings: self._cash_deposits.size > 1
                or self._cash_deposits.item() != 0,
            ),
            has_cash_earnings=dict(
                filter_func=lambda self, metric_settings: self._cash_earnings.size > 1
                or self._cash_earnings.item() != 0,
            ),
        ),
        settings=dict(
            use_asset_returns=False,
            incl_open=False,
        ),
        template_context=dict(incl_open_tags=RepEval("['open', 'closed'] if incl_open else ['closed']")),
    ),
    plots=Config(
        subplots=["orders", "trade_pnl", "cum_returns"],
        settings=dict(
            use_asset_returns=False,
        ),
    ),
)
"""_"""

__pdoc__["portfolio"] = Sub(
    """Sub-config with settings applied to `vectorbtpro.portfolio.base.Portfolio`.

```python
${config_doc}
```"""
)

_settings["portfolio"] = portfolio

pfopt = child_dict(
    pypfopt=Config(
        target="max_sharpe",
        target_is_convex=True,
        weights_sum_to_one=True,
        target_constraints=None,
        target_solver="SLSQP",
        target_initial_guess=None,
        objectives=None,
        constraints=None,
        sector_mapper=None,
        sector_lower=None,
        sector_upper=None,
        discrete_allocation=False,
        allocation_method="lp_portfolio",
        silence_warnings=True,
        ignore_opt_errors=True,
    ),
    riskfolio=Config(
        nan_to_zero=True,
        dropna_rows=True,
        dropna_cols=True,
        dropna_any=True,
        factors=None,
        port=None,
        port_cls=None,
        opt_method=None,
        stats_methods=None,
        model=None,
        asset_classes=None,
        constraints_method=None,
        constraints=None,
        views_method=None,
        views=None,
        solvers=None,
        sol_params=None,
        freq=None,
        year_freq=None,
        pre_opt=False,
        pre_opt_kwargs=Config(),
        pre_opt_as_w=False,
        func_kwargs=Config(),
        silence_warnings=True,
        return_port=False,
    ),
    stats=Config(
        filters=dict(
            alloc_ranges=dict(
                filter_func=lambda self, metric_settings: is_instance_of(self.alloc_records, "AllocRanges"),
            )
        )
    ),
    plots=Config(
        filters=dict(
            alloc_ranges=dict(
                filter_func=lambda self, metric_settings: is_instance_of(self.alloc_records, "AllocRanges"),
            )
        )
    ),
)
"""_"""

__pdoc__["pfopt"] = Sub(
    """Sub-config with settings applied across `vectorbtpro.portfolio.pfopt`.

```python
${config_doc}
```"""
)

_settings["pfopt"] = pfopt

messaging = child_dict(
    telegram=Config(
        token=None,
        use_context=True,
        persistence=True,
        defaults=Config(),
        drop_pending_updates=True,
    ),
    giphy=child_dict(
        api_key=None,
        weirdness=5,
    ),
)
"""_"""

__pdoc__["messaging"] = Sub(
    """Sub-config with settings applied across `vectorbtpro.messaging`.

```python
${config_doc}
```

python-telegram-bot:
    Sub-config with settings applied to 
    [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot).
    
    Set `persistence` to string to use as `filename` in `telegram.ext.PicklePersistence`.
    For `defaults`, see `telegram.ext.Defaults`. Other settings will be distributed across 
    `telegram.ext.Updater` and `telegram.ext.updater.Updater.start_polling`.

GIPHY:
    Sub-config with settings applied to 
    [GIPHY Translate Endpoint](https://developers.giphy.com/docs/api/endpoint#translate).
"""
)

_settings["messaging"] = messaging

pbar = child_dict(
    disable=False,
    type="tqdm_auto",
    kwargs=Config(),
)
"""_"""

__pdoc__["pbar"] = Sub(
    """Sub-config with settings applied across `vectorbtpro.utils.pbar`.

```python
${config_doc}
```"""
)

_settings["pbar"] = pbar

path = child_dict(
    mkdir=child_dict(
        mkdir=False,
        mode=0o777,
        parents=True,
        exist_ok=True,
    ),
)
"""_"""

__pdoc__["path"] = Sub(
    """Sub-config with settings applied across `vectorbtpro.utils.path_`.

```python
${config_doc}
```"""
)

_settings["path"] = path


# ############# Settings config ############# #


class SettingsConfig(Config):
    """Extends `vectorbtpro.utils.config.Config` for global settings."""

    def register_template(self, theme: str) -> None:
        """Register template of a theme."""
        import plotly.io as pio
        import plotly.graph_objects as go

        template_path = self["plotting"]["themes"][theme]["path"]
        if template_path is None:
            raise ValueError(f"Must provide template path for the theme '{theme}'")
        if template_path.startswith("__name__/"):
            template_path = template_path.replace("__name__/", "")
            template = Config(json.loads(pkgutil.get_data(__name__, template_path)))
        else:
            with open(template_path, "r") as f:
                template = Config(json.load(f))
        pio.templates["vbt_" + theme] = go.layout.Template(template)

    def register_templates(self) -> None:
        """Register templates of all themes."""
        for theme in self["plotting"]["themes"]:
            self.register_template(theme)

    def set_theme(self, theme: str) -> None:
        """Set default theme."""
        self.register_template(theme)
        self["plotting"]["color_schema"].update(self["plotting"]["themes"][theme]["color_schema"])
        self["plotting"]["layout"]["template"] = "vbt_" + theme

    def reset_theme(self) -> None:
        """Reset to default theme."""
        self.set_theme(self["plotting"]["default_theme"])

    def substitute_sub_config_docs(self, __pdoc__: dict, prettify_kwargs: tp.KwargsLike = None) -> None:
        """Substitute templates in sub-config docs."""
        if prettify_kwargs is None:
            prettify_kwargs = {}
        for k, v in __pdoc__.items():
            if k in self:
                config_doc = self[k].prettify(**prettify_kwargs.get(k, {}))
                __pdoc__[k] = substitute_templates(
                    v,
                    context=dict(config_doc=config_doc),
                    sub_id="__pdoc__",
                )


settings = SettingsConfig(
    _settings,
    options_=dict(
        reset_dct_copy_kwargs=dict(copy_mode="deep"),
        frozen_keys=True,
        convert_children=Config,
        as_attrs=True,
    )
)
"""Global settings config.

Combines all sub-configs defined in this module."""

settings_name = os.environ.get("VBT_SETTINGS_NAME", "vbt")
if "VBT_SETTINGS_PATH" in os.environ:
    if len(os.environ["VBT_SETTINGS_PATH"]) > 0:
        settings.load_update(os.environ["VBT_SETTINGS_PATH"])
elif settings.file_exists(settings_name):
    settings.load_update(settings_name)

try:
    settings.reset_theme()
    settings.register_templates()
except ImportError:
    pass

settings.make_checkpoint()

settings.substitute_sub_config_docs(__pdoc__)
