# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

import importlib
import pkgutil
import typing

if typing.TYPE_CHECKING:
    from vectorbtpro.base import *
    from vectorbtpro.data import *
    from vectorbtpro.generic import *
    from vectorbtpro.indicators import *
    from vectorbtpro.labels import *
    from vectorbtpro.messaging import *
    from vectorbtpro.ohlcv import *
    from vectorbtpro.portfolio import *
    from vectorbtpro.px import *
    from vectorbtpro.records import *
    from vectorbtpro.registries import *
    from vectorbtpro.returns import *
    from vectorbtpro.signals import *
    from vectorbtpro.utils import *
    from vectorbtpro._opt_deps import *
    from vectorbtpro._settings import *
    from vectorbtpro._typing import *
    from vectorbtpro._version import *
    from vectorbtpro.accessors import *

from vectorbtpro import _typing as tp
from vectorbtpro._settings import settings
from vectorbtpro._version import __version__ as version

# Silence warnings
import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning

warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)
warnings.filterwarnings(
    "ignore", message="The localize method is no longer necessary, as this time zone supports the fold attribute"
)

if settings["importing"]["auto_import"]:
    from vectorbtpro.utils.module_ import check_installed

    def _auto_import(package):
        if isinstance(package, str):
            package = importlib.import_module(package)
        if not hasattr(package, "__all__"):
            package.__all__ = []
        if not hasattr(package, "__exclude_from__all__"):
            package.__exclude_from__all__ = []
        if not hasattr(package, "__import_if_installed__"):
            package.__import_if_installed__ = {}
        blacklist = []
        for k, v in package.__import_if_installed__.items():
            if not check_installed(v) or not settings["importing"][v]:
                blacklist.append(k)

        for importer, mod_name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            relative_name = mod_name.split(".")[-1]
            if relative_name in blacklist:
                continue
            if is_pkg:
                module = _auto_import(mod_name)
            else:
                module = importlib.import_module(mod_name)
            if hasattr(module, "__all__") and relative_name not in package.__exclude_from__all__:
                for k in module.__all__:
                    if hasattr(package, k) and getattr(package, k) is not getattr(module, k):
                        raise ValueError(
                            f"Attempt to override '{k}' in '{package.__name__}' from '{mod_name}'"
                        )
                    setattr(package, k, getattr(module, k))
                    package.__all__.append(k)
        return package

    _auto_import(__name__)

    from vectorbtpro.generic import nb, enums
    from vectorbtpro.indicators import nb as ind_nb, enums as ind_enums
    from vectorbtpro.labels import nb as lab_nb, enums as lab_enums
    from vectorbtpro.portfolio import nb as pf_nb, enums as pf_enums
    from vectorbtpro.records import nb as rec_nb
    from vectorbtpro.returns import nb as ret_nb, enums as ret_enums
    from vectorbtpro.signals import nb as sig_nb, enums as sig_enums
    from vectorbtpro.utils import datetime_ as dt, datetime_nb as dt_nb

__pdoc__ = dict()
__pdoc__["_settings"] = True
__pdoc__["_opt_deps"] = True
