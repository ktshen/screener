# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for modules."""

import warnings
import importlib
import importlib.util
import inspect
import pkgutil
import sys
from pathlib import Path
from types import ModuleType, FunctionType

from vectorbtpro import _typing as tp
from vectorbtpro._opt_deps import opt_dep_config

__all__ = [
    "import_module_from_path",
]


def is_from_module(obj: tp.Any, module: ModuleType) -> bool:
    """Return whether `obj` is from module `module`."""
    mod = inspect.getmodule(inspect.unwrap(obj))
    return mod is None or mod.__name__ == module.__name__


def list_module_keys(
    module_name: str,
    whitelist: tp.Optional[tp.List[str]] = None,
    blacklist: tp.Optional[tp.List[str]] = None,
) -> tp.List[str]:
    """List the names of all public functions and classes defined in the module `module_name`.

    Includes the names listed in `whitelist` and excludes the names listed in `blacklist`."""
    if whitelist is None:
        whitelist = []
    if blacklist is None:
        blacklist = []
    module = sys.modules[module_name]
    return [
        name
        for name, obj in inspect.getmembers(module)
        if (
            not name.startswith("_")
            and is_from_module(obj, module)
            and ((inspect.isroutine(obj) and callable(obj)) or inspect.isclass(obj))
            and name not in blacklist
        )
        or name in whitelist
    ]


def search_package_for_funcs(
    package: tp.Union[str, ModuleType],
    blacklist: tp.Optional[tp.Sequence[str]] = None,
) -> tp.Dict[str, FunctionType]:
    """Search a package for all functions."""
    if blacklist is None:
        blacklist = []
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if ".".join(name.split(".")[:-1]) != package.__name__:
            continue
        try:
            if name in blacklist:
                continue
            module = importlib.import_module(name)
            for attr in dir(module):
                if not attr.startswith("_") and isinstance(getattr(module, attr), FunctionType):
                    results[attr] = getattr(module, attr)
            if is_pkg:
                results.update(search_package_for_funcs(name, blacklist=blacklist))
        except ModuleNotFoundError as e:
            pass
    return results


def find_class(path: str) -> tp.Optional[tp.Type]:
    """Find the class by its path."""
    try:
        path_parts = path.split(".")
        module_path = ".".join(path_parts[:-1])
        class_name = path_parts[-1]
        if module_path.startswith("vectorbtpro.indicators.factory"):
            import vectorbtpro as vbt

            return getattr(vbt, path_parts[-2])(class_name)
        module = importlib.import_module(module_path)
        if hasattr(module, class_name):
            return getattr(module, class_name)
    except Exception as e:
        pass
    return None


def check_installed(pkg_name: str) -> bool:
    """Check if a package is installed."""
    return importlib.util.find_spec(pkg_name) is not None


def get_installed_overview() -> tp.Dict[str, bool]:
    """Get an overview of installed packages in `opt_dep_config`."""
    return {pkg_name: check_installed(pkg_name) for pkg_name in opt_dep_config.keys()}


def assert_can_import(pkg_name: str) -> None:
    """Assert that the package can be imported. Must be listed in `opt_dep_config`."""
    from importlib.metadata import version as get_version

    if pkg_name not in opt_dep_config:
        raise KeyError(f"Package '{pkg_name}' not found in opt_dep_config")
    dist_name = opt_dep_config[pkg_name].get("dist_name", pkg_name)
    version = version_str = opt_dep_config[pkg_name].get("version", "")
    link = opt_dep_config[pkg_name]["link"]
    if not check_installed(pkg_name):
        raise ImportError(f"Please install {dist_name}{version_str} (see {link})")
    if version != "":
        actual_version = "(" + get_version(dist_name).replace(".", ",") + ")"
        if version[0].isdigit():
            operator = "=="
        else:
            operator = version[:2]
            version = version[2:]
            version = "(" + version.replace(".", ",") + ")"
        if not eval(f"{actual_version} {operator} {version}"):
            raise ImportError(f"Please install {dist_name}{version_str} (see {link})")


def warn_cannot_import(pkg_name: str) -> bool:
    """Warn if the package is cannot be imported. Must be listed in `opt_dep_config`."""
    try:
        assert_can_import(pkg_name)
        return False
    except ImportError as e:
        warnings.warn(str(e), stacklevel=2)
        return True


def import_module_from_path(module_path: tp.PathLike, reload: bool = False) -> ModuleType:
    """Import the module from a path."""
    module_path = Path(module_path)
    spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path.resolve()))
    module = importlib.util.module_from_spec(spec)
    if module.__name__ in sys.modules and not reload:
        return sys.modules[module.__name__]
    spec.loader.exec_module(module)
    sys.modules[module.__name__] = module
    return module
