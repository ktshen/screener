# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for working with paths."""

from pathlib import Path
from itertools import islice
import humanize
import shutil

from vectorbtpro import _typing as tp

__all__ = [
    "file_exists",
    "dir_exists",
    "file_size",
    "dir_size",
    "make_dir",
    "remove_file",
    "remove_dir",
]


def file_exists(file_path: tp.PathLike) -> bool:
    """Check whether a file exists."""
    file_path = Path(file_path)
    if file_path.exists() and file_path.is_file():
        return True
    return False


def dir_exists(dir_path: tp.PathLike) -> bool:
    """Check whether a directory exists."""
    dir_path = Path(dir_path)
    if dir_path.exists() and dir_path.is_dir():
        return True
    return False


def file_size(file_path: tp.PathLike, readable: bool = True, **kwargs) -> tp.Union[str, int]:
    """Get size of a file."""
    file_path = Path(file_path)
    if not file_exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found")
    n_bytes = file_path.stat().st_size
    if readable:
        return humanize.naturalsize(n_bytes, **kwargs)
    return n_bytes


def dir_size(dir_path: tp.PathLike, readable: bool = True, **kwargs) -> tp.Union[str, int]:
    """Get size of a directory."""
    dir_path = Path(dir_path)
    if not dir_exists(dir_path):
        raise FileNotFoundError(f"Directory '{dir_path}' not found")
    n_bytes = sum(path.stat().st_size for path in dir_path.glob('**/*') if path.is_file())
    if readable:
        return humanize.naturalsize(n_bytes, **kwargs)
    return n_bytes


def check_mkdir(
    dir_path: tp.PathLike,
    mkdir: tp.Optional[bool] = None,
    mode: tp.Optional[int] = None,
    parents: tp.Optional[bool] = None,
    exist_ok: tp.Optional[bool] = None,
) -> None:
    """Check whether the path to a directory exists and create if it doesn't.

    For defaults, see `mkdir` in `vectorbtpro._settings.path`."""
    from vectorbtpro._settings import settings

    mkdir_cfg = settings["path"]["mkdir"]

    if mkdir is None:
        mkdir = mkdir_cfg["mkdir"]
    if mode is None:
        mode = mkdir_cfg["mode"]
    if parents is None:
        parents = mkdir_cfg["parents"]
    if exist_ok is None:
        exist_ok = mkdir_cfg["exist_ok"]

    dir_path = Path(dir_path)
    if dir_path.exists() and not dir_path.is_dir():
        raise TypeError(f"Path '{dir_path}' is not a directory")
    if not dir_path.exists() and not mkdir:
        raise FileNotFoundError(f"Directory '{dir_path}' not found. Use mkdir=True to proceed.")
    dir_path.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)


def make_dir(dir_path: tp.PathLike, **kwargs) -> None:
    """Make a directory."""
    check_mkdir(dir_path, mkdir=True, **kwargs)


def remove_file(file_path: tp.PathLike, missing_ok: bool = False) -> None:
    """Remove a file."""
    file_path = Path(file_path)
    if file_exists(file_path):
        file_path.unlink()
    elif not missing_ok:
        raise FileNotFoundError(f"File '{file_path}' not found")


def remove_dir(dir_path: tp.PathLike, missing_ok: bool = False, with_contents: bool = False) -> None:
    """Remove a directory."""
    dir_path = Path(dir_path)
    if dir_exists(dir_path):
        if any(dir_path.iterdir()) and not with_contents:
            raise ValueError(f"Directory '{dir_path}' has contents. Use with_contents=True to proceed.")
        shutil.rmtree(dir_path)
    elif not missing_ok:
        raise FileNotFoundError(f"Directory '{dir_path}' not found")


def tree(
    dir_path: Path,
    level: int = -1,
    limit_to_directories: bool = False,
    length_limit: int = 1000,
    sort: bool = True,
    space="    ",
    branch="│   ",
    tee="├── ",
    last="└── ",
) -> str:
    """Given a directory Path object print a visual tree structure.

    Inspired by this answer: https://stackoverflow.com/a/59109706"""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{dir_path}' not found")
    if not dir_path.is_dir():
        raise TypeError(f"Path '{dir_path}' is not a directory")
    files = 0
    directories = 0

    def _inner(dir_path: Path, prefix: str = "", level: int = -1) -> tp.Generator[str, None, None]:
        nonlocal files, directories
        if not level:
            return  # 0, stop iterating
        if limit_to_directories:
            contents = [d for d in dir_path.iterdir() if d.is_dir()]
        else:
            contents = list(dir_path.iterdir())
        if sort:
            contents = sorted(contents)
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + path.name
                directories += 1
                extension = branch if pointer == tee else space
                yield from _inner(path, prefix=prefix + extension, level=level - 1)
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files += 1

    tree_str = dir_path.name
    iterator = _inner(dir_path, level=level)
    for line in islice(iterator, length_limit):
        tree_str += "\n" + line
    if next(iterator, None):
        tree_str += "\n" + f"... length_limit, {length_limit}, reached, counted:"
    tree_str += "\n" + f"\n{directories} directories" + (f", {files} files" if files else "")
    return tree_str
