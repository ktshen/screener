# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for parsing."""

import ast
import inspect
import re
import sys
import io
import contextlib
import warnings
import attr

from vectorbtpro import _typing as tp

__all__ = [
    "Regex",
]


@attr.s(frozen=True)
class Regex:
    """Class for matching a regular expression."""

    pattern: str = attr.ib()
    """Pattern."""

    flags: int = attr.ib(default=0)
    """Flags."""

    def matches(self, string: str) -> bool:
        """Return whether the string matches the regular expression pattern."""
        return re.match(self.pattern, string, self.flags) is not None


def glob2re(pat):
    """Translate a shell pattern to a regular expression.

    Based on https://stackoverflow.com/a/29820981"""
    i, n = 0, len(pat)
    res = ""
    while i < n:
        c = pat[i]
        i = i + 1
        if c == "*":
            res = res + "[^/]*"
        elif c == "?":
            res = res + "[^/]"
        elif c == "[":
            j = i
            if j < n and pat[j] == "!":
                j = j + 1
            if j < n and pat[j] == "]":
                j = j + 1
            while j < n and pat[j] != "]":
                j = j + 1
            if j >= n:
                res = res + "\\["
            else:
                stuff = pat[i:j].replace("\\", "\\\\")
                i = j + 1
                if stuff[0] == "!":
                    stuff = "^" + stuff[1:]
                elif stuff[0] == "^":
                    stuff = "\\" + stuff
                res = "%s[%s]" % (res, stuff)
        else:
            res = res + re.escape(c)
    return res + r"\Z(?ms)"


def get_func_kwargs(func: tp.Callable) -> dict:
    """Get keyword arguments with defaults of a function."""
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_func_arg_names(func: tp.Callable, arg_kind: tp.Optional[tp.MaybeTuple[int]] = None) -> tp.List[str]:
    """Get argument names of a function."""
    signature = inspect.signature(func)
    if arg_kind is not None and isinstance(arg_kind, int):
        arg_kind = (arg_kind,)
    if arg_kind is None:
        return [p.name for p in signature.parameters.values() if p.kind != p.VAR_POSITIONAL and p.kind != p.VAR_KEYWORD]
    return [p.name for p in signature.parameters.values() if p.kind in arg_kind]


def extend_args(func: tp.Callable, args: tp.Args, kwargs: tp.Kwargs, **with_kwargs) -> tp.Tuple[tp.Args, tp.Kwargs]:
    """Extend arguments and keyword arguments with other arguments."""
    kwargs = dict(kwargs)
    new_args = ()
    new_kwargs = dict()
    signature = inspect.signature(func)
    for p in signature.parameters.values():
        if p.kind == p.VAR_POSITIONAL:
            new_args += args
            args = ()
            continue
        if p.kind == p.VAR_KEYWORD:
            for k in list(kwargs.keys()):
                new_kwargs[k] = kwargs.pop(k)
            continue

        arg_name = p.name.lower()
        took_from_args = False
        if arg_name not in kwargs and arg_name in with_kwargs:
            arg_value = with_kwargs[arg_name]
        elif len(args) > 0:
            arg_value = args[0]
            args = args[1:]
            took_from_args = True
        elif arg_name in kwargs:
            arg_value = kwargs.pop(arg_name)
        else:
            continue
        if p.kind == p.POSITIONAL_ONLY or len(args) > 0 or took_from_args:
            new_args += (arg_value,)
        else:
            new_kwargs[arg_name] = arg_value

    return new_args + args, {**new_kwargs, **kwargs}


def annotate_args(
    func: tp.Callable,
    args: tp.Args,
    kwargs: tp.Kwargs,
    only_passed: bool = False,
    allow_partial: bool = False,
) -> tp.AnnArgs:
    """Annotate arguments and keyword arguments using the function's signature."""
    kwargs = dict(kwargs)
    signature = inspect.signature(func)
    if not allow_partial:
        signature.bind(*args, **kwargs)
    ann_args = dict()

    for p in signature.parameters.values():
        if p.kind == p.POSITIONAL_ONLY:
            if len(args) > 0:
                ann_args[p.name] = dict(kind=p.kind, value=args[0])
                args = args[1:]
            elif not only_passed:
                if allow_partial:
                    ann_args[p.name] = dict(kind=p.kind)
                else:
                    raise TypeError(f"missing a required argument: '{p.name}'")
        elif p.kind == p.VAR_POSITIONAL:
            if len(args) > 0 or not only_passed:
                ann_args[p.name] = dict(kind=p.kind, value=args)
                args = ()
        elif p.kind == p.POSITIONAL_OR_KEYWORD:
            if len(args) > 0:
                ann_args[p.name] = dict(kind=p.kind, value=args[0])
                args = args[1:]
            elif p.name in kwargs:
                ann_args[p.name] = dict(kind=p.kind, value=kwargs.pop(p.name))
            elif not only_passed:
                if p.default is not p.empty:
                    ann_args[p.name] = dict(kind=p.kind, value=p.default)
                else:
                    if allow_partial:
                        ann_args[p.name] = dict(kind=p.kind)
                    else:
                        raise TypeError(f"missing a required argument: '{p.name}'")
        elif p.kind == p.KEYWORD_ONLY:
            if p.name in kwargs:
                ann_args[p.name] = dict(kind=p.kind, value=kwargs.pop(p.name))
            elif not only_passed:
                ann_args[p.name] = dict(kind=p.kind, value=p.default)
        else:
            if not only_passed or len(kwargs) > 0:
                ann_args[p.name] = dict(kind=p.kind, value=kwargs)
    return ann_args


def ann_args_to_args(ann_args: tp.AnnArgs) -> tp.Tuple[tp.Args, tp.Kwargs]:
    """Convert annotated arguments back to positional and keyword arguments."""
    args = ()
    kwargs = {}
    p = inspect.Parameter
    for k, v in ann_args.items():
        if v["kind"] in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            args += (v["value"],)
        elif v["kind"] == p.VAR_POSITIONAL:
            args += v["value"]
        elif v["kind"] == p.KEYWORD_ONLY:
            kwargs[k] = v["value"]
        else:
            for _k, _v in v["value"].items():
                kwargs[_k] = _v
    return args, kwargs


def flatten_ann_args(ann_args: tp.AnnArgs) -> tp.FlatAnnArgs:
    """Flatten annotated arguments."""
    flat_ann_args = {}
    for arg_name, ann_arg in ann_args.items():
        if ann_arg["kind"] == inspect.Parameter.VAR_POSITIONAL:
            for i, v in enumerate(ann_arg["value"]):
                flat_ann_args[f"{arg_name}_{i}"] = dict(var_name=arg_name, kind=ann_arg["kind"], value=v)
        elif ann_arg["kind"] == inspect.Parameter.VAR_KEYWORD:
            for var_arg_name, var_value in ann_arg["value"].items():
                flat_ann_args[var_arg_name] = dict(var_name=arg_name, kind=ann_arg["kind"], value=var_value)
        else:
            flat_ann_args[arg_name] = dict(kind=ann_arg["kind"], value=ann_arg["value"])
    return flat_ann_args


def unflatten_ann_args(flat_ann_args: tp.FlatAnnArgs) -> tp.AnnArgs:
    """Unflatten annotated arguments."""
    ann_args = dict()
    for arg_name, ann_arg in flat_ann_args.items():
        ann_arg = dict(ann_arg)
        if ann_arg["kind"] == inspect.Parameter.VAR_POSITIONAL:
            var_arg_name = ann_arg.pop("var_name")
            if var_arg_name not in ann_args:
                ann_args[var_arg_name] = dict(value=(), kind=ann_arg["kind"])
            ann_args[var_arg_name]["value"] = ann_args[var_arg_name]["value"] + (ann_arg["value"],)
        elif ann_arg["kind"] == inspect.Parameter.VAR_KEYWORD:
            var_arg_name = ann_arg.pop("var_name")
            if var_arg_name not in ann_args:
                ann_args[var_arg_name] = dict(value={}, kind=ann_arg["kind"])
            ann_args[var_arg_name]["value"][arg_name] = ann_arg["value"]
        else:
            ann_args[arg_name] = ann_arg
    return ann_args


def match_ann_arg(
    ann_args: tp.AnnArgs,
    query: tp.AnnArgQuery,
    return_name: bool = False,
    return_index: bool = False,
) -> tp.Any:
    """Match an argument from annotated arguments.

    A query can be an integer indicating the position of the argument, or a string containing the name
    of the argument or a regular expression for matching the name of the argument.

    If multiple arguments were matched, returns the first one.

    The position can stretch over any variable argument."""
    if return_name and return_index:
        raise ValueError("Either return_name or return_index can be provided, not both")
    flat_ann_args = flatten_ann_args(ann_args)
    for i, (arg_name, ann_arg) in enumerate(flat_ann_args.items()):
        if (
            (isinstance(query, int) and query == i)
            or (isinstance(query, str) and query == arg_name)
            or (isinstance(query, Regex) and query.matches(arg_name))
        ):
            if return_name:
                return arg_name
            if return_index:
                return i
            return ann_arg["value"]
    raise KeyError(f"Query '{query}' could not be matched with any argument")


def ignore_flat_ann_args(flat_ann_args: tp.FlatAnnArgs, ignore_args: tp.Iterable[tp.AnnArgQuery]) -> tp.FlatAnnArgs:
    """Ignore flattened annotated arguments."""
    new_flat_ann_args = {}
    for i, (arg_name, arg) in enumerate(flat_ann_args.items()):
        arg_matched = False
        for ignore_arg in ignore_args:
            if (
                (isinstance(ignore_arg, int) and ignore_arg == i)
                or (isinstance(ignore_arg, str) and ignore_arg == arg_name)
                or (isinstance(ignore_arg, Regex) and ignore_arg.matches(arg_name))
            ):
                arg_matched = True
                break
        if not arg_matched:
            new_flat_ann_args[arg_name] = arg
    return new_flat_ann_args


class UnhashableArgsError(Exception):
    """Unhashable arguments error."""

    pass


def hash_args(
    func: tp.Callable,
    args: tp.Args,
    kwargs: tp.Kwargs,
    ignore_args: tp.Optional[tp.Iterable[tp.AnnArgQuery]] = None,
) -> int:
    """Get hash of arguments.

    Use `ignore_args` to provide a sequence of queries for arguments that should be ignored."""
    if ignore_args is None:
        ignore_args = []
    ann_args = annotate_args(func, args, kwargs, only_passed=True)
    flat_ann_args = flatten_ann_args(ann_args)
    if len(ignore_args) > 0:
        flat_ann_args = ignore_flat_ann_args(flat_ann_args, ignore_args)
    try:
        return hash(tuple(map(lambda x: (x[0], x[1]["value"]), flat_ann_args.items())))
    except TypeError:
        raise UnhashableArgsError


def get_expr_var_names(expression: str) -> tp.List[str]:
    """Get variable names listed in the expression."""
    return [node.id for node in ast.walk(ast.parse(expression)) if type(node) is ast.Name]


def get_context_vars(
    var_names: tp.Iterable[str],
    frames_back: int = 0,
    local_dict: tp.Optional[tp.Mapping] = None,
    global_dict: tp.Optional[tp.Mapping] = None,
) -> tp.List[tp.Any]:
    """Get variables from the local/global context."""
    call_frame = sys._getframe(frames_back + 1)
    clear_local_dict = False
    if local_dict is None:
        local_dict = call_frame.f_locals
        clear_local_dict = True
    try:
        frame_globals = call_frame.f_globals
        if global_dict is None:
            global_dict = frame_globals
        clear_local_dict = clear_local_dict and frame_globals is not local_dict
        args = []
        for var_name in var_names:
            try:
                a = local_dict[var_name]
            except KeyError:
                a = global_dict[var_name]
            args.append(a)
    finally:
        # See https://github.com/pydata/numexpr/issues/310
        if clear_local_dict:
            local_dict.clear()
    return args


def supress_stdout(func: tp.Callable) -> tp.Callable:
    """Supress output from a function."""

    def wrapper(*a, **ka):
        with contextlib.redirect_stdout(io.StringIO()):
            return func(*a, **ka)

    return wrapper


def warn_stdout(func: tp.Callable) -> tp.Callable:
    """Supress and convert to a warning output from a function."""

    def wrapper(*a, **ka):
        with contextlib.redirect_stdout(io.StringIO()) as f:
            out = func(*a, **ka)
        s = f.getvalue()
        if len(s) > 0:
            warnings.warn(s, stacklevel=2)
        return out

    return wrapper
