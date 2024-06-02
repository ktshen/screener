# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for evaluation and compilation."""

import ast
import inspect

from vectorbtpro import _typing as tp

__all__ = []


def multiline_eval(expr: str, context: tp.KwargsLike = None) -> tp.Any:
    """Evaluate several lines of input, returning the result of the last line."""
    if context is None:
        context = {}
    tree = ast.parse(inspect.cleandoc(expr))
    eval_expr = ast.Expression(tree.body[-1].value)
    exec_expr = ast.Module(tree.body[:-1], type_ignores=[])
    exec(compile(exec_expr, "file", "exec"), context)
    return eval(compile(eval_expr, "file", "eval"), context)
