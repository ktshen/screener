# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for working with tags."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.parsing import get_expr_var_names
from vectorbtpro.utils.template import RepEval

__all__ = []


def match_tags(tags: tp.MaybeIterable[str], in_tags: tp.MaybeIterable[str]) -> bool:
    """Match tags in `tags` to that in `in_tags`.

    Multiple tags in `tags` are combined using OR rule, that is, returns True if any of them is found in `in_tags`.
    If any tag is not an identifier, evaluates it as a boolean expression.
    All tags in `in_tags` must be identifiers.

    Usage:
        ```pycon
        >>> from vectorbtpro.utils.tagging import match_tags

        >>> match_tags('hello', 'hello')
        True
        >>> match_tags('hello', 'world')
        False
        >>> match_tags(['hello', 'world'], 'world')
        True
        >>> match_tags('hello', ['hello', 'world'])
        True
        >>> match_tags('hello and world', ['hello', 'world'])
        True
        >>> match_tags('hello and not world', ['hello', 'world'])
        False
        ```
    """
    if isinstance(tags, str):
        tags = [tags]
    if isinstance(in_tags, str):
        in_tags = [in_tags]
    for in_t in in_tags:
        if not in_t.isidentifier():
            raise ValueError(f"Tag '{in_t}' must be an identifier")

    for t in tags:
        if not t.isidentifier():
            var_names = get_expr_var_names(t)
            eval_context = {var_name: var_name in in_tags for var_name in var_names}
            eval_result = RepEval(t).substitute(eval_context)
            if not isinstance(eval_result, bool):
                raise TypeError(f"Tag expression '{t}' must produce a boolean")
            if eval_result:
                return True
        else:
            if t in in_tags:
                return True
    return False
