# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for working with templates."""

from copy import copy
from string import Template

import attr
import numpy as np
import pandas as pd

import vectorbtpro as vbt
from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import set_dict_item, merge_dicts
from vectorbtpro.utils.eval_ import multiline_eval
from vectorbtpro.utils.hashing import Hashable
from vectorbtpro.utils.parsing import get_func_arg_names

__all__ = [
    "CustomTemplate",
    "Sub",
    "Rep",
    "RepEval",
    "RepFunc",
    "substitute_templates",
]


@attr.s(frozen=True)
class CustomTemplate:
    """Class for substituting templates."""

    template: tp.Any = attr.ib()
    """Template to be processed."""

    context: tp.Optional[tp.Mapping] = attr.ib(default=None)
    """Context mapping."""

    strict: tp.Optional[bool] = attr.ib(default=None)
    """Whether to raise an error if processing template fails.

    If not None, overrides `strict` passed by `substitute_templates`."""

    sub_id: tp.Optional[tp.MaybeCollection[Hashable]] = attr.ib(default=None)
    """Substitution id or ids at which to evaluate this template

    Checks against `sub_id` passed by `substitute_templates`."""

    context_merge_kwargs: tp.KwargsLike = attr.ib(default=None)
    """Keyword arguments passed to `vectorbtpro.utils.config.merge_dicts`."""

    def meets_sub_id(self, sub_id: tp.Optional[Hashable] = None) -> bool:
        """Return whether the substitution id of the template meets the global substitution id."""
        if self.sub_id is not None and sub_id is not None:
            if isinstance(self.sub_id, int):
                if sub_id != self.sub_id:
                    return False
            else:
                if sub_id not in self.sub_id:
                    return False
        return True

    def resolve_context(
        self,
        context: tp.Optional[tp.Mapping] = None,
        sub_id: tp.Optional[Hashable] = None,
    ) -> tp.Kwargs:
        """Resolve `CustomTemplate.context`.

        Merges `context` in `vectorbtpro._settings.template`, `CustomTemplate.context`, and `context`.
        Automatically appends `sub_id`, `np` (NumPy), `pd` (Pandas), and `vbt` (vectorbtpro)."""
        from vectorbtpro._settings import settings

        template_cfg = settings["template"]

        context_merge_kwargs = self.context_merge_kwargs
        if context_merge_kwargs is None:
            context_merge_kwargs = {}
        new_context = merge_dicts(
            template_cfg["context"],
            self.context,
            context,
            **context_merge_kwargs,
        )
        new_context = merge_dicts(
            dict(
                context=new_context,
                sub_id=sub_id,
                np=np,
                pd=pd,
                vbt=vbt,
            ),
            new_context,
        )
        return new_context

    def resolve_strict(self, strict: tp.Optional[bool] = None) -> bool:
        """Resolve `CustomTemplate.strict`.

        If `strict` is None, uses `strict` in `vectorbtpro._settings.template`."""
        if strict is None:
            strict = self.strict
        if strict is None:
            from vectorbtpro._settings import settings

            template_cfg = settings["template"]

            strict = template_cfg["strict"]
        return strict

    def substitute(
        self,
        context: tp.Optional[tp.Mapping] = None,
        strict: tp.Optional[bool] = None,
        sub_id: tp.Optional[Hashable] = None,
    ) -> tp.Any:
        """Abstract method to substitute the template `CustomTemplate.template` using
        the context from merging `CustomTemplate.context` and `context`."""
        raise NotImplementedError


class Sub(CustomTemplate):
    """Template string to substitute parts with the respective values from `context`.

    Always returns a string."""

    def substitute(
        self,
        context: tp.Optional[tp.Mapping] = None,
        strict: tp.Optional[bool] = None,
        sub_id: tp.Optional[Hashable] = None,
    ) -> tp.Any:
        """Substitute parts of `Sub.template` as a regular template."""
        if not self.meets_sub_id(sub_id):
            return self
        context = self.resolve_context(context=context, sub_id=sub_id)
        strict = self.resolve_strict(strict=strict)

        try:
            return Template(self.template).substitute(context)
        except KeyError as e:
            if strict:
                raise e
        return self


class Rep(CustomTemplate):
    """Template string to be replaced with the respective value from `context`."""

    def substitute(
        self,
        context: tp.Optional[tp.Mapping] = None,
        strict: tp.Optional[bool] = None,
        sub_id: tp.Optional[Hashable] = None,
    ) -> tp.Any:
        """Replace `Rep.template` as a key."""
        if not self.meets_sub_id(sub_id):
            return self
        context = self.resolve_context(context=context, sub_id=sub_id)
        strict = self.resolve_strict(strict=strict)

        try:
            return context[self.template]
        except KeyError as e:
            if strict:
                raise e
        return self


class RepEval(CustomTemplate):
    """Template expression to be evaluated using `vectorbtpro.utils.eval_.multiline_eval`
    with `context` used as locals."""

    def substitute(
        self,
        context: tp.Optional[tp.Mapping] = None,
        strict: tp.Optional[bool] = None,
        sub_id: tp.Optional[Hashable] = None,
    ) -> tp.Any:
        """Evaluate `RepEval.template` as an expression."""
        if not self.meets_sub_id(sub_id):
            return self
        context = self.resolve_context(context=context, sub_id=sub_id)
        strict = self.resolve_strict(strict=strict)

        try:
            return multiline_eval(self.template, context)
        except NameError as e:
            if strict:
                raise e
        return self


class RepFunc(CustomTemplate):
    """Template function to be called with argument names from `context`."""

    def substitute(
        self,
        context: tp.Optional[tp.Mapping] = None,
        strict: tp.Optional[bool] = None,
        sub_id: int = 0,
    ) -> tp.Any:
        """Call `RepFunc.template` as a function."""
        if not self.meets_sub_id(sub_id):
            return self
        context = self.resolve_context(context=context, sub_id=sub_id)
        strict = self.resolve_strict(strict=strict)

        func_arg_names = get_func_arg_names(self.template)
        func_kwargs = dict()
        for k, v in context.items():
            if k in func_arg_names:
                func_kwargs[k] = v

        try:
            return self.template(**func_kwargs)
        except TypeError as e:
            if strict:
                raise e
        return self


def has_templates(
    obj: tp.Any,
    except_types: tp.Optional[tp.Sequence[type]] = None,
    max_len: tp.Optional[int] = None,
    max_depth: tp.Optional[int] = None,
    _depth: int = 0,
) -> tp.Any:
    """Check if the object has any templates.

    For arguments, see `substitute_templates`."""
    from vectorbtpro._settings import settings

    template_cfg = settings["template"]

    if except_types is None:
        except_types = template_cfg["except_types"]
    if max_len is None:
        max_len = template_cfg["max_len"]
    if max_depth is None:
        max_depth = template_cfg["max_depth"]

    if except_types is not None and checks.is_instance_of(obj, except_types):
        return False
    if isinstance(obj, (Template, CustomTemplate)):
        return True
    if max_depth is None or _depth < max_depth:
        if isinstance(obj, dict):
            if max_len is None or len(obj) <= max_len:
                for k, v in obj.items():
                    if has_templates(
                        v,
                        except_types=except_types,
                        max_len=max_len,
                        max_depth=max_depth,
                        _depth=_depth + 1,
                    ):
                        return True
        if isinstance(obj, (tuple, list, set, frozenset)):
            if max_len is None or len(obj) <= max_len:
                for v in obj:
                    if has_templates(
                        v,
                        except_types=except_types,
                        max_len=max_len,
                        max_depth=max_depth,
                        _depth=_depth + 1,
                    ):
                        return True
    return False


def substitute_templates(
    obj: tp.Any,
    context: tp.Optional[tp.Mapping] = None,
    strict: tp.Optional[bool] = None,
    make_copy: bool = True,
    sub_id: tp.Optional[Hashable] = None,
    except_types: tp.Optional[tp.Sequence[type]] = None,
    max_len: tp.Optional[int] = None,
    max_depth: tp.Optional[int] = None,
    _depth: int = 0,
) -> tp.Any:
    """Traverses the object recursively and, if any template found, substitutes it using a context.

    Traverses tuples, lists, dicts and (frozen-)sets. Does not look for templates in keys.

    If `except_types` is not None, uses `vectorbtpro.utils.checks.is_instance_of` to check whether
    the object is one of the types that are blacklisted. If so, the object is simply returned.
    By default, out of all sequences, only dicts and tuples are substituted.

    If `max_len` is not None, processes any object only if it's shorter than the specified length.

    If `max_depth` is not None, processes any object only up to a certain recursion level.
    Level of 0 means dicts and other iterables are not processed, only templates are expected.

    If `strict` is True, raises an error if processing template fails. Otherwise, returns the original template.

    For defaults, see `vectorbtpro._settings.template`.

    !!! note
        If the object is deep (such as a dict or a list), creates a copy of it if any template found inside,
        thus loosing the reference to the original. Make sure to do a deep or hybrid copy of the object
        before proceeding for consistent behavior, or disable `make_copy` to override the original in place.

    Usage:
        ```pycon
        >>> import vectorbtpro as vbt

        >>> vbt.substitute_templates(vbt.Sub('$key', {'key': 100}))
        100
        >>> vbt.substitute_templates(vbt.Sub('$key', {'key': 100}), {'key': 200})
        200
        >>> vbt.substitute_templates(vbt.Sub('$key$key'), {'key': 100})
        100100
        >>> vbt.substitute_templates(vbt.Rep('key'), {'key': 100})
        100
        >>> vbt.substitute_templates([vbt.Rep('key'), vbt.Sub('$key$key')], {'key': 100}, except_types=())
        [100, '100100']
        >>> vbt.substitute_templates(vbt.RepFunc(lambda key: key == 100), {'key': 100})
        True
        >>> vbt.substitute_templates(vbt.RepEval('key == 100'), {'key': 100})
        True
        >>> vbt.substitute_templates(vbt.RepEval('key == 100', strict=True))
        NameError: name 'key' is not defined
        >>> vbt.substitute_templates(vbt.RepEval('key == 100', strict=False))
        <vectorbtpro.utils.template.RepEval at 0x7fe3ad2ab668>
        ```
    """
    from vectorbtpro._settings import settings

    template_cfg = settings["template"]

    if except_types is None:
        except_types = template_cfg["except_types"]
    if max_len is None:
        max_len = template_cfg["max_len"]
    if max_depth is None:
        max_depth = template_cfg["max_depth"]
    if context is None:
        context = {}

    if not has_templates(
        obj,
        except_types=except_types,
        max_len=max_len,
        max_depth=max_depth,
        _depth=_depth,
    ):
        return obj

    if isinstance(obj, CustomTemplate):
        return obj.substitute(context=context, strict=strict, sub_id=sub_id)
    if isinstance(obj, Template):
        return obj.substitute(context=context)
    if max_depth is None or _depth < max_depth:
        if except_types is not None and checks.is_instance_of(obj, except_types):
            return obj
        if isinstance(obj, dict):
            if max_len is None or len(obj) <= max_len:
                if make_copy:
                    obj = copy(obj)
                for k, v in obj.items():
                    set_dict_item(
                        obj,
                        k,
                        substitute_templates(
                            v,
                            context=context,
                            strict=strict,
                            sub_id=sub_id,
                            except_types=except_types,
                            max_len=max_len,
                            max_depth=max_depth,
                            _depth=_depth + 1,
                        ),
                        force=True,
                    )
                return obj
        if isinstance(obj, list):
            if max_len is None or len(obj) <= max_len:
                if make_copy:
                    obj = copy(obj)
                for i in range(len(obj)):
                    obj[i] = substitute_templates(
                        obj[i],
                        context=context,
                        strict=strict,
                        sub_id=sub_id,
                        except_types=except_types,
                        max_len=max_len,
                        max_depth=max_depth,
                        _depth=_depth + 1,
                    )
                return obj
        if isinstance(obj, (tuple, set, frozenset)):
            if max_len is None or len(obj) <= max_len:
                result = []
                for o in obj:
                    result.append(
                        substitute_templates(
                            o,
                            context=context,
                            strict=strict,
                            sub_id=sub_id,
                            except_types=except_types,
                            max_len=max_len,
                            max_depth=max_depth,
                            _depth=_depth + 1,
                        )
                    )
                if checks.is_namedtuple(obj):
                    return type(obj)(*result)
                return type(obj)(result)
    return obj
