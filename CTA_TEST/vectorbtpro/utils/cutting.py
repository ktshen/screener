# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for cutting code."""

import inspect
from types import ModuleType, FunctionType
import importlib
from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.utils.template import CustomTemplate, RepEval
from vectorbtpro.utils.path_ import check_mkdir

__all__ = [
    "cut_and_save_module",
    "cut_and_save_func",
]


def collect_blocks(lines: tp.Iterable[str]) -> tp.Dict[str, tp.List[str]]:
    """Collect blocks in the lines."""
    blocks = {}
    block_name = None

    for line in lines:
        sline = line.strip()

        if sline.startswith("# % <block") and sline.endswith(">"):
            block_name = sline[len("# % <block") : -1].strip()
            if len(block_name) == 0:
                raise ValueError("Missing block name")
            blocks[block_name] = []
        elif sline.startswith("# % </block>"):
            block_name = None
        elif block_name is not None:
            blocks[block_name].append(line)

    return blocks


def cut_from_code(
    code: str,
    section_name: str,
    prepend_lines: tp.Optional[tp.Iterable[str]] = None,
    append_lines: tp.Optional[tp.Iterable[str]] = None,
    out_lines_callback: tp.Union[None, tp.Callable, CustomTemplate] = None,
    return_lines: bool = False,
    **kwargs,
) -> tp.Union[str, tp.List[str]]:
    """Parse and cut an annotated section from the code.

    The section should start with `# % <section section_name>` and end with `# % </section>`.

    You can also define blocks. Each block should start with `# % <block block_name>` and end with `# % </block>`.
    Blocks will be collected into the dictionary `blocks` before cutting and can be then inserted using
    Python expressions (see below).

    To skip multiple lines of code, place them between `# % <skip [expression]>` and `# % </skip>`,
    where expression is optional.

    To uncomment multiple lines of code, place them between `# % <uncomment [expression]>` and
    `# % </uncomment>`, where expression is optional.

    Everything else after `# %` will be evaluated as a Python expression and should return
    either None (= skip), a string (= insert one line of code) or an iterable of strings
    (= insert multiple lines of code). The latter will be appended to the queue and parsed.

    Every expression is evaluated strictly, that is, any evaluation error will raise an error
    and stop the program. To evaluate softly without raising any errors, prepend `?`.
    The context includes `lines`, `blocks`, `section_name`, `line`, `out_lines`, and `**kwargs`."""
    lines = code.split("\n")
    blocks = collect_blocks(lines)

    out_lines = []
    if prepend_lines is not None:
        out_lines.extend(list(prepend_lines))
    section_found = False
    uncomment = False
    skip = False
    i = 0

    while i < len(lines):
        line = lines[i]
        sline = line.strip()

        if sline.startswith("# % <section") and sline.endswith(">"):
            if section_found:
                raise ValueError("Missing </section>")
            found_name = sline[len("# % <section") : -1].strip()
            if len(found_name) == 0:
                raise ValueError("Missing section name")
            section_found = found_name == section_name
        elif section_found:
            context = {
                "lines": lines,
                "blocks": blocks,
                "section_name": section_name,
                "line": line,
                "out_lines": out_lines,
                **kwargs,
            }
            if sline.startswith("# % </section>"):
                if append_lines is not None:
                    out_lines.extend(list(append_lines))
                if out_lines_callback is not None:
                    if isinstance(out_lines_callback, CustomTemplate):
                        out_lines_callback = out_lines_callback.substitute(context=context, strict=True)
                    out_lines = out_lines_callback(out_lines)
                if return_lines:
                    return out_lines
                return inspect.cleandoc("\n".join(out_lines))
            if sline.startswith("# % <skip") and sline.endswith(">"):
                if skip:
                    raise ValueError("Missing </skip>")
                expression = sline[len("# % <skip"): -1].strip()
                if len(expression) == 0:
                    skip = True
                else:
                    if expression.startswith("?"):
                        expression = expression[1:]
                        strict = False
                    else:
                        strict = True
                    eval_skip = RepEval(expression).substitute(context=context, strict=strict)
                    if not isinstance(eval_skip, RepEval):
                        skip = eval_skip
            elif sline.startswith("# % </skip>"):
                skip = False
            elif not skip:
                if sline.startswith("# % <uncomment") and sline.endswith(">"):
                    if uncomment:
                        raise ValueError("Missing </uncomment>")
                    expression = sline[len("# % <uncomment"): -1].strip()
                    if len(expression) == 0:
                        uncomment = True
                    else:
                        if expression.startswith("?"):
                            expression = expression[1:]
                            strict = False
                        else:
                            strict = True
                        eval_uncomment = RepEval(expression).substitute(context=context, strict=strict)
                        if not isinstance(eval_uncomment, RepEval):
                            uncomment = eval_uncomment
                elif sline.startswith("# % </uncomment>"):
                    uncomment = False
                elif "# %" in line:
                    expression = line.split("# %")[1].strip()
                    if expression.startswith("?"):
                        expression = expression[1:]
                        strict = False
                    else:
                        strict = True
                    line_woc = line.split("# %")[0].rstrip()
                    context["line"] = line_woc
                    eval_line = RepEval(expression).substitute(context=context, strict=strict)
                    if eval_line is not None:
                        if not isinstance(eval_line, RepEval):
                            if isinstance(eval_line, str):
                                out_lines.append(eval_line)
                            else:
                                lines[i + 1: i + 1] = eval_line
                        else:
                            out_lines.append(line)
                elif uncomment:
                    if sline.startswith("# "):
                        out_lines.append(sline[2:])
                    elif sline.startswith("#"):
                        out_lines.append(sline[1:])
                    else:
                        out_lines.append(line)
                else:
                    out_lines.append(line)

        i += 1
    if section_found:
        raise ValueError(f"Code section '{section_name}' not closed")
    raise ValueError(f"Code section '{section_name}' not found")


def suggest_module_path(
    section_name: str,
    path: tp.Optional[tp.PathLike] = None,
    mkdir_kwargs: tp.KwargsLike = None,
) -> Path:
    """Suggest a path to the target file."""
    if path is None:
        path = Path(".")
    else:
        path = Path(path)
    if not path.is_file() and path.suffix == "":
        path = (path / section_name).with_suffix(".py")
    if mkdir_kwargs is None:
        mkdir_kwargs = {}
    check_mkdir(path.parent, **mkdir_kwargs)
    return path


def cut_and_save(
    code: str,
    section_name: str,
    path: tp.Optional[tp.PathLike] = None,
    mkdir_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> Path:
    """Cut an annotated section from the code and save to a file.

    For arguments see `cut_from_code`."""
    parsed_code = cut_from_code(code, section_name, **kwargs)
    path = suggest_module_path(section_name, path=path, mkdir_kwargs=mkdir_kwargs)
    with open(path, "w") as f:
        f.write(parsed_code)
    return path


def cut_and_save_module(module: tp.Union[str, ModuleType], *args, **kwargs) -> Path:
    """Cut an annotated section from a module and save to a file.

    For arguments see `cut_and_save`."""
    if isinstance(module, str):
        module = importlib.import_module(module)
    code = inspect.getsource(module)
    return cut_and_save(code, *args, **kwargs)


def cut_and_save_func(func: tp.Union[str, FunctionType], *args, **kwargs) -> Path:
    """Cut an annotated function section from a module and save to a file.

    For arguments see `cut_and_save`."""
    if isinstance(func, str):
        module = importlib.import_module(".".join(func.split(".")[:-1]))
        func = getattr(module, func.split(".")[-1])
    else:
        module = inspect.getmodule(func)
    code = inspect.getsource(module)
    return cut_and_save(code, section_name=func.__name__, *args, **kwargs)
