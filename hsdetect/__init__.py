# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:21:29 2023

@author: Rehmer
"""

import hsdetect.annot
import hsdetect.cv
import hsdetect.ipc

import ast
import importlib
import pkgutil
from functools import lru_cache



def _iter_module_infos():
    """
    Yield (fully_qualified_module_name, file_path) for every .py file
    in this package, recursing into subpackages.
    """
    for finder, module_name, is_pkg in pkgutil.walk_packages(
        __path__, prefix=f"{__name__}."
    ):
        if is_pkg:
            continue  # packages themselves aren't engine files; walk_packages recurses into them for us
        spec = finder.find_spec(module_name)
        if spec is None or spec.origin is None:
            continue
        yield module_name, spec.origin


def _engine_module_info(module_path):
    """
    For an engine module (one that sets IS_ENGINE_MODULE = True at top
    level): (name of its build_* function, names of its classes). None for
    any other module.

    Read from the source instead of importing the module, so that looking
    for engines imports nothing. A module is only imported once one of its
    engines is built (see build_engine()), so e.g. sklearn is only loaded
    when one of the classical classifiers is built.
    """
    with open(module_path, 'rb') as io:
        tree = ast.parse(io.read(), filename=module_path)

    is_engine_module = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "IS_ENGINE_MODULE"
                for t in node.targets)
        and isinstance(node.value, ast.Constant) and node.value.value is True
        for node in tree.body)
    if not is_engine_module:
        return None

    build_fns = [node.name for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name.startswith("build_")]
    if len(build_fns) != 1:
        raise RuntimeError(f"Expected exactly one build_* function in "
                           f"{module_path}, found {build_fns}")

    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    return build_fns[0], classes


@lru_cache(maxsize=None)
def _build_registry():
    """{engine class name: (module name, name of its build_* function)}"""
    registry = {}

    for module_name, module_path in _iter_module_infos():
        info = _engine_module_info(module_path)
        if info is None:
            continue
        build_fn, classes = info

        for class_name in classes:
            if class_name in registry and registry[class_name] != (module_name, build_fn):
                raise RuntimeError(
                    f"Duplicate class name '{class_name}' found in "
                    f"multiple engine modules"
                )
            registry[class_name] = (module_name, build_fn)

    return registry


class Engine:
    """
    Base class for all engines. Mainly for typing.
    """
    pass


def build_engine(**kwargs):
    name = kwargs.get("name")
    if name is None:
        raise ValueError("build_engine requires a 'name' kwarg")

    entry = _build_registry().get(name)
    if entry is None:
        raise Exception(f"Engine '{name}' not known in any module!")

    module_name, build_fn = entry
    module = importlib.import_module(module_name)
    return getattr(module, build_fn)(**kwargs)
