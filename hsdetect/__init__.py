# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:21:29 2023

@author: Rehmer
"""

import hsdetect.annot
import hsdetect.cv
import hsdetect.ipc

import importlib
import inspect
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


def _is_engine_module(module_name):
    module = importlib.import_module(module_name)
    return getattr(module, "IS_ENGINE_MODULE", False)


def _find_build_function(module):
    candidates = [
        obj for name, obj in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("build_") and obj.__module__ == module.__name__
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one build_* function in {module.__name__}, "
            f"found {[c.__name__ for c in candidates]}"
        )
    return candidates[0]


@lru_cache(maxsize=None)
def _build_registry():
    registry = {}

    for module_name, module_path in _iter_module_infos():
        if not _is_engine_module(module_name):
            continue

        module = importlib.import_module(module_name)

        if not hasattr(module, "get_classes"):
            raise RuntimeError(
                f"{module_name} has IS_ENGINE_MODULE set but no get_classes()"
            )

        build_fn = _find_build_function(module)

        for class_name in module.get_classes():
            if class_name in registry and registry[class_name] is not build_fn:
                raise RuntimeError(
                    f"Duplicate class name '{class_name}' found in "
                    f"multiple engine modules"
                )
            registry[class_name] = build_fn

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

    registry = _build_registry()
    build_fn = registry.get(name)
    if build_fn is None:
        raise Exception(f"Engine '{name}' not known in any module!")

    return build_fn(**kwargs)