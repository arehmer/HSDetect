# -*- coding: utf-8 -*-
"""
Reading and writing inference engine parameter files.

The parameters of an inference engine (the dict returned by
InferenceEngine.save() and passed to build_engine()) are stored as JSON.

File paths in the parameters (the ONNX model of a DLEngine, the classifier
file of a BoxPoseClassifier) are stored relative to the parameter file, so a
model directory can be moved or copied to another machine as a whole. Which
parameters are paths is declared by the engine that owns them, in
Engine.path_params -- not by a list here, which had to know about every
engine and was only ever updated when something broke.

Numpy arrays are stored together with their dtype.

A parameter file is only useful if the chain it describes can actually be
built, so validate_params() does exactly that, and load_engine_params() calls
it by default. There is no separate schema: the engines and their setters are
the schema.
"""

import copy
import importlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np


# Numpy arrays are stored with their dtype, as {"ndarray": [...], "dtype": ...},
# so they are read back unchanged.
NDARRAY_KEY = 'ndarray'


@lru_cache(maxsize=None)
def _path_params(engine_name: str) -> tuple:
    """The parameters of one engine that hold a filesystem path.

    Read off the engine class, which is found through the registry in
    hsdetect.__init__. That imports the engine's module -- acceptable here,
    because anything reading or writing a parameter file is about to build or
    has just built that engine anyway.

    An unknown name yields no path parameters rather than raising: a dict that
    names no engine we know is left exactly as it is, and build_engine() is
    what complains about it.
    """
    from hsdetect import Engine, _build_registry

    entry = _build_registry().get(engine_name)

    if entry is None:
        return ()

    module_name, _ = entry
    engine_class = getattr(importlib.import_module(module_name),
                           engine_name, None)

    if not (isinstance(engine_class, type)
            and issubclass(engine_class, Engine)):
        return ()

    return tuple(engine_class.path_params)


def _path_params_of(d: dict) -> tuple:
    """The path parameters of whichever engine a parameter dict describes."""
    name = d.get('name')

    return _path_params(name) if isinstance(name, str) else ()


def _walk_dicts(obj):
    """All dicts nested anywhere in obj (lists and dicts), including obj."""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_dicts(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _walk_dicts(value)


def _to_json(obj):
    if isinstance(obj, dict):
        return {str(k): _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return {NDARRAY_KEY: obj.tolist(), 'dtype': obj.dtype.str}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return obj.as_posix()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f'{type(obj).__name__} can not be stored in an engine '
                    f'parameter JSON: {obj!r}')


def _from_json(obj):
    if isinstance(obj, dict):
        if set(obj) == {NDARRAY_KEY, 'dtype'}:
            return np.array(obj[NDARRAY_KEY], dtype=np.dtype(obj['dtype']))
        return {k: _from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_json(v) for v in obj]
    return obj


def json_dumps(obj, level=0) -> str:
    """JSON, indented, but lists of plain values on a single line."""
    pad, inner = '  ' * level, '  ' * (level + 1)
    if isinstance(obj, dict) and obj:
        items = [f'{inner}{json.dumps(k)}: {json_dumps(v, level + 1)}'
                 for k, v in obj.items()]
        return '{\n' + ',\n'.join(items) + '\n' + pad + '}'
    if isinstance(obj, list) and any(isinstance(v, (dict, list)) for v in obj):
        items = [inner + json_dumps(v, level + 1) for v in obj]
        return '[\n' + ',\n'.join(items) + '\n' + pad + ']'
    if isinstance(obj, list):
        return '[' + ', '.join(json.dumps(v, ensure_ascii=False) for v in obj) + ']'
    return json.dumps(obj, ensure_ascii=False)


def find_engine_params_file(directory) -> Path:
    """The first (alphabetical) '.json' file in directory."""
    files = sorted(Path(directory).glob('*.json'))
    if not files:
        raise FileNotFoundError(f'No engine parameter file (.json) in {directory}.')
    return files[0]


def validate_params(params) -> dict:
    """Check that these engine parameters describe a chain that can be built.

    Builds the chain and throws it away. There is deliberately no second,
    declarative schema to keep in step with the engines: building runs every
    engine's own setters and then InferenceEngine._check_chain(), so this one
    call catches a missing or misspelled parameter, a value of the wrong type,
    an engine name that does not exist, a chain whose engines do not fit
    together, and a model whose channel count disagrees with the
    preprocessing.

    Parameters
    ----------
    params : dict or path
        An InferenceEngine.save() result, or the path of a parameter file.

    Returns
    -------
    dict
        The parameters, unchanged, so a caller can validate and use in one
        expression.

    Raises
    ------
    Exception
        Whatever building the chain raises, unwrapped: those messages name the
        engine and the parameter, which a wrapper would only obscure.
    """
    from hsdetect import build_engine

    if not isinstance(params, dict):
        params = load_engine_params(params, validate=False)

    # A deep copy, because build_engine() pops from the dict it is given.
    build_engine(**copy.deepcopy(params))

    return params


def load_engine_params(file, validate: bool = True) -> dict:
    """
    Read engine parameters from a JSON file, ready for build_engine(**params).
    Relative paths (see Engine.path_params) are resolved against the file's
    directory.

    `validate`: build the chain once to check it (see validate_params), which
    is the default so that a broken file is rejected where it is read rather
    than somewhere downstream. Pass False when the parameters are about to be
    changed before being used -- demos/live/multiprocessed/add_live_engines.py
    reorders a chain and inserts engines, so what it loads does not have to be
    valid yet.
    """
    file = Path(file)
    with open(file, 'r', encoding='utf-8') as io:
        params = _from_json(json.load(io))

    for d in _walk_dicts(params):
        for key in _path_params_of(d):
            if isinstance(d.get(key), str):
                d[key] = (file.parent / d[key]).as_posix()

    if validate:
        validate_params(params)

    return params


def save_engine_params(params: dict, file) -> None:
    """
    Write engine parameters, e.g. from InferenceEngine.save(), to a JSON
    file. Paths (see Engine.path_params) are stored relative to the directory
    of the file; numpy arrays and tuples become lists.

    Does not validate: a caller assembling a chain in steps has to be able to
    write an intermediate state. Validate before writing, as
    yolotiny.training.save_detector() does, or load the result with
    validate=True.
    """
    file = Path(file)
    params = _to_json(params)

    for d in _walk_dicts(params):
        for key in _path_params_of(d):
            if isinstance(d.get(key), str):
                path = Path(d[key])
                if path.is_absolute():
                    path = Path(os.path.relpath(path, file.parent.resolve()))
                d[key] = path.as_posix()

    file.write_text(json_dumps(params) + '\n', encoding='utf-8')
