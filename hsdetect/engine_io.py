# -*- coding: utf-8 -*-
"""
Reading and writing inference engine parameter files.

The parameters of an inference engine (the dict returned by
InferenceEngine.save() and passed to build_engine()) are stored as JSON.

File paths in the parameters (the Keras model of a DLEngine, the classifier
file of a BoxPoseClassifier, the checkpoint of a PostureNetEngine) are stored relative to the parameter file, so a
model directory can be moved or copied to another machine as a whole.
Numpy arrays are stored together with their dtype.
"""

import json
import os
from pathlib import Path

import numpy as np


# Parameters holding a file path. In a JSON file they are stored relative to
# the directory of the file.
PATH_KEYS = ('dl_model', 'model_file', 'model_path')

# Numpy arrays are stored with their dtype, as {"ndarray": [...], "dtype": ...},
# so they are read back unchanged.
NDARRAY_KEY = 'ndarray'

# Parameters engines expect as tuple. JSON has only lists, so they are
# converted back when reading.
TUPLE_KEYS = ('yolo_grid_size', 'frame_size')


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


def load_engine_params(file) -> dict:
    """
    Read engine parameters from a JSON file, ready for build_engine(**params).
    Relative paths (see PATH_KEYS) are resolved against the file's directory.
    """
    file = Path(file)
    with open(file, 'r', encoding='utf-8') as io:
        params = _from_json(json.load(io))

    for d in _walk_dicts(params):
        for key in PATH_KEYS:
            if isinstance(d.get(key), str):
                d[key] = (file.parent / d[key]).as_posix()
        for key in TUPLE_KEYS:
            if isinstance(d.get(key), list):
                d[key] = tuple(d[key])

    return params


def save_engine_params(params: dict, file) -> None:
    """
    Write engine parameters, e.g. from InferenceEngine.save(), to a JSON
    file. Paths (see PATH_KEYS) are stored relative to the directory of the
    file; numpy arrays and tuples become lists.
    """
    file = Path(file)
    params = _to_json(params)

    for d in _walk_dicts(params):
        for key in PATH_KEYS:
            if isinstance(d.get(key), str):
                path = Path(d[key])
                if path.is_absolute():
                    path = Path(os.path.relpath(path, file.parent.resolve()))
                d[key] = path.as_posix()

    file.write_text(json_dumps(params) + '\n', encoding='utf-8')
