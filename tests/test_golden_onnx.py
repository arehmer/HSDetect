"""Golden-sample test for the ONNX half of the engine chain.

Pins the behaviour of

    input tensor -> DLEngine (ONNX) -> YOLO_EnDecode_Engine -> boxes

against a recorded result, so that a refactor that changes this output says
so instead of quietly producing slightly different detections.

The fixture is built by tests/make_golden_onnx.py. If a test here fails,
first establish whether the change was intended; regenerating the fixture to
get a green run throws away the only record of what the chain used to do.

Not covered: PreprocEngAWA, BackgroundSub, MedianSub and ZeroCorners. See the
module docstring of make_golden_onnx.py.
"""

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from hsdetect import build_engine
from tests.model_paths import skip_without_run
from tests.make_golden_onnx import (CONF_THRESHOLD, MODEL_DIR, OUT_CSV,
                                    OUT_NPZ, engine_params, make_input)

# ONNX Runtime is not bit-reproducible across versions and execution
# providers; these are tight enough to catch a changed computation and loose
# enough to survive a runtime upgrade.
RAW_RTOL, RAW_ATOL = 1e-4, 1e-5
BOX_ATOL = 1e-3


@pytest.fixture(scope='module')
def golden():
    if not OUT_NPZ.exists():
        pytest.skip(f'{OUT_NPZ.name} missing -- run tests/make_golden_onnx.py')
    with np.load(OUT_NPZ) as data:
        return {'net_input': data['net_input'], 'raw_output': data['raw_output']}


@pytest.fixture(scope='module')
def golden_boxes():
    return pd.read_csv(OUT_CSV, index_col='id')


@pytest.fixture(scope='module')
def chain_params():
    return json.loads((MODEL_DIR / 'inf_engine_params.json').read_text())


@pytest.fixture(scope='module')
def dl_engine(chain_params):
    skip_without_run('v12_adam')
    params = dict(engine_params(chain_params, 'DLEngine'))
    params['dl_model'] = str(MODEL_DIR / params['dl_model'])
    return build_engine(**params)


@pytest.fixture(scope='module')
def decode_engine(chain_params):
    params = dict(engine_params(chain_params, 'YOLO_EnDecode_Engine'))
    params['conf_threshold'] = CONF_THRESHOLD
    return build_engine(**params)


def test_input_is_reproducible(golden):
    """The fixture's input tensor still comes out of make_input()."""
    np.testing.assert_array_equal(make_input(), golden['net_input'])


def test_dl_engine_matches_golden_raw_output(dl_engine, golden):
    """DLEngine turns the recorded input into the recorded network output."""
    raw = dl_engine({'frame': golden['net_input']})

    assert raw.shape == golden['raw_output'].shape
    np.testing.assert_allclose(raw, golden['raw_output'],
                               rtol=RAW_RTOL, atol=RAW_ATOL)


def test_decode_matches_golden_boxes(decode_engine, golden, golden_boxes):
    """YOLO_EnDecode_Engine turns the recorded network output into the
    recorded boxes."""
    boxes = decode_engine(golden['raw_output'])

    assert list(boxes.columns) == list(golden_boxes.columns)
    assert len(boxes) == len(golden_boxes)
    np.testing.assert_allclose(boxes.to_numpy(dtype=float),
                               golden_boxes.to_numpy(dtype=float),
                               atol=BOX_ATOL)


def test_thin_chain_end_to_end(dl_engine, decode_engine, golden, golden_boxes):
    """Both engines in sequence, as InferenceEngine runs them."""
    boxes = decode_engine(dl_engine({'frame': make_input()}))

    np.testing.assert_allclose(boxes.to_numpy(dtype=float),
                               golden_boxes.to_numpy(dtype=float),
                               atol=BOX_ATOL)


def test_decode_drops_everything_below_threshold(decode_engine, golden):
    """Every box that survives clears the confidence threshold -- the guard
    that keeps the whole grid from being returned as detections."""
    boxes = decode_engine(golden['raw_output'])

    assert (boxes['conf'] >= CONF_THRESHOLD).all()


def test_decode_handles_no_output(decode_engine):
    """None in, empty frame of the right shape out, as while BackgroundSub is
    still establishing its background."""
    boxes = decode_engine(None)

    assert len(boxes) == 0
    assert 'conf' in boxes.columns
