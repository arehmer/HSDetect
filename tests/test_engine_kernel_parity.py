"""The engines must be thin wrappers over hsdetect.kernels.

Each test here runs an engine and the kernel it is supposed to delegate to on
the same input and demands identical output. That is what stops the two from
drifting apart again: an engine that grows its own copy of the arithmetic
fails these.

What the engines keep for themselves is configuration and state, because
Engine.save() serializes a chain by walking properties. test_save_* below
pins that boundary from the other side: the kernels and their objects must
not leak into a saved chain.
"""

import numpy as np
import pandas as pd
import pytest

from hsdetect import kernels as k
from hsdetect.cv.preproc_engines import (BackgroundSub, MedianSub,
                                         PreprocEngAWA, ZeroCorners)

H, W, RADIUS = 120, 160, 81
SENSOR_TYPE = 2

AWA = dict(K=2, S=2, a=1, eps=3)
BG = dict(decay=0.95, thresh_update=10, box_pad=5)
N_INIT = 30


@pytest.fixture(scope='module')
def mask():
    return k.detection_mask(H, W, RADIUS)


@pytest.fixture(scope='module')
def sequence():
    """Forty frames of a ~3000 dK scene with noise and a warm patch."""
    rng = np.random.default_rng(11)
    base = np.full((H, W), 3000, np.int32)
    base[40:60, 70:100] = 3080
    return [np.clip(base + rng.integers(-4, 5, (H, W)), 0, 65535).astype(np.uint16)
            for _ in range(40)]


@pytest.fixture
def boxes():
    return pd.DataFrame([{'xtl': 70, 'ytl': 40, 'xbr': 100, 'ybr': 60}])


# ------------------------------ ZeroCorners ---------------------------------

def test_zero_corners_engine_matches_the_kernel(sequence, mask):
    engine = ZeroCorners(SensorType=SENSOR_TYPE)
    frame = sequence[0]

    out = engine._process_frame(frame, 0)['frame']

    expected = k.zero_corners(frame, mask)
    assert out.dtype == expected.dtype
    np.testing.assert_array_equal(out, expected)


def test_zero_corners_engine_matches_the_kernel_for_two_channels(sequence, mask):
    engine = ZeroCorners(SensorType=SENSOR_TYPE)
    stacked = np.stack([sequence[0].astype(np.float32),
                        sequence[1].astype(np.float32) - 3000.0], axis=-1)

    out = engine._process_frame(stacked, 0)['frame']

    expected = k.zero_corners(stacked, mask)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, expected)


def test_zero_corners_engine_passes_none_through():
    engine = ZeroCorners(SensorType=SENSOR_TYPE)

    assert engine._process_frame(None, 7)['frame'] is None


# ------------------------------- MedianSub ----------------------------------

def test_median_sub_engine_matches_the_kernels(sequence, mask):
    """MedianSub corrects channel 1 only, over the detection mask."""
    engine = MedianSub(SensorType=SENSOR_TYPE)
    zeroed = k.zero_corners(sequence[0], mask).astype(np.float32)
    stacked = np.stack([np.zeros((H, W), np.float32), zeroed], axis=-1)

    out = engine._process_frame(stacked, 0)['frame']

    level = k.estimate_scene_level(zeroed, mask)
    expected = k.subtract_scene_level(zeroed, level)
    np.testing.assert_array_equal(out[..., 1], expected)
    # Channel 0 is left alone.
    np.testing.assert_array_equal(out[..., 0], stacked[..., 0])


def test_median_sub_engine_does_not_modify_its_input(sequence, mask):
    engine = MedianSub(SensorType=SENSOR_TYPE)
    zeroed = k.zero_corners(sequence[0], mask).astype(np.float32)
    stacked = np.stack([np.zeros((H, W), np.float32), zeroed], axis=-1)
    before = stacked.copy()

    engine._process_frame(stacked, 0)

    np.testing.assert_array_equal(stacked, before)


# ----------------------------- BackgroundSub --------------------------------

def test_background_engine_matches_the_kernel_over_a_sequence(sequence, boxes):
    """Forty frames through the engine and through update_background() /
    subtract_background() directly, with boxes on alternating frames."""
    engine = BackgroundSub(SensorType=SENSOR_TYPE, N_init=N_INIT,
                           bg=None, bg_mask=None, stack_raw=True, **BG)
    empty = pd.DataFrame(columns=['xtl', 'ytl', 'xbr', 'ybr'])

    bg = bg_mask = None
    established = False

    for i, frame in enumerate(sequence):
        detections = boxes if i % 2 else empty

        if engine._bg_valid:
            established = True
            np.testing.assert_array_equal(engine.subtract(frame),
                                          k.subtract_background(frame, bg))

        engine.update_bg(frame, detections)
        bg, bg_mask = k.update_background(bg, bg_mask, frame, detections, **BG)

        np.testing.assert_allclose(np.asarray(engine.bg, float),
                                   np.asarray(bg, float),
                                   err_msg=f'bg diverged at frame {i}')
        np.testing.assert_array_equal(engine.bg_mask, bg_mask)

    assert established, 'the background never became valid -- test is vacuous'


def test_background_engine_withholds_output_until_established(sequence):
    engine = BackgroundSub(SensorType=SENSOR_TYPE, N_init=N_INIT,
                           bg=None, bg_mask=None, stack_raw=True, **BG)

    assert not engine._bg_valid
    with pytest.warns(UserWarning):
        assert engine.subtract(sequence[0]) is None


# ----------------------------- PreprocEngAWA --------------------------------

def test_awa_engine_matches_the_kernel(sequence):
    """The engine runs the numpy filter, not a second implementation."""
    engine = PreprocEngAWA(SensorType=SENSOR_TYPE, **AWA)
    reference = k.AWAFilter(H, W, **AWA)

    for i, frame in enumerate(sequence[:10]):
        out = engine._process_frame(frame, i)
        idx, filtered = reference.filter(frame, i)

        assert out['idx'] == idx
        np.testing.assert_array_equal(out['frame'], filtered)


def test_awa_engine_is_constructible_without_the_custom_opencv():
    """PreprocEngAWA used to build a cv2.htpa object in __init__, which made
    the whole inference chain need the custom OpenCV build."""
    engine = PreprocEngAWA(SensorType=SENSOR_TYPE, **AWA)

    assert isinstance(engine._awa, k.AWAFilter)
    assert 'cv_preproc_engine' not in engine.__dict__


def test_awa_engine_reset_clears_both_filters(sequence):
    engine = PreprocEngAWA(SensorType=SENSOR_TYPE, **AWA)
    for i, frame in enumerate(sequence[:2 * AWA['K'] + 1]):
        engine._process_frame(frame, i)

    # Stand in for a C++ instance built by __getattr__ without needing cv2.
    engine.__dict__['cv_preproc_engine'] = object()
    engine.reset()

    assert 'cv_preproc_engine' not in engine.__dict__
    out = engine._process_frame(sequence[0], 99)
    assert out['idx'] == 99, 'the frame buffer survived the reset'


def test_awa_engine_rejects_an_unknown_attribute():
    """__getattr__ must not turn every typo into something truthy."""
    engine = PreprocEngAWA(SensorType=SENSOR_TYPE, **AWA)

    with pytest.raises(AttributeError):
        engine.cv_preproc_engin


# --------------------------------- save() -----------------------------------

def test_save_does_not_leak_the_cpp_instance():
    """save() walks properties; cv_preproc_engine is deliberately not one,
    because a C++ object cannot be serialized into a chain's config."""
    engine = PreprocEngAWA(SensorType=SENSOR_TYPE, **AWA)
    engine.__dict__['cv_preproc_engine'] = object()

    saved = engine.save()

    assert 'cv_preproc_engine' not in saved
    assert saved['name'] == 'PreprocEngAWA'
    assert saved['K'] == AWA['K']


def test_save_does_not_leak_the_numpy_filter():
    engine = PreprocEngAWA(SensorType=SENSOR_TYPE, **AWA)

    saved = engine.save()

    assert '_awa' not in saved
    assert not any(isinstance(v, k.AWAFilter) for v in saved.values())


def test_background_save_still_carries_its_state(sequence):
    engine = BackgroundSub(SensorType=SENSOR_TYPE, N_init=N_INIT,
                           bg=None, bg_mask=None, stack_raw=True, **BG)
    engine.update_bg(sequence[0], pd.DataFrame(columns=['xtl', 'ytl',
                                                        'xbr', 'ybr']))

    saved = engine.save()

    assert saved['name'] == 'BackgroundSub'
    assert saved['decay'] == BG['decay']
    assert saved['N_init'] == N_INIT
    assert saved['bg'] is not None
