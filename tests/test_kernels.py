"""Behaviour of hsdetect.kernels, the shared numerical core.

These are the tests that keep the kernels honest on their own terms. The
parity tests that hold the engines and the training side to them live in
test_engine_kernel_parity.py and test_awa_conformance.py.
"""

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

from hsdetect import kernels as k

H, W, RADIUS = 120, 160, 81
SENSOR_TYPE = 2


@pytest.fixture(scope='module')
def frame():
    """A frame at a ~3000 dK scene level with a warm patch."""
    f = np.full((H, W), 3000, np.uint16)
    f[40:60, 70:100] = 3080
    return f


@pytest.fixture(scope='module')
def mask():
    return k.detection_mask(H, W, RADIUS)


def test_kernels_import_nothing_but_numpy():
    """The whole point of the package: no cv2, no hspy, no tensorflow.

    Read from the source rather than sys.modules, which another test may
    already have populated.
    """
    forbidden = {'cv2', 'hspy', 'hspytools', 'tensorflow', 'keras',
                 'sklearn', 'pandas', 'onnxruntime'}
    offenders = []

    for path in sorted(pathlib.Path(k.__file__).parent.glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split('.')[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or '').split('.')[0]]
            else:
                continue
            offenders += [f'{path.name}: {n}' for n in names if n in forbidden]

    assert not offenders, f'kernels must stay dependency-free: {offenders}'


# ------------------------------- mask ---------------------------------------

def test_detection_mask_matches_hspy(mask):
    """The mask the sensor library reports and the one computed here are the
    same array -- the geometry is not duplicated, it is asserted equal."""
    TPArray = pytest.importorskip('hspy.tparray').TPArray

    hspy_mask = np.asarray(TPArray(SensorType=SENSOR_TYPE)._detection_mask)

    assert mask.shape == hspy_mask.shape
    np.testing.assert_array_equal(mask, hspy_mask.astype(bool))


def test_zero_corners_2d_returns_uint16(frame, mask):
    out = k.zero_corners(frame, mask)

    assert out.dtype == np.uint16
    assert (out[~mask] == 0).all()
    np.testing.assert_array_equal(out[mask], frame[mask])


def test_zero_corners_3d_keeps_dtype_and_masks_every_channel(frame, mask):
    """From background subtraction on, a frame carries a second float32
    channel; casting it to uint16 would truncate it."""
    stacked = np.stack([frame.astype(np.float32),
                        frame.astype(np.float32) - 3000.0], axis=-1)

    out = k.zero_corners(stacked, mask)

    assert out.dtype == np.float32
    assert (out[~mask] == 0).all()


def test_zero_corners_does_not_modify_its_input(frame, mask):
    before = frame.copy()

    k.zero_corners(frame, mask)

    np.testing.assert_array_equal(frame, before)


def test_zero_corners_rejects_a_mismatched_mask(frame):
    with pytest.raises(ValueError):
        k.zero_corners(frame, np.ones((10, 10), bool))


# ------------------------------- scene --------------------------------------

def test_scene_level_ignores_zeros_and_pixels_outside_the_mask(frame, mask):
    """Zeros are the mask written by zero_corners, not measurements."""
    zeroed = k.zero_corners(frame, mask)

    assert k.estimate_scene_level(zeroed, mask) == 3000.0
    assert k.estimate_scene_level(zeroed) == 3000.0


def test_scene_level_is_none_when_nothing_is_valid():
    assert k.estimate_scene_level(np.zeros((H, W), np.uint16)) is None


def test_subtract_scene_level_keeps_zeros_zero(frame, mask):
    zeroed = k.zero_corners(frame, mask)

    out = k.subtract_scene_level(zeroed, 3000.0)

    assert out.dtype == np.float32
    assert (out[~mask] == 0).all()
    assert out[0, 0] == 0.0
    assert out[50, 80] == 80.0


def test_subtract_scene_level_passes_through_a_missing_level(frame):
    np.testing.assert_array_equal(k.subtract_scene_level(frame, None),
                                  frame.astype(np.float32))


def test_make_net_input_promotes_two_dimensions(frame):
    out = k.make_net_input(frame)

    assert out.shape == (H, W, 1)
    assert out.dtype == np.float32


def test_net_input_corrects_channel_one(frame, mask):
    zeroed = k.zero_corners(frame, mask)
    bgsub = np.zeros((H, W), np.uint16)

    out = k.net_input(bgsub, zeroed, mask)

    assert out.shape == (H, W, 2)
    assert out.dtype == np.float32
    # Channel 1 has had the scene level removed, channel 0 is untouched.
    assert out[50, 80, 1] == 80.0
    assert out[..., 0].max() == 0.0


# ----------------------------- background -----------------------------------

BG = dict(decay=0.95, thresh_update=10, box_pad=5)


def test_first_update_takes_the_frame_as_background(frame):
    bg, bg_mask = k.update_background(None, None, frame, None, **BG)

    np.testing.assert_array_equal(bg, frame.astype(np.float32))
    assert bg.dtype == np.float32
    assert (bg_mask == 255).all()


def test_small_changes_are_blended_with_the_decay(frame):
    bg, bg_mask = k.update_background(None, None, frame, None, **BG)
    nudged = (frame.astype(np.int32) + 5).astype(np.uint16)

    bg2, _ = k.update_background(bg, bg_mask, nudged, None, **BG)

    # 5 dK is below thresh_update, so it is blended in.
    expected = 0.95 * bg[0, 0] + 0.05 * nudged[0, 0]
    assert bg2[0, 0] == pytest.approx(expected, abs=1e-3)


def test_boxes_are_kept_out_of_the_background(frame):
    """A sub-threshold change is blended in everywhere EXCEPT inside a box,
    which is what keeps a person standing still out of the background."""
    bg, bg_mask = k.update_background(None, None, frame, None, **BG)
    nudged = (frame.astype(np.int32) + 5).astype(np.uint16)
    boxes = pd.DataFrame([{'xtl': 70, 'ytl': 40, 'xbr': 100, 'ybr': 60}])

    bg2, _ = k.update_background(bg, bg_mask, nudged, boxes, **BG)

    np.testing.assert_allclose(bg2[45:55, 75:95], bg[45:55, 75:95])
    assert bg2[0, 80] == pytest.approx(0.95 * bg[0, 80] + 0.05 * nudged[0, 80],
                                       abs=1e-3)


def test_changes_above_the_threshold_are_excluded_too(frame):
    """The guard against foreground that was never detected."""
    bg, bg_mask = k.update_background(None, None, frame, None, **BG)
    jumped = (frame.astype(np.int32) + 100).astype(np.uint16)

    bg2, _ = k.update_background(bg, bg_mask, jumped, None, **BG)

    np.testing.assert_allclose(bg2, bg)


def test_box_mask_accepts_a_dataframe_an_array_and_nothing():
    boxes = [[10, 20, 30, 40]]

    from_array = k.box_mask(np.array(boxes), H, W, 0)
    from_frame = k.box_mask(pd.DataFrame(boxes,
                                         columns=['xtl', 'ytl', 'xbr', 'ybr']),
                            H, W, 0)

    np.testing.assert_array_equal(from_array, from_frame)
    assert (from_array[20:40, 10:30] == 255).all()
    assert k.box_mask(None, H, W, 5).max() == 0
    assert k.box_mask(pd.DataFrame(columns=['xtl', 'ytl', 'xbr', 'ybr']),
                      H, W, 5).max() == 0


def test_box_mask_clips_to_the_image():
    mask = k.box_mask(np.array([[-20, -20, W + 20, H + 20]]), H, W, 5)

    assert (mask == 255).all()


def test_subtract_background_is_an_absolute_difference(frame):
    bg = frame.astype(np.float32)
    colder = (frame.astype(np.int32) - 7).astype(np.uint16)

    out = k.subtract_background(colder, bg)

    assert (out == 7).all()
    assert out.dtype == np.uint16


def test_subtractor_withholds_output_until_n_init_frames(frame):
    sub = k.BackgroundSubtractor(H, W, N_init=5, **BG)

    withheld = []
    for _ in range(6):
        withheld.append(sub.subtract(frame) is None)
        sub.update(frame, None)

    assert withheld[0] is True
    assert withheld[-1] is False
    assert sub.valid


# -------------------------------- AWA ---------------------------------------

AWA = dict(K=2, S=2, a=1, eps=3)


def test_awa_passes_frames_through_while_its_buffer_fills(frame):
    awa = k.AWAFilter(H, W, **AWA)

    for i in range(2 * AWA['K']):
        idx, out = awa.filter(frame, i)

        assert idx == i
        np.testing.assert_array_equal(out, frame)


def test_awa_lags_by_k_frames_once_full(frame):
    awa = k.AWAFilter(H, W, **AWA)

    indices = [awa.filter(frame, i)[0] for i in range(2 * AWA['K'] + 3)]

    # The first 2K calls return their own index, then the index from K ago.
    assert indices[:4] == [0, 1, 2, 3]
    assert indices[4:] == [2, 3, 4]


def test_awa_leaves_a_uniform_sequence_alone():
    """Every support pixel equals the pixel, so the weighted mean is the
    pixel. A filter that altered a static, uniform scene would be wrong."""
    uniform = np.full((H, W), 3000, np.uint16)
    awa = k.AWAFilter(H, W, **AWA)

    for i in range(2 * AWA['K'] + 1):
        idx, out = awa.filter(uniform, i)

    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, uniform)


def test_awa_smooths_a_noisy_sequence_towards_the_scene():
    """With noise around a constant level the filtered frame sits closer to
    that level than the frame it came from."""
    rng = np.random.default_rng(0)
    level = 3000
    awa = k.AWAFilter(H, W, **AWA)

    for i in range(2 * AWA['K'] + 1):
        noisy = (level + rng.integers(-6, 7, (H, W))).astype(np.uint16)
        idx, out = awa.filter(noisy, i)

    middle = awa._frames[AWA['K']]
    assert (np.abs(out.astype(float) - level).mean()
            < np.abs(middle.astype(float) - level).mean())


def test_awa_reset_empties_the_buffer(frame):
    awa = k.AWAFilter(H, W, **AWA)
    for i in range(2 * AWA['K'] + 1):
        awa.filter(frame, i)

    awa.reset()

    idx, out = awa.filter(frame, 99)

    assert idx == 99
    np.testing.assert_array_equal(out, frame)
