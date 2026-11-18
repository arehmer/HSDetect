"""The numpy AWA filter and the C++ one must produce the same frames.

hsdetect.kernels.awa.AWAFilter and cv::htpa::AWA_Filter are the same
algorithm written twice, once for Python and once for the device. That
duplication is deliberate -- it crosses a language boundary -- and it is only
acceptable while something holds the two to the same output. This is that
something.

The whole module skips where the custom HSOpenCV build is not installed,
because the PyPI opencv-python package carries no cv2.htpa. A skipped run is
not a passed run: until this has executed at least once, PreprocEngAWA's
switch from the C++ filter to the numpy one is unverified.
"""

import numpy as np
import pytest

from hsdetect import kernels as k

H, W = 120, 160
SENSOR_TYPE = 2
AWA = dict(K=2, S=2, a=1, eps=3)
N_FRAMES = 12


def _cpp_filter():
    cv2 = pytest.importorskip('cv2', reason='OpenCV is not installed')
    if not hasattr(cv2, 'htpa'):
        pytest.skip('cv2.htpa is missing: this is the PyPI opencv-python '
                    'build, not the custom HSOpenCV one')
    return cv2.htpa.PreprocEngAWA(SENSOR_TYPE, AWA['K'], AWA['S'],
                                  AWA['a'], AWA['eps'])


@pytest.fixture
def sequence():
    """A moving warm patch on a noisy ~3000 dK scene.

    Movement matters: the filter weights a support pixel by how much it
    differs from the pixel, so a static scene would exercise only the
    uniform-weight case.
    """
    rng = np.random.default_rng(3)
    frames = []
    for i in range(N_FRAMES):
        f = np.full((H, W), 3000, np.int32)
        y, x = 40 + 2 * i, 60 + 3 * i
        f[y:y + 18, x:x + 12] = 3085
        f += rng.integers(-5, 6, (H, W))
        frames.append(np.clip(f, 0, 65535).astype(np.uint16))
    return frames


def test_numpy_and_cpp_awa_agree_frame_by_frame(sequence):
    cpp = _cpp_filter()
    numpy_filter = k.AWAFilter(H, W, **AWA)

    compared = 0
    for i, frame in enumerate(sequence):
        cpp_idx, cpp_out = cpp.process(frame.copy(), i)
        np_idx, np_out = numpy_filter.filter(frame, i)

        assert cpp_idx == np_idx, f'index differs at call {i}'
        np.testing.assert_array_equal(
            np.asarray(cpp_out), np_out,
            err_msg=f'filtered frame differs at call {i}')
        compared += 1

    assert compared == N_FRAMES


def test_numpy_and_cpp_awa_agree_through_the_engine(sequence):
    """The same check through PreprocEngAWA, which is what the chain runs."""
    from hsdetect.cv.preproc_engines import PreprocEngAWA

    engine = PreprocEngAWA(SensorType=SENSOR_TYPE, **AWA)
    cpp = _cpp_filter()

    for i, frame in enumerate(sequence):
        out = engine._process_frame(frame, i)
        cpp_idx, cpp_out = cpp.process(frame.copy(), i)

        assert out['idx'] == cpp_idx
        np.testing.assert_array_equal(np.asarray(cpp_out), out['frame'])
