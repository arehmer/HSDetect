"""Regenerate the ONNX-boundary golden fixture.

    python tests/make_golden_onnx.py

Run this ONLY when the output is meant to change (a retrained model, a
deliberate change to decoding). A diff in the fixture that you did not intend
is the failure this fixture exists to catch -- do not regenerate to make
test_golden_onnx.py pass.

What is pinned
--------------
The second half of the engine chain, from the network input tensor to the
decoded boxes:

    input tensor -> DLEngine (ONNX) -> YOLO_EnDecode_Engine -> boxes

The first half (PreprocEngAWA, BackgroundSub, MedianSub, ZeroCorners) is NOT
covered: it needs cv2.htpa and 31+ consecutive frames, since BackgroundSub is
stateful with N_init=30, and the repository holds a single frame. A full-chain
fixture is a separate job, once preprocessing is pure numpy.

The input tensor
----------------
Derived from tests/data/Video2485236_Frame80_img.pkl, the one real sensor
frame in the repository, by a transform defined HERE rather than by the
preprocessing engines -- deliberately, so that this fixture keeps testing the
network and the decoding even while preprocessing is being refactored. It is
physically plausible, not the real preprocessing output.
"""

import json
import pathlib
import pickle

import numpy as np

from hsdetect import build_engine

REPO = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = REPO / 'demos' / 'live' / 'multiprocessed' / 'models' / 'v12_adam'
FRAME_PKL = REPO / 'tests' / 'data' / 'Video2485236_Frame80_img.pkl'
OUT_NPZ = REPO / 'tests' / 'data' / 'golden_onnx_v12_adam.npz'
OUT_CSV = REPO / 'tests' / 'data' / 'golden_onnx_v12_adam_boxes.csv'

CONF_THRESHOLD = 0.5


def engine_params(params: dict, name: str) -> dict:
    """The parameter dict of one engine of a saved chain, by name."""
    for entry in params['engines']:
        if entry.get('name') == name:
            return entry
    raise KeyError(f'{name} not in chain: '
                   f'{[e.get("name") for e in params["engines"]]}')


def make_input() -> np.ndarray:
    """A deterministic (H, W, 2) float32 network input from the real frame.

    Channel 0 is the frame with its scene level (the median) removed, which
    is what the network is trained on; channel 1 is the same signal with a
    coarse background -- a heavily blurred copy of itself, computed with a
    box filter in pure numpy -- removed, standing in for BackgroundSub.
    """
    frame = pickle.loads(FRAME_PKL.read_bytes()).astype(np.float64)

    scene = frame - np.median(frame)

    # 9x9 box blur via a summed-area table: a background estimate that needs
    # no cv2 and no second frame.
    pad = 4
    padded = np.pad(scene, pad, mode='edge')
    sat = padded.cumsum(0).cumsum(1)
    sat = np.pad(sat, ((1, 0), (1, 0)))
    k = 2 * pad + 1
    h, w = scene.shape
    blur = (sat[k:k + h, k:k + w] - sat[0:h, k:k + w]
            - sat[k:k + h, 0:w] + sat[0:h, 0:w]) / (k * k)

    return np.stack([scene, scene - blur], axis=-1).astype(np.float32)


def main() -> None:
    params = json.loads((MODEL_DIR / 'inf_engine_params.json').read_text())

    dl_params = dict(engine_params(params, 'DLEngine'))
    dl_params['dl_model'] = str(MODEL_DIR / dl_params['dl_model'])
    dl_engine = build_engine(**dl_params)

    decode_params = dict(engine_params(params, 'YOLO_EnDecode_Engine'))
    decode_params['conf_threshold'] = CONF_THRESHOLD
    decode_engine = build_engine(**decode_params)

    net_input = make_input()
    raw = dl_engine({'frame': net_input})
    boxes = decode_engine(raw)

    np.savez_compressed(OUT_NPZ, net_input=net_input, raw_output=raw)
    boxes.to_csv(OUT_CSV)

    print(f'input      {net_input.shape} {net_input.dtype}')
    print(f'raw output {raw.shape} {raw.dtype}  conf max {raw[..., 4].max():.6f}')
    print(f'boxes      {len(boxes)} rows at conf >= {CONF_THRESHOLD}')
    print(f'wrote {OUT_NPZ.relative_to(REPO)}')
    print(f'wrote {OUT_CSV.relative_to(REPO)}')


if __name__ == '__main__':
    main()
