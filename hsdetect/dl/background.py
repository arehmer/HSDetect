"""
background.py

Standalone background construction and background-subtraction logic,
implemented completely independently of the YOLOv2-tiny model (model.py).
This corresponds to Section 3.2 ("Background Subtraction") and Algorithm 1
of Vandersteegen et al. (2023).

Two separate steps, as requested:

  1. Background construction / initialization / update
     -> `BackgroundModel.initialize()` and `BackgroundModel.update()`

  2. Background subtraction (producing the actual network input)
     -> `BackgroundModel.subtract()`

These operate on plain NumPy arrays (single frames, H x W, float), since
this preprocessing happens on the raw sensor stream *before* a frame is
ever handed to the TensorFlow model.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import copy
import warnings

from hspy.tparray import TPArray

try:
    from hsod.misc.annot import AnnotatedVideo, AnnotatedFrame
except:
    from hsdetect.annot.annot import AnnotatedVideo, AnnotatedFrame

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


BoundingBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2), pixel coords


def _enlarge_boxes(boxes: List[BoundingBox], H: int, W: int, pad: int = 1) -> List[BoundingBox]:
    """Enlarge every bounding box by `pad` pixels in every direction,
    clipped to the image boundaries. Matches Algorithm 1's
    `D <- enlarge_boxes(D)` step.
    """
    enlarged = []
    for (x1, y1, x2, y2) in boxes:
        enlarged.append((
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(W, x2 + pad),
            min(H, y2 + pad),
        ))
    return enlarged


def _boxes_to_mask(boxes: List[BoundingBox], H: int, W: int) -> np.ndarray:
    """Build the binary mask M in {0,1}^(HxW): 1 where a pixel overlaps
    any (enlarged) bounding box, 0 otherwise. Matches Algorithm 1's
    construction of M.
    """
    mask = np.zeros((H, W), dtype=np.float32)
    for (x1, y1, x2, y2) in boxes:
        xi1, yi1 = int(np.floor(x1)), int(np.floor(y1))
        xi2, yi2 = int(np.ceil(x2)), int(np.ceil(y2))
        mask[yi1:yi2, xi1:xi2] = 1.0
    return mask





if __name__ == "__main__":
    # Self-test with synthetic frames.
    H, W = 24, 32
    rng = np.random.default_rng(0)

    # Fake a stable background pattern.
    base_bg = rng.normal(loc=20.0, scale=0.2, size=(H, W)).astype(np.float32)

    bg_engine = BackgroundEngine(height=H, width=W, alpha=0.99, update_every_n_frames=1)

    init_frames = [base_bg + rng.normal(scale=0.05, size=(H, W)) for _ in range(3)]
    bg = bg_engine.initialize(init_frames)
    print("Initial background mean/std:", bg.mean(), bg.std())

    # Simulate a frame with a "hot" person-like blob and a GT box around it.
    frame = base_bg.copy()
    frame[10:14, 12:16] += 8.0  # warm blob
    gt_boxes = [(12, 10, 16, 14)]

    bg_engine.update(frame, gt_boxes)
    net_input = bg_engine.subtract(frame)

    print("Background-subtracted frame shape:", net_input.shape)
    print("Max response inside GT box (should be large, ~person signal):",
          net_input[10:14, 12:16].max())
    print("Max response outside GT box (should be small, ~near zero):",
          np.delete(net_input, np.s_[10:14], axis=0).max())