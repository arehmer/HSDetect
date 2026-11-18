# -*- coding: utf-8 -*-
"""
Sensor geometry: which pixels carry data, and zeroing the ones that do not.

Pure numpy. The mask is a property of the sensor and its optics; hspy's
TPArray._detection_mask is the same array for a given SensorType, and
tests/test_kernels.py asserts they agree. These functions take height, width
and radius so that this module stays free of hspy, which imports its USB and
I2C driver stack eagerly.
"""
import numpy as np


def detection_mask(height: int, width: int, radius: float) -> np.ndarray:
    """True for pixels at most `radius` from the image centre (hspy
    TPArray._detection_mask)."""
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    r = np.sqrt((x - (width - 1) / 2) ** 2 + (y - (height - 1) / 2) ** 2)
    return r <= radius


def zero_corners(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy of `frame` with every pixel outside `mask` set to 0.

    A 2-D frame is returned as uint16, as the sensor delivers it. A 3-D frame
    is returned with its dtype unchanged and the mask applied to every
    channel: from background subtraction onwards a frame carries a second,
    already scene-level-corrected channel (see hsdetect.kernels.scene) and is
    float32, which a cast to uint16 would truncate.

    The input is never modified -- upstream engines may still hold a
    reference to the array that was passed in.
    """
    mask = np.asarray(mask).astype(bool)

    if frame.ndim == 2:
        out = frame.astype(np.uint16)
    elif frame.ndim == 3:
        out = frame.copy()
    else:
        raise ValueError(f'Expected a (H,W) or (H,W,C) frame, got shape '
                         f'{frame.shape}.')

    if mask.shape != out.shape[:2]:
        raise ValueError(f'Mask {mask.shape} does not match frame '
                         f'{out.shape[:2]}.')

    out[~mask] = 0
    return out
