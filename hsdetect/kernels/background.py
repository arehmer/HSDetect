# -*- coding: utf-8 -*-
"""
Running background estimate and background subtraction.

Pure numpy, no cv2, no hspy. The arithmetic is a line-by-line port of the
reference C++ BackSub::update_bg, and lives here so that the inference engine
(hsdetect.cv.preproc_engines.BackgroundSub) and the training-data preparation
in HSDetectTrain share one implementation rather than two that agree today.

The functions are pure: they take the background and return a new one, and
know nothing about how many frames have been seen. Deciding when a background
counts as established (N_init) is policy, not arithmetic, and stays with the
caller -- BackgroundSub keeps that bookkeeping in properties, because
Engine.save() walks properties to serialize a chain.
"""
import numpy as np


def box_mask(boxes, height: int, width: int, box_pad: int = 0) -> np.ndarray:
    """(H, W) uint8 mask, 255 inside any box enlarged by `box_pad`.

    `boxes` may be a DataFrame carrying the columns xtl, ytl, xbr, ybr, an
    (n, 4) array of those values in that order, or None / empty for a mask of
    zeros. Enlarged boxes are clipped to the image.
    """
    mask = np.zeros((int(height), int(width)), np.uint8)

    if boxes is None or len(boxes) == 0:
        return mask

    if hasattr(boxes, 'columns'):
        boxes = boxes[['xtl', 'ytl', 'xbr', 'ybr']].to_numpy()

    for xtl, ytl, xbr, ybr in np.asarray(boxes):
        xtl_c = max(0, int(xtl) - box_pad)
        ytl_c = max(0, int(ytl) - box_pad)
        xbr_c = min(int(width), int(xbr) + box_pad)
        ybr_c = min(int(height), int(ybr) + box_pad)
        mask[ytl_c:ybr_c, xtl_c:xbr_c] = 255

    return mask


def update_background(bg, bg_mask, frame: np.ndarray, boxes,
                      decay: float, thresh_update: float,
                      box_pad: int = 0):
    """Blend one frame into the background estimate.

    Pixels inside a box (enlarged by `box_pad`) are kept out of the update,
    as are pixels deviating from the current estimate by more than
    `thresh_update` -- the guard against foreground that was not detected.
    Every other pixel is blended in with an exponential moving average.

    Parameters
    ----------
    bg : np.ndarray or None
        Current estimate, (H, W). None means no frame has been seen yet: the
        frame itself becomes the estimate and no blending happens, exactly as
        on the first call of the C++ original.
    bg_mask : np.ndarray or None
        Accumulated mask of pixels ever validly observed. Ignored when `bg`
        is None.
    frame : np.ndarray
        (H, W) frame, cast to uint16 as the sensor delivers it.
    boxes
        Boxes to keep out of the background, as accepted by box_mask(): the
        ground truth while training, the model's own detections at inference
        time.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The new background, float32, clipped to the uint16 range, and the new
        accumulated mask.
    """
    frame = np.asarray(frame).astype(np.uint16)
    h, w = frame.shape[:2]
    det_mask = box_mask(boxes, h, w, box_pad)

    # First frame ever: take it as the background, blend nothing.
    if bg is None:
        return frame.astype(np.float32), np.bitwise_not(det_mask)

    bg = np.asarray(bg, dtype=np.float32)
    frame_f = frame.astype(np.float32)

    abs_diff = np.abs(frame_f - bg)
    diff_mask = np.where(abs_diff > thresh_update, np.uint8(255), np.uint8(0))
    det_mask = np.bitwise_or(det_mask, diff_mask)

    bg_mask = np.bitwise_or(bg_mask, np.bitwise_not(det_mask))

    # Excluded pixels keep the old estimate, valid ones take the frame.
    bg_cand = np.where(det_mask.astype(bool), bg, frame_f)
    bg = decay * bg + (1.0 - decay) * bg_cand

    return np.clip(bg, 0, 65535).astype(np.float32), bg_mask


def subtract_background(frame: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """|frame - bg| as uint16: channel 0 of the network input."""
    frame = np.asarray(frame).astype(np.uint16).astype(np.float32)
    return np.abs(frame - np.asarray(bg, dtype=np.float32)).astype(np.uint16)


class BackgroundSubtractor:
    """Stateful background estimate over a sequence of frames.

    The convenience wrapper around the functions above, for callers that walk
    a whole recording in order and have no configuration of their own to keep
    (HSDetectTrain's preprocess_video). The inference engine does not use this
    class: it holds the same state in properties so that a chain can be
    serialized and restored mid-stream.

    subtract() returns None until N_init frames have been seen.
    """

    def __init__(self, height: int, width: int, decay: float,
                 thresh_update: float, N_init: int, box_pad: int):
        self.height, self.width = int(height), int(width)
        self.decay = decay
        self.thresh_update = thresh_update
        self.N_init = N_init
        self.box_pad = box_pad
        self.reset()

    def reset(self):
        self.bg = None
        self.bg_mask = None
        self._n_init = 0
        self._valid = False

    @property
    def valid(self) -> bool:
        return self._valid

    def update(self, frame: np.ndarray, boxes) -> None:
        if frame is None:
            return

        first = self.bg is None

        self.bg, self.bg_mask = update_background(
            self.bg, self.bg_mask, frame, boxes,
            decay=self.decay, thresh_update=self.thresh_update,
            box_pad=self.box_pad)

        if first:
            self._n_init = 1
            return

        # Bookkeeping exactly as in the original (two sequential ifs).
        if not self._valid and self._n_init < self.N_init - 1:
            self._n_init += 1
        if not self._valid and self._n_init == self.N_init - 1:
            self._valid = True

    def subtract(self, frame: np.ndarray):
        if not self._valid:
            return None
        return subtract_background(frame, self.bg)
