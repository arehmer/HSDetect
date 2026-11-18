# -*- coding: utf-8 -*-
"""
Adaptive weighted average filter.

Pure numpy, no cv2, no hspy: the single Python implementation of the AWA
filter, imported by hsdetect.cv.preproc_engines.PreprocEngAWA at inference
time and by HSDetectTrain's yolotiny while preparing training data, so that
both see the same filtered frames.

The C++ filter cv::htpa::AWA_Filter (HSOpenCV) implements the same algorithm
for the device. That is a port across a language boundary, and
tests/test_awa_conformance.py asserts the two agree frame by frame; it skips
where cv2.htpa is not installed.
"""
import numpy as np


class AWAFilter:
    """Adaptive weighted average filter, port of cv::htpa::AWA_Filter.

    Every pixel of the middle frame of a buffer of 2K+1 frames is replaced by
    a weighted mean over its (2S+1)^2 spatial neighbourhood in all buffered
    frames. A support pixel differing by d from the pixel gets the weight
    1 / (1 + a * max(eps^2, d^2)), so movement is filtered less.

    filter() returns (index, frame): while the buffer fills up (the first 2K
    calls) the unfiltered input frame with its own index, afterwards the
    filtered frame from K calls ago with its index.

    The arithmetic follows the C++ code step by step (float32 product,
    float64 accumulation in the same order), so the results are the same.
    """

    def __init__(self, height: int, width: int, K: int, S: int,
                 a: float, eps: int):
        self.height, self.width = int(height), int(width)
        self.K, self.S = int(K), int(S)
        self.a = np.float32(a)
        self.eps2 = int(eps) * int(eps)

        # Spatial offsets in the order of AWA_Filter::calc_spatial_support();
        # the support pixel of p is p - offset
        self._offsets = [(ox, oy) for oy in range(-self.S, self.S + 1)
                         for ox in range(-self.S, self.S + 1)]

        # Which support pixels lie inside the image, per offset
        S, H, W = self.S, self.height, self.width
        inside = np.pad(np.ones((H, W), bool), S, constant_values=False)
        self._valid = [inside[S - oy:S - oy + H, S - ox:S - ox + W]
                       for ox, oy in self._offsets]
        self.reset()

    def reset(self):
        self._frames = []     # int64 frames, oldest first
        self._indices = []

    def filter(self, frame: np.ndarray, index: int):
        frame = np.asarray(frame).astype(np.uint16).astype(np.int64)

        self._frames.append(frame)
        self._indices.append(int(index))
        n = 2 * self.K + 1
        if len(self._frames) > n:
            del self._frames[0], self._indices[0]

        if len(self._frames) < n:
            return int(index), frame.astype(np.uint16)

        S, H, W = self.S, self.height, self.width
        pixel = self._frames[self.K]
        padded = [np.pad(f, S) for f in self._frames]

        c = np.zeros((H, W), np.float64)
        p_filt = np.zeros((H, W), np.float64)
        for (ox, oy), valid in zip(self._offsets, self._valid):
            for f in padded:
                supp = f[S - oy:S - oy + H, S - ox:S - ox + W]
                diff = pixel - supp
                prod = self.a * np.maximum(self.eps2, diff * diff).astype(np.float32)
                h = 1.0 / (1.0 + prod.astype(np.float64))
                h = np.where(valid, h, 0.0)
                c += h
                p_filt += h * supp.astype(np.float32).astype(np.float64)

        out = (p_filt / c).astype(np.float32)
        out = np.clip(np.rint(out), 0, 65535).astype(np.uint16)
        return self._indices[self.K], out
