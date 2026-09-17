# -*- coding: utf-8 -*-
"""
Box helpers and pair features with gating.

compute_pair_features() is the only place pair features are computed; the
tracker and the training-data simulator must both use it. To add appearance
features, return extra [T, D, k] blocks from extra_pair_features().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = (
    "res_cx", "res_cy", "res_w", "res_h",   # (z - Hx) / sqrt(diag S)
    "maha_d2",
    "log_w_ratio", "log_h_ratio",
    "iou",
    "det_score", "track_score",
    "class_match",
    "log_dt", "log_age", "log_hits",        # log1p(frames_since_update), log1p(age), log1p(hits)
    "confirmed",
)


def xyxy_to_cxcywh(b):
    b = np.asarray(b, dtype=float)
    w, h = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
    return np.stack([b[..., 0] + w / 2, b[..., 1] + h / 2, w, h], axis=-1)


def cxcywh_to_xyxy(b):
    b = np.asarray(b, dtype=float)
    return np.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2,
                     b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], axis=-1)


def iou_matrix(a, b):
    """IoU between xyxy boxes a [N, 4] and b [M, 4]."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def feature_names(cfg) -> list:
    names = list(FEATURE_NAMES)
    if cfg.class_pair_embedding:
        names += [f"class_pair_{a}_{b}" for a in range(cfg.num_classes) for b in range(cfg.num_classes)]
    return names


def extra_pair_features(tracks, dets, cfg) -> list:
    """Hook for appearance features: extra [T, D, k] blocks. None for geometry only."""
    return []


@dataclass
class PairFeatures:
    features: np.ndarray   # [T, D, F]
    allowed: np.ndarray    # [T, D] False = gated out
    d2: np.ndarray         # [T, D]
    iou: np.ndarray        # [T, D]


def compute_pair_features(tracks, dets, kf, cfg) -> PairFeatures:
    """
    Features of every (track, detection) pair relative to the Kalman-predicted
    track state, using S = H P H^T + max(1 - score, r_min) R for each pair.
    dets: [D, >=6] = x1, y1, x2, y2, score, class_id.
    """
    T, D = len(tracks), len(dets)
    if T == 0 or D == 0:
        F = len(feature_names(cfg))
        return PairFeatures(np.zeros((T, D, F)), np.zeros((T, D), bool), np.zeros((T, D)), np.zeros((T, D)))

    mean = np.stack([t.mean for t in tracks])
    cov = np.stack([t.cov for t in tracks])
    pred = mean[:, :4]
    z = xyxy_to_cxcywh(dets[:, :4])

    R = kf.nsa(dets[:, 4])[None, :, None] * kf.meas_var(mean)[:, None, :]      # [T, D, 4]
    S = cov[:, None, :4, :4] + R[..., None] * np.eye(4)                         # [T, D, 4, 4]
    r = z[None] - pred[:, None]                                                 # [T, D, 4]
    d2 = np.einsum("tdi,tdi->td", r, np.linalg.solve(S, r[..., None])[..., 0])
    iou = iou_matrix(cxcywh_to_xyxy(pred), dets[:, :4])

    det_cls = dets[:, 5].astype(int)
    trk_cls = np.array([t.class_id for t in tracks])
    same_cls = trk_cls[:, None] == det_cls[None, :]
    trk = np.array([[t.score.value, np.log1p(t.frames_since_update), np.log1p(t.age),
                     np.log1p(t.hits), t.confirmed] for t in tracks], dtype=float)

    blocks = [
        r / np.sqrt(np.diagonal(S, axis1=-2, axis2=-1)),
        d2[..., None],
        np.log(z[None, :, 2:4] / pred[:, None, 2:4]),
        iou[..., None],
        np.broadcast_to(dets[None, :, 4:5], (T, D, 1)),
        np.broadcast_to(trk[:, None, :1], (T, D, 1)),
        same_cls[..., None],
        np.broadcast_to(trk[:, None, 1:], (T, D, 4)),
    ]
    if cfg.class_pair_embedding:
        C = cfg.num_classes
        blocks.append(np.eye(C * C)[np.clip(trk_cls, 0, C - 1)[:, None] * C + np.clip(det_cls, 0, C - 1)[None, :]])
    blocks += extra_pair_features(tracks, dets, cfg)

    allowed = d2 <= cfg.chi2_gate
    if cfg.gate_by_class:
        allowed &= same_cls
    return PairFeatures(np.concatenate(blocks, axis=-1), allowed, d2, iou)
