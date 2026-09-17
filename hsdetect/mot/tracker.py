# -*- coding: utf-8 -*-
"""
Online tracker: tracks with M-of-N confirmation, augmented Hungarian assignment,
IoU / Mahalanobis costs, and the cascaded per-frame loop.

Per frame: predict; split detections at tau_high; associate confirmed x high (A),
leftover confirmed x low (B), tentative x leftover high (C); update / miss;
spawn from unmatched high detections >= tau_init; confirm / delete; output.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from hsdetect.mot.config import TrackerConfig
from hsdetect.mot.features import compute_pair_features, cxcywh_to_xyxy, xyxy_to_cxcywh
from hsdetect.mot.kalman import KalmanBoxFilter, make_score_smoother

BIG = 1e6


class Track:

    def __init__(self, det, internal_id: int, kf: KalmanBoxFilter, cfg: TrackerConfig):
        self.mean, self.cov = kf.initiate(xyxy_to_cxcywh(det[:4]))
        self.score = make_score_smoother(cfg, det[4])
        self.class_id = int(det[5])
        self.internal_id = internal_id
        self.public_id = None                  # assigned at confirmation
        self.frames_since_update = 0
        self.age = 0
        self.hits = 1
        self.hit_history = deque([True], maxlen=cfg.N)
        self.confirmed = False
        self.matched = True

    def predict(self, kf, size_vel_damping=1.0):
        if self.frames_since_update and size_vel_damping != 1.0:
            self.mean[6:8] *= size_vel_damping
        self.mean, self.cov = kf.predict(self.mean, self.cov)
        self.frames_since_update += 1
        self.age += 1
        self.matched = False

    def update(self, det, kf):
        self.mean, self.cov = kf.update(self.mean, self.cov, xyxy_to_cxcywh(det[:4]), det[4])
        self.score.update(det[4])
        self.frames_since_update = 0
        self.hits += 1
        self.hit_history.append(True)
        self.matched = True

    def miss(self):
        self.score.miss()
        self.hit_history.append(False)

    def should_delete(self, cfg: TrackerConfig) -> bool:
        if self.confirmed:
            return self.frames_since_update > cfg.max_age
        remaining = max(cfg.N - (self.age + 1), 0)      # frames left in the first N
        return sum(self.hit_history) + remaining < cfg.M


def assign(cost, cost_no_match_track, cost_no_match_det, big=BIG):
    """Hungarian on the (T + D) x (D + T) matrix with explicit no-match options."""
    T, D = cost.shape
    if T == 0 or D == 0:
        return [], list(range(T)), list(range(D))
    aug = np.full((T + D, D + T), big)
    aug[:T, :D] = cost
    aug[:T, D:][np.diag_indices(T)] = cost_no_match_track
    aug[T:, :D][np.diag_indices(D)] = cost_no_match_det
    aug[T:, D:] = 0.0
    rows, cols = linear_sum_assignment(aug)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols) if r < T and c < D and cost[r, c] < big]
    unmatched_tracks = sorted(set(range(T)) - {r for r, _ in matches})
    unmatched_dets = sorted(set(range(D)) - {c for _, c in matches})
    return matches, unmatched_tracks, unmatched_dets


class IoUCost:
    def __init__(self, iou_min):
        self.iou_min = iou_min

    def __call__(self, pf):
        nm = (1 - self.iou_min) / 2
        return 1 - pf.iou, nm, nm


class MahalanobisCost:
    def __init__(self, maha_lambda):
        self.maha_lambda = maha_lambda

    def __call__(self, pf):
        nm = self.maha_lambda / 2
        return pf.d2, nm, nm


def build_cost_model(cfg: TrackerConfig, name=None):
    name = name or cfg.cost
    if name == "iou":
        return IoUCost(cfg.iou_min)
    if name == "mahalanobis":
        return MahalanobisCost(cfg.maha_lambda)
    raise ValueError(f"Unknown cost model {name!r}")


@dataclass(frozen=True)
class TrackOutput:
    public_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    is_coasting: bool


class Tracker:

    def __init__(self, config: TrackerConfig, similarity):
        self.cfg = config
        self.similarity = similarity        # callable: PairFeatures -> (cost, no_match_track, no_match_det)
        self.kf = KalmanBoxFilter(config)
        self.reset()

    def reset(self):
        self.tracks = []
        self._next_internal_id = 1
        self._next_public_id = 1
        self._last_frame = None

    def update(self, detections, frame_idx: int):
        """detections: [D, 6] = x1, y1, x2, y2, score, class_id. Skipped frames count as empty frames."""
        d = np.asarray(detections, dtype=float)
        d = d.reshape(-1, d.shape[-1]) if d.ndim == 2 else d.reshape(-1, 6)
        d = d[(d[:, 4] >= self.cfg.tau_low) & (d[:, 2] > d[:, 0]) & (d[:, 3] > d[:, 1])]
        if self._last_frame is not None:
            if frame_idx <= self._last_frame:
                raise ValueError("frame_idx must increase; call reset() for a new sequence")
            for _ in range(frame_idx - self._last_frame - 1):
                self._step(np.zeros((0, d.shape[1])))
        self._last_frame = frame_idx
        return self._step(d)

    def _associate(self, track_ids, det_ids, dets):
        if not track_ids or not det_ids:
            return [], track_ids, det_ids
        pf = compute_pair_features([self.tracks[i] for i in track_ids], dets[det_ids], self.kf, self.cfg)
        cost, nm_track, nm_det = self.similarity(pf)
        matches, un_t, un_d = assign(np.where(pf.allowed, cost, BIG), nm_track, nm_det)
        return ([(track_ids[r], det_ids[c]) for r, c in matches],
                [track_ids[r] for r in un_t], [det_ids[c] for c in un_d])

    def _step(self, dets):
        cfg = self.cfg
        for t in self.tracks:
            t.predict(self.kf, cfg.coast_size_vel_damping)

        high = np.flatnonzero(dets[:, 4] >= cfg.tau_high).tolist()
        low = np.flatnonzero(dets[:, 4] < cfg.tau_high).tolist()
        confirmed = [i for i, t in enumerate(self.tracks) if t.confirmed]
        tentative = [i for i, t in enumerate(self.tracks) if not t.confirmed]

        m_a, un_confirmed, un_high = self._associate(confirmed, high, dets)
        m_b, _, _ = self._associate(un_confirmed, low, dets)
        m_c, _, un_high = self._associate(tentative, un_high, dets)

        matched = dict(m_a + m_b + m_c)
        for i, t in enumerate(self.tracks):
            if i in matched:
                t.update(dets[matched[i]], self.kf)
            else:
                t.miss()

        for j in un_high:
            if dets[j, 4] >= cfg.tau_init:
                self.tracks.append(Track(dets[j], self._next_internal_id, self.kf, cfg))
                self._next_internal_id += 1

        for t in self.tracks:
            if not t.confirmed and sum(t.hit_history) >= cfg.M:
                t.confirmed = True
                t.public_id = self._next_public_id
                self._next_public_id += 1
        self.tracks = [t for t in self.tracks if not t.should_delete(cfg)]

        outputs = [TrackOutput(t.public_id, *map(float, cxcywh_to_xyxy(t.mean[:4])), t.score.value,
                               t.class_id, not t.matched)
                   for t in self.tracks
                   if t.confirmed and (t.matched or (t.frames_since_update <= cfg.max_coast_output
                                                     and t.score.value >= cfg.tau_out))]
        return sorted(outputs, key=lambda o: o.public_id)
