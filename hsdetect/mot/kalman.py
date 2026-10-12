# -*- coding: utf-8 -*-
"""
Constant-velocity Kalman filter over boxes, and the track score smoothers.

State [cx, cy, w, h, vcx, vcy, vw, vh], measurement [cx, cy, w, h]. Noise is
proportional to box size (BoT-SORT) and R is scaled by detection confidence
(StrongSORT NSA): R_k = max(1 - score, r_min) * R.
"""

from __future__ import annotations

import math

import numpy as np

from hsdetect.mot.config import TrackerConfig


class KalmanBoxFilter:

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)

    def _std(self, mean, sigma):
        w, h = mean[..., 2], mean[..., 3]
        return sigma * np.stack([w, h, w, h], axis=-1)

    def meas_var(self, mean):
        """Unscaled measurement noise variances [..., 4] for states [..., 8]."""
        return self._std(mean, self.cfg.sigma_pos) ** 2

    def nsa(self, score):
        return np.maximum(1.0 - np.asarray(score, dtype=float), self.cfg.r_min)

    def initiate(self, z):
        mean = np.concatenate([z, np.zeros(4)])
        mean[2:4] = np.maximum(mean[2:4], self.cfg.min_size)
        std = np.concatenate([2 * self._std(mean, self.cfg.sigma_pos),
                              10 * self._std(mean, self.cfg.sigma_vel)])
        return mean, np.diag(std ** 2)

    def predict(self, mean, cov):
        q = np.concatenate([self._std(mean, self.cfg.sigma_pos), self._std(mean, self.cfg.sigma_vel)]) ** 2
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + np.diag(q)
        mean[2:4] = np.maximum(mean[2:4], self.cfg.min_size)
        return mean, cov

    def innovation_cov(self, mean, cov, score):
        """S = H P H^T + R_k."""
        return cov[:4, :4] + np.diag(self.nsa(score) * self.meas_var(mean))

    def update(self, mean, cov, z, score):
        S = self.innovation_cov(mean, cov, score)
        K = np.linalg.solve(S, cov[:4, :]).T              # P H^T S^-1
        mean = mean + K @ (np.asarray(z, dtype=float) - mean[:4])
        cov = cov - K @ S @ K.T
        mean[2:4] = np.maximum(mean[2:4], self.cfg.min_size)
        return mean, 0.5 * (cov + cov.T)


class EMAScore:
    """s <- alpha * s + (1 - alpha) * c, with c = c_miss on frames without a match."""

    def __init__(self, cfg: TrackerConfig, score: float):
        self.cfg = cfg
        self.value = float(score)

    def update(self, score: float) -> None:
        self.value = self.cfg.alpha * self.value + (1 - self.cfg.alpha) * float(score)

    def miss(self) -> None:
        self.update(self.cfg.c_miss)


class LogitKalmanScore:
    """Random-walk Kalman filter on logit(score)."""

    EPS = 1e-3

    def __init__(self, cfg: TrackerConfig, score: float):
        self.cfg = cfg
        self.x = self._logit(score)
        self.p = cfg.logit_r

    def _logit(self, s):
        s = min(max(float(s), self.EPS), 1 - self.EPS)
        return math.log(s / (1 - s))

    @property
    def value(self) -> float:
        return 1 / (1 + math.exp(-self.x))

    def update(self, score: float) -> None:
        p = self.p + self.cfg.logit_q
        k = p / (p + self.cfg.logit_r)
        self.x += k * (self._logit(score) - self.x)
        self.p = (1 - k) * p

    def miss(self) -> None:
        self.update(self.cfg.c_miss)


def make_score_smoother(cfg: TrackerConfig, score: float):
    return (EMAScore if cfg.score_method == "ema" else LogitKalmanScore)(cfg, score)
