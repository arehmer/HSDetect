# -*- coding: utf-8 -*-
"""Tracker configuration: one dataclass, loadable from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class TrackerConfig:
    # Detection thresholds
    tau_cache: float = 0.05          # decode threshold when caching YOLO predictions
    tau_bg: float = 0.5              # production threshold; only these boxes update the background
    tau_low: float = 0.1
    tau_high: float = 0.5
    tau_init: float = 0.6
    # Kalman filter
    sigma_pos: float = 1 / 20
    sigma_vel: float = 1 / 160
    r_min: float = 0.05
    min_size: float = 0.5            # predicted w, h are clamped to at least this (pixels)
    coast_size_vel_damping: float = 1.0
    # Score smoothing
    score_method: str = "ema"        # ema | logit_kalman
    alpha: float = 0.8
    c_miss: float = 0.0
    logit_q: float = 0.5
    logit_r: float = 1.0
    # Track lifecycle
    M: int = 3
    N: int = 5
    max_age: int = 30
    max_coast_output: int = 5
    tau_out: float = 0.3
    # Features and gating
    gate_by_class: bool = True
    chi2_gate: float = 13.2767       # chi-square 0.99 quantile, 4 dof
    class_pair_embedding: bool = False
    num_classes: int = 1
    # Association cost
    cost: str = "mahalanobis"        # iou | mahalanobis
    iou_min: float = 0.3
    maha_lambda: float = 9.4877      # chi-square 0.95 quantile, 4 dof
    logit_offset: float = 0.0
    # Ground truth assignment / evaluation
    iou_assign: float = 0.5

    def __post_init__(self):
        if not self.tau_cache <= self.tau_low <= self.tau_high <= self.tau_init:
            raise ValueError("Require tau_cache <= tau_low <= tau_high <= tau_init")
        if not 1 <= self.M <= self.N:
            raise ValueError("Require 1 <= M <= N")
        if self.score_method not in ("ema", "logit_kalman"):
            raise ValueError(f"Unknown score_method {self.score_method!r}")


def load_config(path) -> TrackerConfig:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    unknown = set(data) - {f.name for f in fields(TrackerConfig)}
    if unknown:
        raise KeyError(f"Unknown config keys: {sorted(unknown)}")
    return TrackerConfig(**data)


def save_config(cfg: TrackerConfig, path) -> None:
    import yaml

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
