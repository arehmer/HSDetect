# -*- coding: utf-8 -*-
"""
Cached YOLO predictions and ground truth, one {frame_idx: rows} dict per sequence.

detections: x1, y1, x2, y2, score, class_id
GT:         x1, y1, x2, y2, class_id, identity
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from hsdetect.mot.config import TrackerConfig
from hsdetect.mot.features import iou_matrix


def save_frames(path, frames: dict) -> None:
    np.savez_compressed(path, frame_idx=np.array(list(frames)), counts=np.array([len(r) for r in frames.values()]),
                        rows=np.vstack(list(frames.values())))


def load_frames(path) -> dict:
    with np.load(path) as z:
        rows = np.split(z["rows"], np.cumsum(z["counts"])[:-1])
        return dict(zip(z["frame_idx"].tolist(), rows))


def cache_predictions(video, engine, decode_fn, cfg: TrackerConfig) -> dict:
    """
    Run a fresh hsod InferenceEngine over an AnnotatedVideo, decoding at tau_cache but
    passing only detections >= tau_bg to the background update, exactly as production
    does with conf_threshold = tau_bg. Greedy NMS never lets a lower-score box suppress
    a higher-score one, so the cached rows >= tau_bg equal the production detections.
    """
    import pandas as pd

    frames = {}
    for frame in video.frames:
        raw = np.asarray(frame.image).astype(np.uint16)
        bg_sub = engine.bg_engine.subtract(raw)
        if bg_sub is None:                                   # background not established yet
            engine.bg_engine.update_bg(raw, [])
            frames[int(frame.index)] = np.zeros((0, 6))
            continue
        y_pred = engine.model(bg_sub[None, :, :, None].astype(np.float32), training=False)[0]
        found = decode_fn(y_pred, engine.anchors, engine.img_h, engine.img_w, num_classes=engine.num_classes,
                          conf_threshold=cfg.tau_cache, nms_iou_threshold=engine.nms_iou_threshold,
                          max_detections=engine.max_detections)
        dets = np.array([(d.xtl, d.ytl, d.xbr, d.ybr, d.conf, d.class_id) for d in found], dtype=float).reshape(-1, 6)
        keep = dets[dets[:, 4] >= cfg.tau_bg]
        bg_boxes = pd.DataFrame(keep[:, [0, 1, 2, 3, 5, 4]], columns=["xtl", "ytl", "xbr", "ybr", "label", "score"])
        engine.bg_engine.update_bg(raw, bg_boxes.astype({"label": int}).rename_axis("id"))
        frames[int(frame.index)] = dets
    return frames


def gt_from_video(video, class_map=None) -> dict:
    """
    GT rows per frame from an AnnotatedVideo with CVAT track annotations.
    Rows with outside == True are dropped, identity is track_id, class comes from
    label_name via class_map. Boxes without a track and unknown labels raise.
    """
    class_map = class_map or {"person": 0}
    frames = {}
    for frame in video.frames:
        df = frame.annotations
        if df is None or len(df) == 0:
            frames[int(frame.index)] = np.zeros((0, 6))
            continue
        if "outside" in df.columns:
            df = df[~df["outside"].astype(bool)]
        if (df["track_id"] < 0).any():
            raise ValueError(f"Frame {frame.index}: annotation without track_id")
        unknown = set(df["label_name"]) - set(class_map)
        if unknown:
            raise ValueError(f"Frame {frame.index}: labels {unknown} not in class_map")
        frames[int(frame.index)] = np.column_stack([
            df[["xtl", "ytl", "xbr", "ybr"]].to_numpy(dtype=float),
            df["label_name"].map(class_map).to_numpy(dtype=float),
            df["track_id"].to_numpy(dtype=float)])
    return frames


def assign_gt_identities(dets, gt, iou_thr=0.5):
    """Class-aware Hungarian matching on IoU; identity per detection, -1 for false positives."""
    identity = np.full(len(dets), -1, dtype=np.int64)
    if len(dets) == 0 or len(gt) == 0:
        return identity
    iou = iou_matrix(dets[:, :4], gt[:, :4])
    valid = (iou >= iou_thr) & (dets[:, 5, None].astype(int) == gt[None, :, 4].astype(int))
    rows, cols = linear_sum_assignment(np.where(valid, 1 - iou, 1e6))
    ok = valid[rows, cols]
    identity[rows[ok]] = gt[cols[ok], 5]
    return identity
