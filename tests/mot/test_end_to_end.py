"""Synthetic sequence with crossing boxes, jitter, dropped detections and false positives."""

import unittest

import numpy as np

from hsdetect.mot.config import TrackerConfig
from hsdetect.mot.data import assign_gt_identities
from hsdetect.mot.evaluate import run_sequence, smoothing_metrics
from hsdetect.mot.tracker import Tracker, build_cost_model

IMG_W, IMG_H, N_FRAMES = 160, 120, 150
# start cx, start cy, vx, vy, w, h, first frame
OBJECTS = [
    (5, 40, 1.0, 0.0, 10, 20, 0),       # crosses the next one around frame 79
    (155, 52, -0.9, 0.0, 11, 22, 0),
    (40, 5, 0.0, 0.7, 10, 21, 0),       # vertical, crosses the first and fourth
    (155, 92, -1.2, 0.0, 12, 22, 20),   # crosses the next one around frame 99
    (5, 100, 0.8, 0.0, 10, 19, 30),
]


def make_sequence(seed):
    rng = np.random.default_rng(seed)
    gt, dets = {}, {}
    for k in range(N_FRAMES):
        g, d = [], []
        for i, (cx, cy, vx, vy, w, h, t0) in enumerate(OBJECTS):
            x, y = cx + vx * (k - t0), cy + vy * (k - t0)
            if k < t0 or not (w / 2 <= x <= IMG_W - w / 2 and h / 2 <= y <= IMG_H - h / 2):
                continue
            g.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2, 0, i])
            if rng.random() < 0.15:
                continue                                                    # dropped detection
            jx, jy, jw, jh = rng.normal(0, 0.04, 4) * [w, h, w, h]
            score = rng.uniform(0.2, 0.5) if rng.random() < 0.15 else rng.uniform(0.55, 0.95)
            d.append([x + jx - (w + jw) / 2, y + jy - (h + jh) / 2, x + jx + (w + jw) / 2, y + jy + (h + jh) / 2, score, 0])
        for _ in range(rng.poisson(0.3)):                                   # false positives
            x, y = rng.uniform(10, IMG_W - 10), rng.uniform(15, IMG_H - 15)
            d.append([x - 5, y - 10, x + 5, y + 10, rng.uniform(0.1, 0.7), 0])
        gt[k], dets[k] = np.array(g).reshape(-1, 6), np.array(d).reshape(-1, 6)
    return gt, dets


def id_switches(gt, tracks):
    last, switches = {}, 0
    for f, g in gt.items():
        for row, identity in zip(tracks[f], assign_gt_identities(tracks[f], g)):
            if identity >= 0:
                switches += identity in last and last[identity] != row[6]
                last[identity] = row[6]
    return switches


class EndToEndTest(unittest.TestCase):

    def test_mahalanobis_no_id_switches_and_smoother_than_raw(self):
        cfg = TrackerConfig()
        tracker = Tracker(cfg, build_cost_model(cfg, "mahalanobis"))
        for seed in range(5):
            gt, dets = make_sequence(seed)
            tracks = run_sequence(tracker, dets)
            tracked = smoothing_metrics(gt, tracks)
            raw = smoothing_metrics(gt, {f: d[d[:, 4] >= cfg.tau_high] for f, d in dets.items()})
            with self.subTest(seed=seed):
                self.assertEqual(id_switches(gt, tracks), 0)
                self.assertLess(tracked["jitter"], 0.85 * raw["jitter"])
                self.assertLess(tracked["flicker"], raw["flicker"] / 4)

    def test_outputs_are_online(self):
        cfg = TrackerConfig()
        tracker = Tracker(cfg, build_cost_model(cfg, "mahalanobis"))
        _, dets = make_sequence(0)
        full = run_sequence(tracker, dets)
        prefix = run_sequence(tracker, {f: dets[f] for f in range(80)})
        for f in range(80):
            np.testing.assert_array_equal(full[f], prefix[f])


if __name__ == "__main__":
    unittest.main()
