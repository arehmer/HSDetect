import math
import unittest
from dataclasses import replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from hsdetect.mot.config import TrackerConfig
from hsdetect.mot.features import compute_pair_features
from hsdetect.mot.kalman import KalmanBoxFilter, make_score_smoother
from hsdetect.mot.tracker import Track, Tracker, assign, build_cost_model

CFG = TrackerConfig()


def box(x, y, w=10, h=20, score=0.9, cls=0):
    return [x, y, x + w, y + h, score, cls]


def dets(*rows):
    return np.array(rows, dtype=float).reshape(-1, 6)


def make_tracker(cost="mahalanobis", **overrides):
    cfg = replace(CFG, **overrides)
    return Tracker(cfg, build_cost_model(cfg, cost))


class KalmanTest(unittest.TestCase):
    kf = KalmanBoxFilter(CFG)

    def test_constant_velocity_target_converges(self):
        rng = np.random.default_rng(0)
        mean, cov = self.kf.initiate(np.array([20.0, 60.0, 10.0, 20.0]))
        for k in range(1, 80):
            mean, cov = self.kf.predict(mean, cov)
            z = np.array([20 + 1.5 * k, 60 - 0.5 * k, 10, 20]) + rng.normal(0, [0.3, 0.6, 0.2, 0.4])
            mean, cov = self.kf.update(mean, cov, z, 0.9)
        self.assertEqual((mean.shape, cov.shape), ((8,), (8, 8)))
        np.testing.assert_allclose(mean[4:6], [1.5, -0.5], atol=0.1)

    def test_covariance_grows_while_coasting(self):
        mean, cov = self.kf.initiate(np.array([50.0, 50.0, 10.0, 20.0]))
        traces = []
        for _ in range(10):
            mean, cov = self.kf.predict(mean, cov)
            traces.append(np.trace(cov[:4, :4]))
        self.assertTrue(np.all(np.diff(traces) > 0))

    def test_low_score_detection_pulls_less(self):
        mean, cov = self.kf.predict(*self.kf.initiate(np.array([50.0, 50.0, 10.0, 20.0])))
        z = mean[:4] + [2, 2, 0, 0]
        high, _ = self.kf.update(mean, cov, z, 0.95)
        low, _ = self.kf.update(mean, cov, z, 0.2)
        self.assertLess(np.linalg.norm(low[:2] - mean[:2]), np.linalg.norm(high[:2] - mean[:2]))


class AssignmentTest(unittest.TestCase):

    def test_augmented_beats_posthoc_thresholding(self):
        cost, nm = np.array([[1.0, 6.0], [2.0, 100.0]]), 2.5       # accept iff cost < 5
        rows, cols = linear_sum_assignment(cost)                    # raw Hungarian: cross assignment
        posthoc = [(r, c) for r, c in zip(rows, cols) if cost[r, c] < 2 * nm]
        self.assertEqual(posthoc, [(1, 0)])                         # loses the best pair (0, 0)
        self.assertEqual(assign(cost, nm, nm), ([(0, 0)], [1], [1]))

    def test_empty_inputs(self):
        self.assertEqual(assign(np.zeros((0, 3)), 1.0, 1.0), ([], [], [0, 1, 2]))
        self.assertEqual(assign(np.zeros((2, 0)), 1.0, 1.0), ([], [0, 1], []))


class LifecycleTest(unittest.TestCase):

    def test_confirmation_timing(self):
        t = make_tracker()
        self.assertEqual(t.update(dets(box(40, 40)), 0), [])
        self.assertEqual(t.update(dets(), 1), [])
        self.assertEqual(t.update(dets(box(40, 40)), 2), [])
        self.assertEqual([o.public_id for o in t.update(dets(box(40, 40)), 3)], [1])   # 3 hits in 4 frames

    def test_tentative_deleted_when_m_unreachable(self):
        t = make_tracker()
        t.update(dets(box(40, 40)), 0)
        t.update(dets(), 1)
        t.update(dets(), 2)
        self.assertEqual(len(t.tracks), 1)       # 1 hit + 2 remaining frames can still reach M=3
        t.update(dets(), 3)
        self.assertEqual(len(t.tracks), 0)

    def test_confirmed_deleted_after_max_age(self):
        t = make_tracker(max_age=3)
        for k in range(3):
            t.update(dets(box(40, 40)), k)
        t.update(dets(), 5)                      # skipped frame 4 counts as empty
        self.assertEqual(len(t.tracks), 1)
        t.update(dets(), 7)
        self.assertEqual(len(t.tracks), 0)


class CascadeTest(unittest.TestCase):

    def setUp(self):
        self.t = make_tracker("iou", chi2_gate=math.inf)
        for k in range(3):
            self.t.update(dets(box(40, 40)), k)
        (self.confirmed,) = self.t.tracks

    def test_tentative_cannot_steal_from_confirmed(self):
        self.t.update(dets(box(40, 40), box(42, 41)), 3)            # duplicate spawns a tentative track
        tentative = next(tr for tr in self.t.tracks if not tr.confirmed)
        self.t.update(dets(box(42, 41)), 4)                         # exactly on the tentative track
        self.assertEqual(self.confirmed.frames_since_update, 0)
        self.assertEqual(tentative.frames_since_update, 1)

    def test_confirmed_takes_low_score_detection(self):
        out = self.t.update(dets(box(40.5, 40, score=0.3)), 3)
        self.assertEqual(self.confirmed.frames_since_update, 0)
        self.assertEqual((len(self.t.tracks), len(out)), (1, 1))


class ScoreSmoothingTest(unittest.TestCase):

    def test_fades_during_misses_and_recovers(self):
        for method in ("ema", "logit_kalman"):
            s = make_score_smoother(replace(CFG, score_method=method), 0.9)
            faded = [s.miss() or s.value for _ in range(5)]
            recovered = [s.update(0.9) or s.value for _ in range(5)]
            with self.subTest(method=method):
                self.assertTrue(np.all(np.diff([0.9] + faded) < 0))
                self.assertTrue(np.all(np.diff([faded[-1]] + recovered) > 0))


class FeaturesTest(unittest.TestCase):

    def test_batch_matches_single_pairs_and_gating(self):
        kf = KalmanBoxFilter(CFG)
        tracks = [Track(np.array(box(40, 40)), 1, kf, CFG), Track(np.array(box(90, 30, cls=1)), 2, kf, CFG)]
        for tr in tracks:
            tr.predict(kf)
        d = dets(box(40.5, 40), box(90, 31, cls=1), box(90, 31, cls=0), box(0, 90))
        pf = compute_pair_features(tracks, d, kf, CFG)
        for i in range(2):
            for j in range(4):
                single = compute_pair_features([tracks[i]], d[j:j + 1], kf, CFG)
                np.testing.assert_allclose(single.features[0, 0], pf.features[i, j], rtol=1e-10)
        np.testing.assert_array_equal(pf.allowed, [[True, False, False, False],
                                                   [False, True, False, False]])   # class and distance gates


if __name__ == "__main__":
    unittest.main()
