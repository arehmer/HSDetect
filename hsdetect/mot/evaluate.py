# -*- coding: utf-8 -*-
"""
Run trackers over cached predictions, evaluate with TrackEval and report smoothing metrics.

    python -m hsdetect.mot.evaluate --cache-dir cache --out-dir results --costs iou mahalanobis [--config cfg.yaml]

The cache dir holds <seq>.dets.npz and <seq>.gt.npz written with data.save_frames().
MOTChallenge files (frame, id, x, y, w, h, score, -1, -1, -1) are written under
<out-dir>/trackeval/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hsdetect.mot.config import TrackerConfig, load_config
from hsdetect.mot.data import assign_gt_identities, load_frames
from hsdetect.mot.features import xyxy_to_cxcywh
from hsdetect.mot.tracker import Tracker, build_cost_model


def run_sequence(tracker: Tracker, dets: dict) -> dict:
    """Rows per frame: x1, y1, x2, y2, score, class_id, public_id."""
    tracker.reset()
    return {f: np.array([[o.x1, o.y1, o.x2, o.y2, o.score, o.class_id, o.public_id]
                         for o in tracker.update(d, f)], dtype=float).reshape(-1, 7)
            for f, d in dets.items()}


def write_mot(path: Path, frames: dict, is_gt=False) -> None:
    """Frames are numbered by position (1-based), so GT and results must share frame keys."""
    lines = []
    for k, rows in enumerate(frames.values(), start=1):
        for r in rows:
            box = f"{r[0]:.3f},{r[1]:.3f},{r[2] - r[0]:.3f},{r[3] - r[1]:.3f}"
            lines.append(f"{k},{int(r[5]) + 1},{box},1,1,1" if is_gt else f"{k},{int(r[6])},{box},{r[4]:.5f},-1,-1,-1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines))


def evaluate_trackeval(gt: dict, results: dict, work_dir: Path, iou_thr=0.5, classes=(0,)) -> list:
    """gt: {seq: frames}, results: {tracker: {seq: frames}}. HOTA, MOTA, IDF1, IDSW, Frag per class and sequence."""
    import trackeval

    table = []
    for c in classes:
        root = Path(work_dir) / f"class_{c}"
        for seq, g in gt.items():
            write_mot(root / "gt" / seq / "gt" / "gt.txt", {f: r[r[:, 4] == c] for f, r in g.items()}, is_gt=True)
            for name, res in results.items():
                write_mot(root / "trackers" / name / "data" / f"{seq}.txt", {f: r[r[:, 5] == c] for f, r in res[seq].items()})

        dataset_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_cfg.update(GT_FOLDER=str(root / "gt"), TRACKERS_FOLDER=str(root / "trackers"),
                           OUTPUT_FOLDER=str(root / "output"), TRACKERS_TO_EVAL=list(results), BENCHMARK="MOT17",
                           SPLIT_TO_EVAL="train", SKIP_SPLIT_FOL=True, PRINT_CONFIG=False,
                           SEQ_INFO={seq: len(g) for seq, g in gt.items()})
        eval_cfg = trackeval.Evaluator.get_default_eval_config()
        eval_cfg.update(USE_PARALLEL=False, PRINT_RESULTS=False, PRINT_CONFIG=False, TIME_PROGRESS=False,
                        OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False)
        metrics = [trackeval.metrics.HOTA(),
                   trackeval.metrics.CLEAR({"THRESHOLD": iou_thr, "PRINT_CONFIG": False}),
                   trackeval.metrics.Identity({"THRESHOLD": iou_thr, "PRINT_CONFIG": False})]
        res, _ = trackeval.Evaluator(eval_cfg).evaluate([trackeval.datasets.MotChallenge2DBox(dataset_cfg)], metrics)
        for name in results:
            for seq, r in res["MotChallenge2DBox"][name].items():
                r = r["pedestrian"]
                table.append({"tracker": name, "class_id": c, "sequence": seq,
                              "HOTA": float(np.mean(r["HOTA"]["HOTA"])), "MOTA": float(r["CLEAR"]["MOTA"]),
                              "IDF1": float(r["Identity"]["IDF1"]), "IDSW": int(r["CLEAR"]["IDSW"]),
                              "Frag": int(r["CLEAR"]["Frag"])})
    return table


def smoothing_metrics(gt: dict, hyp: dict, iou_thr=0.5) -> dict:
    """
    jitter:  mean |change of (hyp - gt) / [w, h, w, h]| between consecutive frames in which a
             GT identity is matched, i.e. box jitter with the object's own motion removed.
    flicker: matched/unmatched transitions per GT identity.
    hyp rows start with x1, y1, x2, y2, score, class_id.
    """
    prev, jitter, transitions, identities = {}, [], 0, set()
    for f, g in gt.items():
        h = hyp.get(f, np.zeros((0, 6)))
        ident = assign_gt_identities(h, g, iou_thr)
        cur = {}
        for row in g:
            i = int(row[5])
            identities.add(i)
            j = np.flatnonzero(ident == i)
            gbox = xyxy_to_cxcywh(row[:4])
            cur[i] = (xyxy_to_cxcywh(h[j[0], :4]) - gbox) / gbox[[2, 3, 2, 3]] if len(j) else None
            if i in prev:
                transitions += (prev[i] is None) != (cur[i] is None)
                if prev[i] is not None and cur[i] is not None:
                    jitter.append(np.mean(np.abs(cur[i] - prev[i])))
        prev = cur
    return {"jitter": float(np.mean(jitter)) if jitter else 0.0, "flicker": transitions / max(len(identities), 1)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--costs", nargs="+", default=["iou", "mahalanobis"])
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config) if args.config else TrackerConfig()
    cache = Path(args.cache_dir)
    seqs = sorted(path.name[:-len(".dets.npz")] for path in cache.glob("*.dets.npz"))
    dets = {s: load_frames(cache / f"{s}.dets.npz") for s in seqs}
    gt = {s: load_frames(cache / f"{s}.gt.npz") for s in seqs}

    results = {}
    for cost in args.costs:
        tracker = Tracker(cfg, build_cost_model(cfg, cost))
        results[cost] = {s: run_sequence(tracker, dets[s]) for s in seqs}

    raw = {s: {f: d[d[:, 4] >= cfg.tau_high] for f, d in dets[s].items()} for s in seqs}
    for name, res in [("raw_yolo", raw), *results.items()]:
        m = [smoothing_metrics(gt[s], res[s], cfg.iou_assign) for s in seqs]
        print(f"{name:<12} jitter={np.mean([x['jitter'] for x in m]):.4f} flicker={np.mean([x['flicker'] for x in m]):.2f}")
    for row in evaluate_trackeval(gt, results, Path(args.out_dir) / "trackeval", cfg.iou_assign):
        print(row)


if __name__ == "__main__":
    main()
