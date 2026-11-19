# Decisions

Things that were decided and would otherwise only be visible as an absence.
Newest first.

## 2026-11-19 — Trained networks are not committed

The six `YOLOv2Tiny8X.onnx` files under `demos/live/multiprocessed/models`
were removed, together with a presentation that had ended up beside them.
They were three unique models stored twice each: a `<run>_live` folder kept a
byte-identical copy of its parent's network, because `add_live_engines.py`
expects the model in its output directory. 30 MB of working tree for three
networks, down to 2 MB.

`.gitignore` now excludes `*.onnx`, `*.keras` and `*.avideo`, which is the
part that actually matters: deleting the files did not shrink any clone, since
git history still holds the blobs. The rules stop the next run folder being
committed.

The tests find run folders through `HSDETECT_MODELS` and skip without one,
naming the file they wanted. Without models the suite runs 76 tests and skips
28; with `HSDETECT_MODELS` set it runs all 102. See
`demos/live/multiprocessed/models/README.md`.

## 2026-11-19 — ImportData.py stays on the old CVAT path for now

`HSDetectTrain/ImportData.py` uses `hsod.cvat.sdk.CVATProjectMgr`, and the
path going forward is `hsod.project.LabelingProject`. It was not moved,
because it cannot be moved on its own: everything the script produces is
`data/original/<cvat_id>.avideo`, pickled `AnnotatedVideo` objects, and
`LabelingProject`'s layout has no `.avideo` — it stores `sequences/<seq>.h5`
and `annotations/raw/<task>.json`.

The consumers are `yolotiny/dataset.py` and
`SkLearnPose/01_ImportAnnotations.py`, both via `common.avideo.load_avideo()`.
So the script, its two consumers and `common/avideo.py` move together, in one
piece of work that wants a recording and a TensorFlow environment at hand so
the training arrays can be compared before and after. `ImportData.py` carries
a header saying so.

## 2026-11-19 — Posture detection is dropped

`PostureNetEngine` and `hsdetect/cv/posture_engines.py` (1032 lines) were
removed in `ae34a55`, the commit that replaced TensorFlow with ONNX Runtime in
the runtime. The engine wrapped a Keras CNN, and it was not ported.

This is deliberate, not an oversight. Nothing in the repository references it:
no import, no engine chain names it, and the registry does not know it. The
posture CNN itself lives outside the repository and is unaffected.

If posture detection comes back, it comes back as a new engine running an ONNX
model through `onnxruntime`, with the chain contract of
`hsdetect.Engine` (`accepts` / `emits`, see `hsdetect/__init__.py`) — not by
restoring the deleted module, whose Keras dependency is the thing the runtime
was moved off. The old code is recoverable with
`git show ae34a55^:hsdetect/cv/posture_engines.py`.

Note that pose classification (`BoxPoseClassifier`, lying vs. non-lying from
box geometry) is a different thing and is still in use.

## 2026-11-19 — The pose classifier stays in the portable JSON format, not ONNX

`hsdetect/cv/sklearn_portable_classifiers.py` stores a trained scikit-learn
classifier as the numbers the model actually is — trees and baselines, or
support vectors with scaler and sigmoid calibration — and evaluates it in
numpy. So `BoxPoseClassifier` needs neither scikit-learn nor a particular
scikit-learn version at inference time.

ONNX was considered, since the detector already runs through `onnxruntime`.
Rejected for now because the gain is small and the cost is not: the existing
classifier files cannot be converted (they rebuild into numpy objects, not
scikit-learn estimators, so the pose training would have to be re-run), and
the feature computation stays in Python either way, so only the estimator
would move.

What *was* wrong, and is fixed, is that `HSDetectTrain/SkLearnPose/03_classify.py`
wrote a pickle holding the estimator's internal scikit-learn state, which the
runtime cannot read — unpickling needs scikit-learn and `__setstate__` pins the
version. It now writes the portable JSON with HSDetect's own writer.

The reimplementation is held to scikit-learn by
`tests/test_portable_classifiers.py` and, on the training side, by
`POSE_EXPORT_TOLERANCE` in `03_classify.py`. Measured agreement:
`HistGradientBoostingClassifier` exactly, the calibrated RBF SVM to ~1e-14.
