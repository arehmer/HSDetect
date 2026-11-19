# Models

What is here, what is not, and how to get the rest.

## Committed

| | |
|---|---|
| `<run>/inf_engine_params.json` | the engine chain a training run exported. Small, text, and it *describes* the chain rather than being a trained network. |
| `<run>_live/inf_engine_params_scaled.json` | the same chain after `add_live_engines.py` reordered it and added the pose classifier. |
| `PoseClassifier/*.json` | pose classifiers in the portable format (see `hsdetect/cv/sklearn_portable_classifiers.py`). Text, and read with numpy alone. |

## Not committed

**The trained networks.** `YOLOv2Tiny8X.onnx` and the `.keras` weights beside
it are the output of a training run and belong next to that run. Six of them
used to be committed here — three unique models, each stored twice, because a
`<run>_live` folder kept a byte-identical copy of its parent's model — which
is 30 MB of working tree for three networks. `.gitignore` now keeps them out.

Note that removing them did not shrink any clone: git history still holds the
blobs. The rules are there so the *next* run folder does not get committed.

## Running the tests

The tests that need a network find run folders through `HSDETECT_MODELS` and
skip without one, naming the file they wanted:

```
HSDETECT_MODELS=T:/runs .venv/Scripts/python -m pytest
```

That directory is expected to hold one folder per run, each with the
`inf_engine_params.json` training wrote and the `YOLOv2Tiny8X.onnx` it refers
to — exactly the layout `yolotiny.training.save_detector()` produces.

Without it, the suite still runs everything that does not need a trained
network: the preprocessing kernels, the chain contract, `engine_io`, the
portable pose classifiers, and the engine/kernel parity tests. Roughly a
quarter of the suite skips.

To rebuild the ONNX-boundary golden fixture after a retrain, point
`HSDETECT_MODELS` at the run and see `tests/make_golden_onnx.py` — but read
its docstring first: regenerating it to make a test pass throws away the only
record of what the chain used to do.

## The `_live` folders

`add_live_engines.py` writes these. It loads a run's chain with
`validate=False`, because it exists to *repair* that chain: it reorders the
engines and inserts the pose classifier. Its output is validated when it is
next loaded.

`v12_adam/inf_engine_params.json` and `v12_sgd/inf_engine_params.json` put
`BackgroundSub` before `ZeroCorners`, which cannot work — the background is
fed a frame that is `None` until it is established, so it never is, and the
chain detects nothing at all without raising. Loading either now fails with a
message naming the two engines. They are kept as they are, deliberately: they
are what the check exists to catch. See `docs/decisions.md`.
