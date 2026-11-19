"""Where the tests find trained models, and what they do without them.

Trained weights are not committed (see
demos/live/multiprocessed/models/README.md): a run folder is the output of a
training run and belongs next to that run, not in this repository. The tests
that need one therefore have to be able to skip.

Point HSDETECT_MODELS at a directory of run folders to run them:

    HSDETECT_MODELS=T:/runs .venv/Scripts/python -m pytest

The directory is expected to hold one folder per run, each with the
inf_engine_params.json that training wrote and the YOLOv2Tiny8X.onnx it
refers to.
"""

import os
import pathlib

import pytest

MODELS_ENV = 'HSDETECT_MODELS'

# Where the pose classifiers and the chain configs live. Those are committed:
# they are small, they are text, and they describe the chain rather than being
# a trained network.
REPO_MODELS = pathlib.Path('demos/live/multiprocessed/models')


def models_dir() -> pathlib.Path:
    """The directory holding run folders, from HSDETECT_MODELS or in-repo."""
    override = os.environ.get(MODELS_ENV)

    return pathlib.Path(override) if override else REPO_MODELS


def model_file(run: str, name: str = 'YOLOv2Tiny8X.onnx') -> pathlib.Path:
    """Path of one run's network, wherever the run folders are."""
    return models_dir() / run / name


def chain_config(run: str, name: str = 'inf_engine_params.json') -> pathlib.Path:
    """Path of one run's engine chain config."""
    return models_dir() / run / name


def requires_run(*runs: str):
    """A skip marker for tests that need these run folders complete.

    Complete means both the chain config and the network it refers to are
    present, since building the chain opens the model.
    """
    missing = []

    for run in runs:
        for path in (chain_config(run), model_file(run)):
            if not path.exists():
                missing.append(path.as_posix())

    reason = (f'needs a trained model: {", ".join(missing)} not found. '
              f'Set {MODELS_ENV} to a directory of run folders.')

    return pytest.mark.skipif(bool(missing), reason=reason)


def skip_without_run(*runs: str) -> None:
    """Skip the test being executed unless these run folders are complete.

    The imperative form of requires_run(), for use inside a test or a helper
    where a decorator cannot see which run a parametrised case needs.
    """
    for run in runs:
        for path in (chain_config(run), model_file(run)):
            if not path.exists():
                pytest.skip(f'needs a trained model: {path.as_posix()} not '
                            f'found. Set {MODELS_ENV} to a directory of run '
                            f'folders.')
