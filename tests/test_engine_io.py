"""Reading, writing and validating engine parameter files.

Stage 3 of the refactor: a parameter file is only useful if the chain it
describes can be built, so validate_params() builds it, and the two hardcoded
lists that engine_io used to keep -- PATH_KEYS and TUPLE_KEYS -- are gone.
Which parameters are paths is now declared by the engine that owns them; the
tuple list turned out to be redundant, because the engines' own setters
coerce.
"""

import copy
import json
import pathlib

import pytest

from hsdetect import Engine, build_engine
from hsdetect import engine_io
from hsdetect.engine_io import (load_engine_params, save_engine_params,
                                validate_params)

MODELS = pathlib.Path('demos/live/multiprocessed/models')
GOOD = MODELS / 'v14_sgd' / 'inf_engine_params.json'
BROKEN = MODELS / 'v12_adam' / 'inf_engine_params.json'


# --------------------------- the lists are gone -----------------------------

def test_the_hardcoded_key_lists_are_gone():
    """Both had to know about every engine. PATH_KEYS is replaced by
    Engine.path_params; TUPLE_KEYS was redundant."""
    assert not hasattr(engine_io, 'PATH_KEYS')
    assert not hasattr(engine_io, 'TUPLE_KEYS')


def test_path_params_are_read_from_the_engine_that_owns_them():
    assert engine_io._path_params('DLEngine') == ('dl_model',)
    assert engine_io._path_params('BoxPoseClassifier') == ('model_file',)
    assert engine_io._path_params('ZeroCorners') == ()
    assert Engine.path_params == ()


def test_an_unknown_engine_name_has_no_path_params():
    """A dict naming nothing we know is left alone; build_engine() is what
    complains about it."""
    assert engine_io._path_params('NoSuchEngine') == ()


def test_the_tuple_coercion_is_the_engines_job():
    """yolo_grid_size stays a list in the dict and becomes a tuple on the
    engine, which is why TUPLE_KEYS was not replaced by anything."""
    params = load_engine_params(GOOD)
    decoder = next(e for e in params['engines']
                   if e['name'] == 'YOLO_EnDecode_Engine')

    assert isinstance(decoder['yolo_grid_size'], list)

    chain = build_engine(**params)

    assert chain['YOLO_EnDecode_Engine'].yolo_grid_size == (15, 20)


# ------------------------------- validation ---------------------------------

def test_validate_accepts_a_good_chain_as_a_dict_and_as_a_path():
    params = load_engine_params(GOOD, validate=False)

    assert validate_params(params) is params
    assert validate_params(GOOD)['name'] == 'InferenceEngine'


def test_validate_rejects_a_broken_chain():
    params = load_engine_params(BROKEN, validate=False)

    with pytest.raises(ValueError, match='BackgroundSub'):
        validate_params(params)


def test_validate_does_not_mutate_what_it_is_given():
    """build_engine() pops from the dict it is handed, so validate_params()
    has to work on a copy -- otherwise validating would consume the caller's
    parameters."""
    params = load_engine_params(GOOD, validate=False)
    before = copy.deepcopy(params)

    validate_params(params)

    assert params == before
    # and the result is still buildable, i.e. nothing was consumed
    assert build_engine(**params).emits is not None


def test_validate_rejects_an_unknown_engine():
    params = load_engine_params(GOOD, validate=False)
    params['engines'][1]['name'] = 'NoSuchEngine'

    with pytest.raises(Exception, match='NoSuchEngine'):
        validate_params(params)


def test_validate_rejects_a_missing_parameter():
    params = load_engine_params(GOOD, validate=False)
    del params['engines'][0]['SensorType']

    with pytest.raises(Exception):
        validate_params(params)


def test_validate_rejects_a_chain_whose_model_wants_other_channels():
    """What _check_input_channels() used to do, now from the declarations:
    the model takes a stacked frame, so BackgroundSub must stack."""
    params = load_engine_params(GOOD, validate=False)
    background = next(e for e in params['engines']
                      if e['name'] == 'BackgroundSub')
    background['stack_raw'] = False

    with pytest.raises(ValueError, match='DLEngine'):
        validate_params(params)


# --------------------------------- loading ----------------------------------

def test_load_validates_by_default():
    with pytest.raises(ValueError):
        load_engine_params(BROKEN)


def test_load_can_skip_validation():
    """For a caller that is about to repair the chain, as
    demos/live/multiprocessed/add_live_engines.py does."""
    params = load_engine_params(BROKEN, validate=False)

    assert [e['name'] for e in params['engines']][:3] == \
        ['PreprocEngAWA', 'BackgroundSub', 'MedianSub']


def test_load_resolves_a_declared_path_against_the_file():
    params = load_engine_params(GOOD.resolve())
    dl_model = pathlib.Path(next(e for e in params['engines']
                                 if e['name'] == 'DLEngine')['dl_model'])

    assert dl_model.is_absolute()
    assert dl_model.exists()
    assert dl_model.parent == GOOD.resolve().parent


# --------------------------------- saving -----------------------------------

def test_save_stores_a_declared_path_relative_to_the_file(tmp_path):
    """So that a model directory can be moved or copied as a whole."""
    params = load_engine_params(GOOD.resolve())
    out = tmp_path / 'inf_engine_params.json'

    save_engine_params(params, out)

    written = json.loads(out.read_text(encoding='utf-8'))
    dl_model = next(e for e in written['engines']
                    if e['name'] == 'DLEngine')['dl_model']

    assert not pathlib.Path(dl_model).is_absolute()
    assert dl_model.endswith('YOLOv2Tiny8X.onnx')


def test_save_does_not_validate(tmp_path):
    """A caller assembling a chain in steps has to be able to write an
    intermediate state; the check happens when it is loaded again."""
    params = load_engine_params(BROKEN, validate=False)
    out = tmp_path / 'inf_engine_params.json'

    save_engine_params(params, out)

    assert out.exists()
    # Rejected on load. Not necessarily for the chain order: written into a
    # different directory, the model path dangles too, and whichever the
    # build hits first is the one that raises.
    with pytest.raises(Exception):
        load_engine_params(out)


def test_a_chain_round_trips_through_a_file(tmp_path):
    original = load_engine_params(GOOD.resolve())
    out = tmp_path / 'chain.json'

    # The model lives next to the original file, so keep the absolute path.
    save_engine_params(original, out)
    reloaded = load_engine_params(out, validate=False)

    names = [e['name'] for e in reloaded['engines']]
    assert names == [e['name'] for e in original['engines']]
