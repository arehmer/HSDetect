"""The chain contract: Payload, Feedback and InferenceEngine._check_chain().

The bug this exists for: a chain with BackgroundSub before ZeroCorners built
without complaint, then fed update_bg() a None frame on every step -- so the
background never became valid, subtract() returned None forever, and the chain
produced no detections at all, silently, with no exception anywhere. Two
configs in this repository are ordered that way.

A chain that cannot work must now fail when it is built.
"""

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from hsdetect import Engine, Feedback, Payload, SAME, build_engine
from hsdetect.cv.inf_engines import InferenceEngine
from hsdetect.cv.preproc_engines import (BackgroundSub, MedianSub,
                                         PreprocEngAWA, ZeroCorners)
from hsdetect.engine_io import load_engine_params

MODELS = pathlib.Path('demos/live/multiprocessed/models')
SENSOR_TYPE = 2
H, W = 120, 160

AWA = dict(SensorType=SENSOR_TYPE, K=2, S=2, a=1, eps=3)
BG = dict(SensorType=SENSOR_TYPE, decay=0.95, thresh_update=10, N_init=30,
          box_pad=5, bg=None, bg_mask=None, stack_raw=True)


# --------------------------- the primitives ---------------------------------

def test_payload_knows_which_members_are_frames():
    assert Payload.FRAME_RAW.is_frame
    assert Payload.FRAME_MASKED.is_frame
    assert Payload.FRAME_STACKED.is_frame
    assert not Payload.TENSOR.is_frame
    assert not Payload.BOXES.is_frame


def test_feedback_needs_a_method_name():
    with pytest.raises(ValueError):
        Feedback('')
    with pytest.raises(ValueError):
        Feedback(None)


def test_bare_engine_is_untyped():
    assert Engine().accepted_payloads() == ()
    assert Engine.emits is None
    assert Engine.feedback is None


# ------------------- the contract must not be serializable ------------------

CONTRACT_FIELDS = ('accepts', 'emits', 'feedback')

TAGGED = [('hsdetect.cv.preproc_engines', 'PreprocEngAWA'),
          ('hsdetect.cv.preproc_engines', 'ZeroCorners'),
          ('hsdetect.cv.preproc_engines', 'BackgroundSub'),
          ('hsdetect.cv.preproc_engines', 'MedianSub'),
          ('hsdetect.cv.preproc_engines', 'ShrinkCenter'),
          ('hsdetect.dl.dl_engines', 'DLEngine'),
          ('hsdetect.dl.decode', 'YOLO_EnDecode_Engine'),
          ('hsdetect.cv.inf_engines', 'InferenceEngine'),
          ('hsdetect.cv.postproc_engines', 'PostprocEngineNMS'),
          ('hsdetect.cv.postproc_engines', 'BoxRescale'),
          ('hsdetect.cv.classifier_engine', 'BoxPoseClassifier')]


@pytest.mark.parametrize('module_name,class_name', TAGGED)
def test_contract_fields_are_not_properties(module_name, class_name):
    """save() serializes an engine by walking its PROPERTIES. A contract
    field that became a property would be written into every config file and
    then read back as a parameter, which is why these must stay plain
    attributes."""
    module = pytest.importorskip(module_name)
    cls = getattr(module, class_name)

    for field in CONTRACT_FIELDS:
        attr = getattr(cls, field, None)
        assert not isinstance(attr, property), (
            f'{class_name}.{field} is a property, so save() would write it '
            f'into the chain config')


def test_save_does_not_carry_the_contract():
    for engine in (PreprocEngAWA(**AWA), ZeroCorners(SensorType=SENSOR_TYPE),
                   BackgroundSub(**BG), MedianSub(SensorType=SENSOR_TYPE)):
        saved = engine.save()

        for field in CONTRACT_FIELDS:
            assert field not in saved, f'{engine.name}.save() leaked {field}'


def test_the_five_base_classes_share_one_save():
    """Five identical copies of the property walker were deleted; these must
    be inheriting the one on Engine, not a copy of their own."""
    from hsdetect.cv.preproc_engines import PreprocEngine

    assert PreprocEngine.save is Engine.save

    for module_name, class_name in [('hsdetect.cv.postproc_engines',
                                     'PostprocEngine'),
                                    ('hsdetect.cv.classifier_engine',
                                     'ClassifierEngine')]:
        module = pytest.importorskip(module_name)
        assert getattr(module, class_name).save is Engine.save


# --------------------------- engines as tagged ------------------------------

def test_background_sub_requires_the_corners_zeroed():
    """The declaration that makes the broken order detectable."""
    assert BackgroundSub(**BG).accepted_payloads() == (Payload.FRAME_MASKED,)


def test_background_sub_emits_follows_stack_raw():
    stacking = BackgroundSub(**{**BG, 'stack_raw': True})
    plain = BackgroundSub(**{**BG, 'stack_raw': False})

    assert stacking.emits is Payload.FRAME_STACKED
    assert plain.emits is Payload.FRAME_MASKED

    # And it follows a later change, because it is set in the setter.
    plain.stack_raw = True
    assert plain.emits is Payload.FRAME_STACKED


def test_dl_engine_accepts_what_its_model_was_built_for():
    """Replaces the old _check_input_channels(), which had to name DLEngine
    and BackgroundSub to compare their channel counts."""
    params = load_engine_params(MODELS / 'v14_sgd' / 'inf_engine_params.json')
    spec = next(e for e in params['engines'] if e['name'] == 'DLEngine')

    engine = build_engine(**spec)

    assert engine.input_shape[-1] == 2
    assert engine.accepted_payloads() == (Payload.FRAME_STACKED,)
    assert engine.emits is Payload.TENSOR


def test_dl_engine_without_a_model_falls_back_to_the_class_contract():
    from hsdetect.dl.dl_engines import DLEngine

    engine = DLEngine(SensorType=SENSOR_TYPE)

    assert set(engine.accepted_payloads()) == {Payload.FRAME_MASKED,
                                               Payload.FRAME_STACKED}


# ------------------------------ _check_chain --------------------------------

def _good_chain_specs():
    return load_engine_params(MODELS / 'v14_sgd' / 'inf_engine_params.json')


def test_a_well_ordered_chain_builds_and_resolves_its_payloads():
    chain = build_engine(**_good_chain_specs())

    assert chain.engine_names == ['PreprocEngAWA', 'ZeroCorners',
                                  'BackgroundSub', 'MedianSub', 'DLEngine',
                                  'YOLO_EnDecode_Engine']
    assert chain._engine_inputs['BackgroundSub'] is Payload.FRAME_MASKED
    assert chain._engine_inputs['DLEngine'] is Payload.FRAME_STACKED
    assert chain.emits is Payload.BOXES
    assert chain._boxes_source == 'YOLO_EnDecode_Engine'


def test_background_before_zerocorners_is_rejected():
    """The bug. Before the contract this built fine and then detected
    nothing, forever."""
    with pytest.raises(ValueError) as excinfo:
        InferenceEngine(engines=[PreprocEngAWA(**AWA), BackgroundSub(**BG)])

    message = str(excinfo.value)
    assert 'BackgroundSub' in message
    assert str(Payload.FRAME_MASKED) in message
    assert str(Payload.FRAME_RAW) in message


@pytest.mark.parametrize('config', ['v12_adam', 'v12_sgd'])
def test_the_committed_broken_configs_are_rejected_on_load(config):
    """load_engine_params() validates by default, so a broken file is
    rejected where it is read."""
    with pytest.raises(ValueError, match='BackgroundSub'):
        load_engine_params(MODELS / config / 'inf_engine_params.json')


@pytest.mark.parametrize('config', ['v12_adam', 'v12_sgd'])
def test_the_committed_broken_configs_are_rejected_on_build(config):
    """And still rejected when loaded unvalidated, which is what the
    add_live_engines scripts do before repairing the order."""
    params = load_engine_params(MODELS / config / 'inf_engine_params.json',
                                validate=False)

    with pytest.raises(ValueError, match='BackgroundSub'):
        build_engine(**params)


@pytest.mark.parametrize('config', ['v14_sgd'])
def test_the_committed_good_configs_build(config):
    chain = build_engine(**load_engine_params(MODELS / config
                                              / 'inf_engine_params.json'))

    assert chain.emits is Payload.BOXES


def test_duplicate_engine_names_are_rejected():
    """Engines are addressed by name, so one would hide the other."""
    with pytest.raises(ValueError, match='more than one engine'):
        InferenceEngine(engines=[PreprocEngAWA(**AWA),
                                 ZeroCorners(SensorType=SENSOR_TYPE),
                                 ZeroCorners(SensorType=SENSOR_TYPE)])


def test_feedback_without_a_boxes_producer_is_rejected():
    """BackgroundSub is updated with the chain's detections; a chain that
    stops before the decoder cannot give it any."""
    with pytest.raises(ValueError, match='no engine in the chain emits'):
        InferenceEngine(engines=[PreprocEngAWA(**AWA),
                                 ZeroCorners(SensorType=SENSOR_TYPE),
                                 BackgroundSub(**BG)])


# ---- test-local engines, for contracts no real engine has ----

class _Boxer(Engine):
    name = '_Boxer'
    accepts = Payload.FRAME_RAW
    emits = Payload.BOXES

    def __call__(self, data):
        return pd.DataFrame(columns=['xtl', 'ytl', 'xbr', 'ybr'])


class _WantsFrameFeedback(Engine):
    name = '_WantsFrameFeedback'
    accepts = Payload.BOXES
    emits = SAME
    feedback = Feedback('update', frame=True, boxes=False)

    def __call__(self, data):
        return data

    def update(self, frame):
        self.seen = frame


class _MissingMethod(Engine):
    name = '_MissingMethod'
    accepts = Payload.FRAME_RAW
    emits = Payload.BOXES
    feedback = Feedback('does_not_exist', frame=False, boxes=True)

    def __call__(self, data):
        return pd.DataFrame(columns=['xtl', 'ytl', 'xbr', 'ybr'])


def test_frame_feedback_on_an_engine_that_never_sees_a_frame_is_rejected():
    with pytest.raises(ValueError, match='has to be a frame'):
        InferenceEngine(engines=[_Boxer(), _WantsFrameFeedback()])


def test_a_feedback_method_that_does_not_exist_is_rejected():
    with pytest.raises(ValueError, match='does_not_exist'):
        InferenceEngine(engines=[_MissingMethod()])


def test_an_untyped_engine_passes_its_payload_through():
    """An engine without a contract must not break a chain it sits in."""
    class _Untyped(Engine):
        name = '_Untyped'

        def __call__(self, data):
            return data

    chain = InferenceEngine(engines=[PreprocEngAWA(**AWA),
                                     _Untyped(),
                                     ZeroCorners(SensorType=SENSOR_TYPE)])

    assert chain._engine_inputs['ZeroCorners'] is Payload.FRAME_RAW
    assert chain.emits is Payload.FRAME_MASKED


# ------------------------- the feedback edge, live --------------------------

def test_feedback_reaches_the_engine_without_naming_it():
    """update_engines() drives the declared Feedback, not three literal
    engine names: the background becomes valid while frames flow."""
    chain = build_engine(**_good_chain_specs())
    background = chain['BackgroundSub']

    rng = np.random.default_rng(2)
    base = np.full((H, W), 3000, np.int32)
    base[40:60, 70:100] = 3080

    assert not background._bg_valid

    for i in range(35):
        frame = np.clip(base + rng.integers(-4, 5, (H, W)),
                        0, 65535).astype(np.uint16)
        chain.step(frame, i)

    assert background._bg_valid, 'the feedback edge never updated the engine'
    assert background.bg is not None
    assert background.bg.shape == (H, W)


def test_ground_truth_overrides_the_chains_own_boxes():
    chain = build_engine(**_good_chain_specs())
    background = chain['BackgroundSub']

    rng = np.random.default_rng(4)
    base = np.full((H, W), 3000, np.int32)
    gt = pd.DataFrame([{'xtl': 70, 'ytl': 40, 'xbr': 100, 'ybr': 60}])

    for i in range(32):
        frame = np.clip(base + rng.integers(-4, 5, (H, W)),
                        0, 65535).astype(np.uint16)
        chain.step(frame, i, gt=gt)

    # The boxed region was held out of the estimate on every single step, so
    # it stayed at the first frame's values while the rest was blended.
    assert background._bg_valid


def test_step_records_every_engines_output_under_its_name():
    chain = build_engine(**_good_chain_specs())

    result = chain.step(np.full((H, W), 3000, np.uint16), 0)

    assert 'input' in result
    for name in chain.engine_names:
        assert name in result


# ------------------------- the chain's dependencies -------------------------

NO_CV2_PROGRAM = '''
import builtins, sys
import hspy.tparray                      # HSPy imports cv2 eagerly itself
for m in [m for m in sys.modules if m == "cv2" or m.startswith("cv2.")]:
    del sys.modules[m]
real = builtins.__import__
def fake(name, *a, **k):
    if name == "cv2" or name.startswith("cv2."):
        raise ImportError("the inference chain must not need cv2")
    return real(name, *a, **k)
builtins.__import__ = fake

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from hsdetect import build_engine
from hsdetect.engine_io import load_engine_params

chain = build_engine(**load_engine_params(r"{config}"))
for i in range(32):
    result = chain.step(np.full((120, 160), 3000, np.uint16), i)

assert chain["BackgroundSub"]._bg_valid, "background never became valid"
assert "cv2" not in sys.modules, "something imported cv2 after all"
print("OK")
'''


def test_the_inference_chain_needs_no_cv2():
    """The device runtime is numpy, onnxruntime and hspy -- not OpenCV.

    hsdetect.annot (and the custom cv2.htpa build behind it) belongs to
    preparing and reviewing data, so the frame-by-frame path imports it
    lazily and the dispatch guards keep the per-frame path away from it.

    Run in a subprocess: hiding cv2 inside this one would disturb every
    other test.
    """
    import subprocess
    import sys

    config = (MODELS / 'v14_sgd' / 'inf_engine_params.json').resolve()
    result = subprocess.run([sys.executable, '-c',
                             NO_CV2_PROGRAM.format(config=config)],
                            capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert 'OK' in result.stdout
