# -*- coding: utf-8 -*-
"""
posture_engines.py

A SELF-CONTAINED engine around Frederic's PyTorch posture classifier
("PostureNet", T:/Kunden/PureSec/Frederic/Heimann/CNNDetection). It interacts
with no other engine: it takes the raw sensor frame, removes the background
itself, classifies, post-processes, and emits annotations. Put it in a chain on
its own, or drive it directly without an InferenceEngine at all.

Every processing step is the CNNDetection implementation, not a reimplementation
of the same idea, and each one names the reference below. The reference is the
authority on all of them -- if a step here disagrees with it, this file is wrong.

  PRE-PROCESSING
  1. Background estimate -- backgroundRemover.py::computeBackground():
     the per-pixel MEAN of the first N frames (N = 5), which are assumed to show
     the empty room, rounded back to integer deciKelvin. Static: estimated once
     and never updated, which is what the reference does and what the model was
     trained against (training subtracted a known, constant floor temperature).
  2. Background removal -- backgroundRemover.py::removeBackground():
     NoBG = frame - background, as an exact SIGNED integer difference. Not an
     absolute value: the model's clip range deliberately keeps the negative side
     (CNNDetection/README.md Section 1.4).
  3. Model input -- renderCNN.py::toModelInput():
     clip(NoBG, -15, 70) / 15 as float32, shaped (1,1,120,160). A frozen global
     affine, never a per-frame normalisation -- a dK value has to map to the same
     network input in every frame.

  INFERENCE -- renderCNN.py::predict(), model/net.py::PostureNet:
     forward(x, tau=cfg.tauEnd, withAux=False) -> softmax -> (tag, conf, E[k]).
     The auxiliary heads are NOT run, exactly as the reference does not run them.

  POST-PROCESSING -- renderCNN.py, all three stages:
  4. MajorityTagVoter: the argmax of the summed probability vectors over a
     trailing window of MOTION_FRAME_GAP frames.
  5. MotionEstimator: the noise-gated frame difference over a frame GAP,
     normalised by body area and trailing-averaged.
  6. The ALARM gate (a demo of Section 1.13): majority tag is 'lying' AND motion
     is below ALARM_MOTION_MAX, i.e. fallen and still.

The network itself is rebuilt here from a declarative spec rather than imported,
because CNNDetection is not an installed package -- see _TRUNK_SPEC. torch is
imported lazily, so `import hsdetect` keeps working without it.

Deviations from the reference, both forced by running live rather than over a
finished recording, and nothing else:

  * The reference computes the background from the first N frames of a recording
    and then subtracts it from every frame of that recording, those N included.
    Live, a frame has to be answered when it arrives, and the first N-1 arrive
    before the background they would need exists; they return zero annotations
    and only feed the estimate. The Nth frame completes the estimate and IS
    classified with it, which is what keeps this as close to the reference as a
    live stream allows.
  * MotionEstimator and MajorityTagVoter are reset per sequence there. Here that
    is reset(), which also drops the background -- call it when a new stream
    starts.

@author: Rehmer
"""

from __future__ import annotations

import ast
import logging
from collections import deque

import numpy as np
import pandas as pd

from hsdetect import Engine

# Marker identifying this .py as a module containing engines
IS_ENGINE_MODULE = True

logger = logging.getLogger(__name__)


def build_posture_engine(**kwargs):
    name = kwargs.pop('name')
    if name == 'PostureNetEngine':
        return PostureNetEngine(**kwargs)
    else:
        raise TypeError(f'Unknown Posture Engine {name}!')


def get_classes():
    with open(__file__, 'r') as f:
        tree = ast.parse(f.read(), filename=__file__)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


class PostureNetEngine(Engine):
    """Background removal, posture CNN, and the reference's post-processing.

    Parameters
    ----------
    SensorType : int
        Encodes the type of HTPA. Carried for consistency with the other engines
        and for save()/build_engine() round-tripping only: this engine takes its
        frame geometry from the checkpoint, which is authoritative for what the
        network was trained on, and needs no TPArray.
    model_path : str or pathlib.Path
        A PostureNet checkpoint -- CNNDetection/checkpoints/best.pt, or any of
        the archived model.pt copies. Either a dict with a 'model' key (what
        train.py writes, which also carries the training Config) or a bare
        state_dict.
    bg_frames : int, optional
        Leading frames averaged into the background estimate. Default 5, which
        is backgroundRemover.py's DEFAULT_BG_FRAMES.
    background : np.ndarray or str, optional
        A pre-computed background in deciKelvin, or the path to one of
        backgroundRemover.py's ``background.np`` files. Given one, the engine
        skips its own estimation and is ready on the first frame.
    device : str, optional
        'auto' (default) -> cuda if available, else cpu. Or anything
        torch.device() accepts.
    tau : float, optional
        Attention-softmax temperature. Default is the checkpoint's tauEnd, which
        is what the reference runs inference at.
    motion_sigma, motion_gap, motion_smooth, noise_sigma_dK, alarm_motion :
        The reference's motion and alarm constants; see the class constants
        below for the defaults and what each one means.
    emit_no_person : bool, optional
        Emit a row for a frame classified ``no_person`` too, instead of zero
        rows. Default False -- an annotation means something is there, and zero
        rows is how this project's other decoder reports nothing. The prediction
        itself is unaffected and is always in `last`.
    """

    name = 'PostureNetEngine'

    # Tag set of CNNDetection/config.py, in order. The order IS the ordinal
    # index k of the criticality scale and of the E[k] output contract, so it
    # must not be reordered.
    TAG_NAMES = ('no_person',          # 0 -- floor + noise
                 'upright',            # 1 -- standing / walking
                 'furniture_sitting',  # 2 -- chair / sofa
                 'near_miss_floor',    # 3 -- sitting-floor / squat / kneel
                 'lying')              # 4 -- on the floor (the target)

    # --- reference constants, verbatim -------------------------------------- #
    # backgroundRemover.py
    DEFAULT_BG_FRAMES = 5
    # renderCNN.py: fixed per-pixel noise sigma (the generator's noiseSigmaDK,
    # measured on Data_Batch0), the gate in difference-sigma, the frame gap the
    # difference is taken over, the trailing window of the displayed value, and
    # the motion below which a lying majority raises the alarm.
    MOTION_NOISE_SIGMA_DK = 2.4
    MOTION_SIGMA_K = 4.0
    MOTION_FRAME_GAP = 10
    MOTION_SMOOTH_FRAMES = 5
    ALARM_MOTION_MAX = 8.0

    # The trunk of CNNDetection/model/net.py as data: (kind, attribute name, in
    # channels, out channels, kernel, stride), in execution order. 'stage' is a
    # ConvBnRelu, 'res' a SepResBlock. Driving both the module construction and
    # the forward pass off this one list is what keeps them in step, and the
    # attribute names are the state_dict prefixes -- change a name here and the
    # trained weights no longer load, which _load_weights() reports rather than
    # silently running on a partially initialized net.
    _TRUNK_SPEC = (
        ('stage', 'stage1', 1,   16,  5, 2),   # 16x60x80
        ('stage', 'stage2', 16,  32,  3, 2),   # 32x30x40
        ('res',   'res2',   32,  32,  3, 1),
        ('stage', 'stage3', 32,  64,  3, 2),   # 64x15x20
        ('res',   'res3',   64,  64,  3, 1),
        ('stage', 'stage4', 64,  128, 3, 2),   # 128x8x10
        ('res',   'res4',   128, 128, 3, 1),
    )

    # Feature width the head sits on, i.e. the last spec entry's out channels.
    _FEATURE_CHANNELS = 128

    # Fallbacks for the input contract, used only when a checkpoint carries no
    # Config (a bare state_dict). The live values come from the checkpoint.
    _DEFAULT_CFG = {'frameHeight': 120, 'frameWidth': 160,
                    'affineClipLoDK': -15.0, 'affineClipHiDK': 70.0,
                    'affineScaleDK': 15.0,
                    'useAttention': True, 'tauEnd': 1.0}

    # Read-only properties derived from the checkpoint. They are not constructor
    # parameters, so storing them in a parameter dict would only invite a
    # hand-edited value that build_engine() silently ignores.
    _SAVE_EXCLUDE = ('input_shape', 'affine', 'background_established')

    def __init__(self, SensorType: int, **attr_dict):

        self.SensorType = SensorType

        # ---- configuration (properties, so save() picks all of them up) ----
        self.device = attr_dict.pop('device', 'auto')
        self.bg_frames = attr_dict.pop('bg_frames', self.DEFAULT_BG_FRAMES)
        self.motion_sigma = attr_dict.pop('motion_sigma', self.MOTION_SIGMA_K)
        self.motion_gap = attr_dict.pop('motion_gap', self.MOTION_FRAME_GAP)
        self.motion_smooth = attr_dict.pop('motion_smooth',
                                           self.MOTION_SMOOTH_FRAMES)
        self.noise_sigma_dK = attr_dict.pop('noise_sigma_dK',
                                            self.MOTION_NOISE_SIGMA_DK)
        self.alarm_motion = attr_dict.pop('alarm_motion', self.ALARM_MOTION_MAX)
        self.emit_no_person = attr_dict.pop('emit_no_person', False)

        # Everything the reference computed for the most recent frame: tag,
        # conf, E[k], the probability vector, the majority tag, the motion and
        # the alarm, plus the attention map for diagnostics. Not a property --
        # per-frame state has no business in save().
        self.last = None

        # ---- the network ----------------------------------------------------
        # Setting model_path loads the checkpoint, builds the net and reads the
        # input contract out of the checkpoint's Config, so a wrong or
        # incompatible checkpoint fails here rather than on the first frame.
        self.model_path = attr_dict.pop('model_path', None)

        # tau defaults to the checkpoint's end-of-schedule temperature, which is
        # what the reference runs inference at, so it is resolved afterwards.
        self.tau = attr_dict.pop('tau', self._cfg['tauEnd'])

        # ---- per-stream state (background + the two temporal estimators) ----
        self.reset()

        # A caller-supplied background skips the estimation phase entirely.
        self.background = attr_dict.pop('background', None)

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def SensorType(self):
        return self._SensorType

    @SensorType.setter
    def SensorType(self, SensorType: int):
        self._SensorType = SensorType

    @property
    def model_path(self):
        """The checkpoint this engine was built from.

        Deliberately the PATH and not the loaded network: save() stores whatever
        a property returns, and a parameter dict has to stay picklable and
        rebuildable by build_engine(). Note that InferenceEngine.PATH_PARAMS
        does not list this engine, so a dict saved on Windows and built under
        WSL needs 'PostureNetEngine': ('model_path',) added there for
        remap_paths() to translate it.
        """
        return self._model_path

    @model_path.setter
    def model_path(self, model_path):
        self._model_path = None if model_path is None else str(model_path)
        self._net = None
        self._cfg = dict(self._DEFAULT_CFG)

        if self._model_path is not None:
            self._build_net()

    @property
    def background(self):
        """The background estimate in deciKelvin, or None while it is building.

        Assigning an array (or a path to one of backgroundRemover.py's
        ``background.np`` files) installs it and ends the estimation phase; None
        clears it and starts estimation over.
        """
        return self._background

    @background.setter
    def background(self, background):
        if background is None:
            self._background = None
            self._bg_buffer = []
            return

        if isinstance(background, (str, bytes)) or hasattr(background, '__fspath__'):
            # backgroundRemover.py writes these with pickle.dump(), so they are
            # pickled ndarrays rather than .npy -- allow_pickle is required, and
            # a stray singleton axis is squeezed, exactly as its loadFrame does.
            background = np.asarray(np.load(background, allow_pickle=True)).squeeze()

        background = self._as_frame(np.asarray(background))

        # rint-then-int64, as computeBackground() returns, so that the
        # difference stays the exact integer difference the reference writes.
        self._background = np.rint(background).astype(np.int64)
        self._bg_buffer = []

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: str):
        self._device = device

    @property
    def bg_frames(self):
        return self._bg_frames

    @bg_frames.setter
    def bg_frames(self, bg_frames: int):
        bg_frames = int(bg_frames)
        if bg_frames < 1:
            raise ValueError(f'bg_frames must be at least 1, received '
                             f'{bg_frames}.')
        self._bg_frames = bg_frames

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau: float):
        tau = float(tau)
        if tau <= 0:
            raise ValueError(f'tau must be positive, received {tau}.')
        self._tau = tau

    @property
    def motion_sigma(self):
        return self._motion_sigma

    @motion_sigma.setter
    def motion_sigma(self, motion_sigma: float):
        self._motion_sigma = float(motion_sigma)

    @property
    def motion_gap(self):
        return self._motion_gap

    @motion_gap.setter
    def motion_gap(self, motion_gap: int):
        # max(1, ...) mirrors MotionEstimator and MajorityTagVoter, which both
        # clamp their window the same way.
        self._motion_gap = max(1, int(motion_gap))
        # The reference sizes both the difference buffer and the vote window by
        # the gap, so a change here has to resize the live deques too -- a deque
        # keeps the maxlen it was built with.
        if hasattr(self, '_motion_buffer'):
            self._motion_buffer = deque(self._motion_buffer,
                                        maxlen=self._motion_gap)
            self._vote_window = deque(self._vote_window,
                                      maxlen=self._motion_gap)

    @property
    def motion_smooth(self):
        return self._motion_smooth

    @motion_smooth.setter
    def motion_smooth(self, motion_smooth: int):
        self._motion_smooth = max(1, int(motion_smooth))
        if hasattr(self, '_motion_smooth_buffer'):
            self._motion_smooth_buffer = deque(self._motion_smooth_buffer,
                                               maxlen=self._motion_smooth)

    @property
    def noise_sigma_dK(self):
        return self._noise_sigma_dK

    @noise_sigma_dK.setter
    def noise_sigma_dK(self, noise_sigma_dK: float):
        self._noise_sigma_dK = float(noise_sigma_dK)

    @property
    def alarm_motion(self):
        return self._alarm_motion

    @alarm_motion.setter
    def alarm_motion(self, alarm_motion: float):
        self._alarm_motion = float(alarm_motion)

    @property
    def emit_no_person(self):
        return self._emit_no_person

    @emit_no_person.setter
    def emit_no_person(self, emit_no_person: bool):
        self._emit_no_person = bool(emit_no_person)

    @property
    def input_shape(self):
        """(height, width) of the frame the loaded checkpoint expects."""
        return (self._cfg['frameHeight'], self._cfg['frameWidth'])

    @property
    def affine(self):
        """(clip_lo_dK, clip_hi_dK, scale_dK) of the frozen input affine."""
        return (self._cfg['affineClipLoDK'],
                self._cfg['affineClipHiDK'],
                self._cfg['affineScaleDK'])

    @property
    def background_established(self):
        """Whether enough frames have been seen to classify."""
        return self._background is not None

    # ------------------------------------------------------------------ #
    #  Engine interface
    # ------------------------------------------------------------------ #

    def __call__(self, data, **kwargs):
        """Run one raw frame through the whole chain, and return its annotations.

        Parameters
        ----------
        data : dict or np.ndarray
            In an InferenceEngine chain this is the chain's input dict, which
            carries the RAW frame under 'frame' and its index under 'idx'. A
            bare array is accepted too, for driving this engine directly.

        Returns
        -------
        pd.DataFrame
            At most one row -- PostureNet classifies the whole frame, so there is
            nothing to detect more than once of. Zero rows while the background
            is still being estimated, and for a frame classified ``no_person``
            unless emit_no_person is set. Columns: this project's annotation
            columns, then the reference's own outputs ('label', 'E_k',
            'majority', 'motion', 'alarm').
        """
        return self.process(data, **kwargs)

    def process(self, data, **kwargs):
        """call() for the dl engines, process() for the cv engines."""
        frame = data.get('frame') if isinstance(data, dict) else data

        if frame is None:
            return self._empty_annotations()

        frame = self._as_frame(np.asarray(frame))

        # Estimation phase: the frame feeds the background and nothing else.
        # Deliberately before the temporal estimators, which must not see a
        # frame that has not had its background removed.
        if not self.update_background(frame):
            self.last = {}
            return self._empty_annotations()

        frame_no_bg = self.remove_background(frame)

        # The reference's order in renderSituation(): predict, then vote, then
        # motion. The two estimators are fed on EVERY classified frame, whatever
        # the tag, or their windows no longer line up with the stream.
        tag, conf, e_k, probs = self.predict(frame_no_bg)
        majority = self.update_majority(probs)
        motion = self.update_motion(frame_no_bg)

        alarm = (majority == 'lying'
                 and motion is not None and motion < self.alarm_motion)

        # frame_nobg is kept so a consumer can display or log exactly the
        # frame the network saw, without subtracting the background a second
        # time and risking a different result than the one just classified.
        self.last.update({'tag': tag, 'conf': conf, 'E_k': e_k, 'probs': probs,
                          'majority': majority, 'motion': motion,
                          'alarm': alarm, 'frame_nobg': frame_no_bg})

        return self.decode(tag, conf, e_k, probs, majority, motion, alarm)

    def reset(self):
        """Start a new stream: drop the background and both temporal estimators.

        The reference builds its background per recording and constructs a fresh
        MotionEstimator and MajorityTagVoter per situation folder, so this is
        what "a new sequence" means here. Note that it discards the background
        too -- the next bg_frames frames have to show the empty room again.
        """
        self._background = None
        self._bg_buffer = []
        self._motion_buffer = deque(maxlen=self.motion_gap)
        self._motion_smooth_buffer = deque(maxlen=self.motion_smooth)
        self._vote_window = deque(maxlen=self.motion_gap)
        self.last = {}

    def save(self):
        """
        Overrides standard save method, which only stores attributes, but
        is not capable of storing objects which have a save() method of
        their own.

        Returns
        -------
        attr_dict.

        """

        attr_dict = {}

        for attr_name in dir(self):
            # get the attribute *on the class*, not the instance
            attr = getattr(self.__class__, attr_name, None)

            # check if it's a property
            if isinstance(attr, property) and attr_name not in self._SAVE_EXCLUDE:

                # Retrieve the value of the property attr
                attr_val = getattr(self, attr_name)

                # If the property has a save() method, save the return value of
                # that method
                if hasattr(attr_val, "save") and callable(attr_val.save):
                    attr_dict[attr_name] = attr_val.save()
                else:
                    attr_dict[attr_name] = attr_val

        # Append name
        attr_dict['name'] = self.name

        return attr_dict

    # ------------------------------------------------------------------ #
    #  1. Background estimation -- backgroundRemover.py::computeBackground()
    # ------------------------------------------------------------------ #

    def update_background(self, frame: np.ndarray) -> bool:
        """Feed one raw frame into the background estimate.

        The background is the per-pixel mean of the first bg_frames frames,
        which are assumed to show the empty ("dark") room before the subject
        enters -- computed in float for accuracy, then rounded back to the
        frames' native integer deciKelvin so the difference stays exact.

        It is estimated ONCE and never updated afterwards. That is not an
        omission: the model was trained on frames from which a known, constant
        floor temperature had been subtracted (CNNDetection/data/dataset.py), so
        a running estimate would hand it a differently-conditioned input, and a
        person who stops moving would be absorbed into the background and then
        subtracted away.

        Returns
        -------
        bool
            Whether the background is established, i.e. whether this frame can
            be classified. True already on the frame that COMPLETES the estimate
            -- the reference subtracts its background from every frame of the
            recording, including the ones it averaged, so that frame is
            classified rather than dropped.
        """
        if self._background is not None:
            return True

        self._bg_buffer.append(frame.astype(np.float64))

        if len(self._bg_buffer) < self.bg_frames:
            return False

        stack = np.stack(self._bg_buffer, axis=0)
        self._background = np.rint(stack.mean(axis=0)).astype(np.int64)
        self._bg_buffer = []

        logger.info('Background established from %d frames (median %.0f dK).',
                    self.bg_frames, float(np.median(self._background)))

        return True

    # ------------------------------------------------------------------ #
    #  2. Background removal -- backgroundRemover.py::removeBackground()
    # ------------------------------------------------------------------ #

    def remove_background(self, frame: np.ndarray) -> np.ndarray:
        """Raw frame -> NoBG contrast in deciKelvin.

        The exact integer difference ``frame - background``, SIGNED. Cold
        objects and an imperfect subtraction produce genuine negative contrast;
        the model's clip range keeps it, and taking an absolute value here would
        mirror it into the positive half and misrepresent it as warmth.
        """
        if self._background is None:
            raise RuntimeError('No background yet -- call update_background() '
                               'until it returns True.')

        return frame.astype(np.int64) - self._background

    # ------------------------------------------------------------------ #
    #  3. Model input -- renderCNN.py::toModelInput()
    # ------------------------------------------------------------------ #

    def to_model_input(self, frame_no_bg: np.ndarray):
        """NoBG contrast in dK -> the network's (1,1,H,W) float32 input tensor.

        The frozen global affine of the input contract (Section 1.4) and nothing
        else: clip to [lo, hi] dK, divide by scale. Chosen once from pilot
        percentiles and frozen, never per-frame min/max -- a per-frame
        normalisation would discard the calibration that is the thermal sensor's
        one real advantage over RGB, and the network cannot recover it.
        """
        torch = self._torch()

        lo, hi, scale = self.affine
        x = np.clip(frame_no_bg.astype(np.float32), lo, hi) / scale

        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)
                                )[None, None, :, :].to(self._resolve_device())

    # ------------------------------------------------------------------ #
    #  4. Inference -- renderCNN.py::predict()
    # ------------------------------------------------------------------ #

    def predict(self, frame_no_bg: np.ndarray):
        """Classify one NoBG frame.

        Returns
        -------
        tuple
            ``(tag, conf, E[k], probs)`` -- the predicted tag name, its
            probability, the expected criticality and the full probability
            vector. The same four values, in the same order, that the
            reference's predict() returns.
        """
        torch = self._torch()

        if self._net is None:
            raise RuntimeError('No model loaded: construct this engine with a '
                               'model_path pointing at a PostureNet '
                               'checkpoint.')

        x = self.to_model_input(frame_no_bg)

        with torch.no_grad():
            out = self._forward(x)

        logits = out['logits']
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)

        # The output contract to the state engine (Section 1.9):
        # E[k] = sum_k k * p_k. A single monotone scalar on the criticality
        # scale which, unlike argmax, degrades gracefully when the network is
        # uncertain between two adjacent classes.
        k = torch.arange(probs.shape[0], dtype=probs.dtype, device=probs.device)
        e_k = float((probs * k).sum())

        self.last['attn'] = out['attn'][0, 0].cpu().numpy()

        return (self.TAG_NAMES[int(idx)], float(conf), e_k,
                probs.cpu().numpy())

    def _forward(self, x) -> dict:
        """The forward pass of CNNDetection/model/net.py, written out here.

        Walks _TRUNK_SPEC in order -- which is net.trunk()'s order -- then the
        attention+average pooling head of Section 1.6 and the 5-way linear. The
        three auxiliary heads are neither built nor run: they are a training aid
        the reference runs with withAux=False, and net.py deletes them at export.
        """
        torch = self._torch()
        net = self._net

        for kind, mod_name, _, _, _, _ in self._TRUNK_SPEC:
            block = net[mod_name]
            if kind == 'stage':
                # ConvBnRelu: conv -> BN -> ReLU.
                x = torch.relu(block['bn'](block['conv'](x)))
            else:
                # SepResBlock: depthwise -> BN -> ReLU -> pointwise -> BN, then
                # the residual add and a final ReLU.
                out = torch.relu(block['bn1'](block['depthwise'](x)))
                out = block['bn2'](block['pointwise'](out))
                x = torch.relu(out + x)

        feat = x                                         # (1, C, 8, 10)
        b, c, h, w = feat.shape

        v_avg = feat.mean(dim=(2, 3))                    # (1, C)

        if self._cfg['useAttention']:
            scores = net['pool']['score'](feat).view(b, 1, h * w) / self.tau
            alpha = torch.softmax(scores, dim=2)
            flat = feat.view(b, c, h * w)
            v_attn = torch.bmm(flat, alpha.transpose(1, 2)).squeeze(2)
            attn = alpha.view(b, 1, h, w)
        else:
            # Plain-GAP ablation: the attention branch is replaced by the
            # average, and the map is uniform -- head.py concatenates two copies
            # so the classifier's input width is unchanged.
            v_attn = v_avg
            attn = feat.new_full((b, 1, h, w), 1.0 / (h * w))

        # Order matters: attention branch first, average second, as in head.py's
        # torch.cat([vAttn, vAvg]).
        pooled = torch.cat([v_attn, v_avg], dim=1)       # (1, 2C)

        return {'logits': net['classifier'](pooled), 'attn': attn}

    # ------------------------------------------------------------------ #
    #  5. Post-processing -- renderCNN.py::MajorityTagVoter, MotionEstimator
    # ------------------------------------------------------------------ #

    def update_majority(self, probs: np.ndarray) -> str:
        """Majority posture over the last motion_gap frames.

        The argmax of the SUMMED probability vectors over the window, not a
        count of per-frame argmaxes: a run of frames that each lean weakly
        towards lying outvotes one frame that is confidently something else,
        which is the point of summing probabilities. Temporally steadier than
        the per-frame tag, and what the alarm gate reads.
        """
        self._vote_window.append(np.asarray(probs))
        summed = np.sum(self._vote_window, axis=0)
        return self.TAG_NAMES[int(np.argmax(summed))]

    def update_motion(self, frame_no_bg: np.ndarray):
        """Noise-gated frame-difference motion over a frame GAP.

        The sum of |frame_N - frame_{N-GAP}| over only those pixels whose change
        exceeds a few sigma of the DIFFERENCE noise, divided by the current body
        area, then trailing-averaged. Three ideas, all the reference's:

          * the noise gate removes the per-pixel noise floor, which otherwise
            swamps a raw absolute-difference sum. The sigma is FIXED at the
            generator's 2.4 dK and a difference of two frames has a sigma
            sqrt(2) larger, hence the sqrt(2) below;
          * the GAP is load-bearing. Between consecutive frames a walker shifts
            a pixel or two, which looks like a lying person's small stirs; over
            GAP frames the walker's blob translates clear of its old position
            (old vanishes, new appears) and the gated change grows by about an
            order of magnitude while a lying person, still overlapping itself,
            barely rises;
          * normalising by body area cancels person size and distance, because a
            translating blob changes about twice its own area by about its own
            contrast.

        Returns
        -------
        float or None
            The smoothed, normalised motion, or None for the first GAP frames of
            a stream, when there is no frame a GAP back to compare against.
        """
        # Gate the DIFFERENCE at motion_sigma of the difference noise, and the
        # body area at motion_sigma of a single frame's noise.
        diff_threshold = self.motion_sigma * self.noise_sigma_dK * (2 ** 0.5)
        body_threshold = self.motion_sigma * self.noise_sigma_dK

        frame = frame_no_bg.astype(np.float64)

        # Once the buffer is full its oldest entry is exactly GAP frames back.
        baseline = (self._motion_buffer[0]
                    if len(self._motion_buffer) == self.motion_gap else None)
        self._motion_buffer.append(frame)

        if baseline is None:
            return None

        d = np.abs(frame - baseline)
        change = float(d[d > diff_threshold].sum())   # gated: noise floor gone
        body_area = int((frame > body_threshold).sum())
        raw = change / body_area if body_area > 0 else 0.0

        self._motion_smooth_buffer.append(raw)

        return (sum(self._motion_smooth_buffer)
                / len(self._motion_smooth_buffer))

    # ------------------------------------------------------------------ #
    #  6. Decoding to annotations
    # ------------------------------------------------------------------ #

    def decode(self, tag, conf, e_k, probs, majority, motion, alarm):
        """The reference's per-frame result -> one annotation row.

        The box is the WHOLE FRAME, and that is not a placeholder: PostureNet
        classifies the frame, it does not localise. The reference runs it with
        withAux=False, so the auxiliary centroid/axis/area heads that could
        narrow the box down are not evaluated at all, and nothing else in the
        reference produces a box. A row therefore says "this frame shows <tag>",
        which is exactly as much as the network claims.

        ``conf`` is the predicted tag's own probability, as the reference's
        predict() reports it. The full distribution is in class_0..class_4, the
        ordinal in 'E_k', and the two temporal post-processing results in
        'majority' and 'motion', with the gate's verdict in 'alarm'.
        """
        class_id = int(np.argmax(probs))

        if class_id == 0 and not self.emit_no_person:
            return self._empty_annotations()

        height, width = self.input_shape

        row = {'xtl': 0.0, 'ytl': 0.0, 'xbr': float(width), 'ybr': float(height),
               'w': float(width), 'h': float(height),
               'conf': float(conf)}

        # The full posture distribution, under the column names this project's
        # other decoder uses for its class probabilities, so a consumer written
        # against that output finds what it expects.
        for i in range(len(probs)):
            row[f'class_{i}'] = float(probs[i])

        # Placeholders. A YOLO annotation names the grid cell and anchor a box
        # was decoded from; PostureNet has neither, and these columns exist only
        # so that anything indexing an annotation by name keeps working.
        row['cell_y'], row['cell_x'], row['anchor_idx'] = 0, 0, 0

        # The reference's outputs.
        row['class_id'] = class_id
        row['label'] = tag
        row['E_k'] = float(e_k)
        row['majority'] = majority
        row['motion'] = np.nan if motion is None else float(motion)
        row['alarm'] = bool(alarm)

        df = pd.DataFrame([row], index=pd.Index([0], name='id'))

        return df[list(self._columns())]

    def _columns(self):
        """The columns of a decoded annotation, in order.

        The first block is exactly YOLO_EnDecode_Engine's, so that an existing
        consumer of decoded detections -- the plotting processes, the trackers --
        sees what it expects; the posture columns are appended after it.
        """
        return (['xtl', 'ytl', 'xbr', 'ybr', 'w', 'h', 'conf'] +
                [f'class_{i}' for i in range(len(self.TAG_NAMES))] +
                ['cell_y', 'cell_x', 'anchor_idx'] +
                ['class_id', 'label', 'E_k', 'majority', 'motion', 'alarm'])

    def _empty_annotations(self) -> pd.DataFrame:
        """An empty annotation DataFrame carrying the same columns and index as
        a non-empty decode() result.

        pd.DataFrame() comes back without any columns at all, which breaks every
        consumer that indexes one instead of simply iterating over zero rows --
        the same reason YOLO_EnDecode_Engine has this method.
        """
        return pd.DataFrame(columns=self._columns(),
                            index=pd.Index([], name='id'))

    # ------------------------------------------------------------------ #
    #  Frames and model construction
    # ------------------------------------------------------------------ #

    def _as_frame(self, frame: np.ndarray) -> np.ndarray:
        """Bring one incoming array to the (H,W) the checkpoint expects.

        squeeze() first, as the reference's loadFrame() does -- a stray singleton
        axis is common in this project's stored frames. A flat frame of the right
        pixel count is reshaped, since a live frame may arrive as a 1-D buffer;
        anything else fails loudly rather than being interpolated into shape,
        because a geometry mismatch here is nearly always the wrong checkpoint.
        """
        expected = self.input_shape
        frame = np.asarray(frame).squeeze()

        if frame.shape == expected:
            return frame

        if frame.size == expected[0] * expected[1]:
            return frame.reshape(expected)

        raise ValueError(
            f'Frame has shape {frame.shape} ({frame.size} values), but the '
            f'checkpoint {self.model_path} was trained on {expected}. This '
            'engine takes the raw pixel frame in deciKelvin -- not a full '
            'sensor data packet with its PTAT and electrical offsets, and not '
            'the 32x32 of CNNDetection32, whose PostureNet32 has a different '
            'trunk and needs its own engine.')

    def _torch(self):
        """Import torch on first use.

        Imported here rather than at module level on purpose: hsdetect's engine
        registry imports EVERY module of the package to find out which ones
        declare IS_ENGINE_MODULE, so a torch import at module level would make
        torch a hard requirement of `import hsdetect` -- on the live inference
        machines too, which only have tensorflow.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                'PostureNetEngine needs PyTorch, which is not installed in '
                'this environment. It is imported lazily, so the rest of '
                'hsdetect works without it.') from exc
        return torch

    def _resolve_device(self):
        """The torch.device this engine runs on.

        A method and not a property on purpose: save() walks every property of
        the class, and a torch.device has no business in a parameter dict -- nor
        does importing torch in order to build one.
        """
        torch = self._torch()
        if self.device != 'auto':
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _build_net(self) -> None:
        """Load the checkpoint, read its input contract, build the net and load
        the weights into it."""
        torch = self._torch()

        try:
            checkpoint = torch.load(self.model_path, map_location='cpu',
                                    weights_only=True)
        except Exception:
            # torch < 1.13 has no weights_only, and a checkpoint carrying
            # non-tensor metadata (train.py stores the Config and the metrics
            # alongside the weights) needs the full unpickler. This is the
            # project's own checkpoint, not a downloaded one.
            checkpoint = torch.load(self.model_path, map_location='cpu')

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state = checkpoint['model']
            # The training Config travels inside the checkpoint, which makes it
            # authoritative for the affine, the geometry and whether the head
            # uses attention -- rather than a copy of config.py that can drift
            # away from the weights it is supposed to describe.
            self._cfg.update({k: v for k, v in
                              checkpoint.get('config', {}).items()
                              if k in self._DEFAULT_CFG})
        else:
            state = checkpoint

        self._assert_posturenet(state)

        self._net = self._make_modules()
        self._load_weights(state)

    def _assert_posturenet(self, state: dict) -> None:
        """Fail with a useful message when the checkpoint is a different net.

        The sibling CNNDetection32 trains a PostureNet32 on 32x32 frames with a
        re-derived trunk (two stride-2 stages, 64 channels, an 8x8 grid and a
        fourth aux head), whose state_dict loads into nothing here. Its weights
        differ in shape at the very first layer, so this is cheap to detect and
        far clearer than a shape error from inside a convolution.
        """
        required = {mod_name + ('.conv.weight' if kind == 'stage'
                                else '.depthwise.weight')
                    for kind, mod_name, _, _, _, _ in self._TRUNK_SPEC}
        missing = sorted(required - set(state))

        if missing:
            raise ValueError(
                f'{self.model_path} does not look like a PostureNet checkpoint: '
                f'no weights for {missing}. The 32x32 PostureNet32 of '
                'CNNDetection32 has a different trunk and needs its own engine.')

        shape = tuple(state['stage1.conv.weight'].shape)
        expected = (self._TRUNK_SPEC[0][3], self._TRUNK_SPEC[0][2],
                    self._TRUNK_SPEC[0][4], self._TRUNK_SPEC[0][4])
        if shape != expected:
            raise ValueError(
                f'{self.model_path} has stage1.conv.weight of shape {shape}, '
                f'but this engine builds {expected}.')

    def _make_modules(self):
        """Build the network as a ModuleDict whose keys ARE the state_dict
        prefixes.

        A ModuleDict used as the root module produces exactly the key names
        net.py's named attributes produce ('stage1.conv.weight',
        'res2.depthwise.weight', 'pool.score.weight', ...), which is what lets
        the trained weights load into a network that was never defined as
        classes. nn.Sequential would number its children instead and no key
        would match.
        """
        torch = self._torch()
        nn = torch.nn

        modules = {}

        for kind, mod_name, in_c, out_c, kernel, stride in self._TRUNK_SPEC:
            if kind == 'stage':
                # ConvBnRelu: 'same'-style padding for odd kernels keeps the
                # documented output sizes; BN carries the bias, so the
                # convolution has none.
                modules[mod_name] = nn.ModuleDict({
                    'conv': nn.Conv2d(in_c, out_c, kernel, stride=stride,
                                      padding=kernel // 2, bias=False),
                    'bn': nn.BatchNorm2d(out_c)})
            else:
                # SepResBlock: one 3x3 kernel per channel (groups == channels,
                # no mixing), then a 1x1 that mixes channels without spatial
                # extent.
                modules[mod_name] = nn.ModuleDict({
                    'depthwise': nn.Conv2d(in_c, in_c, kernel,
                                           padding=kernel // 2,
                                           groups=in_c, bias=False),
                    'bn1': nn.BatchNorm2d(in_c),
                    'pointwise': nn.Conv2d(in_c, out_c, 1, bias=False),
                    'bn2': nn.BatchNorm2d(out_c)})

        c = self._FEATURE_CHANNELS

        # The attention score conv of the pooling head.
        modules['pool'] = nn.ModuleDict({'score': nn.Conv2d(c, 1, 1)})
        # concat(attention-pooled, average-pooled) is 2C wide.
        modules['classifier'] = nn.Linear(2 * c, len(self.TAG_NAMES))

        net = nn.ModuleDict(modules).to(self._resolve_device())

        # eval() matters: the BN layers must use their stored running
        # statistics, not batch statistics computed over a batch of one.
        net.eval()

        return net

    def _load_weights(self, state: dict) -> None:
        """Load the checkpoint into the built network, and verify it.

        Loaded non-strictly because the checkpoint carries entries this network
        does not have -- the ordinal index buffer `_kIndex`, and the three
        auxiliary heads, which are not built because inference does not run them
        -- but every parameter of the network MUST come from the checkpoint. A
        missing key means a randomly initialized layer and a model that predicts
        plausible nonsense, which is exactly the failure that is hardest to
        notice downstream, so it raises.
        """
        result = self._net.load_state_dict(state, strict=False)

        if result.missing_keys:
            raise ValueError(
                f'{self.model_path} provides no weights for '
                f'{sorted(result.missing_keys)}. Refusing to run a partially '
                'initialized network.')

        ignorable = {'_kIndex'}
        unexpected = [k for k in result.unexpected_keys
                      if k not in ignorable
                      and not k.startswith(('centroidHead', 'axisHead',
                                            'areaHead'))]
        if unexpected:
            logger.warning('%s carries unused entries %s', self.model_path,
                           sorted(unexpected))
