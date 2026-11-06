"""
netinput.py

Single source of truth for the array the detector consumes. Deliberately
numpy-only (no tensorflow, no hspy), so it can be imported from both ends of
the pipeline:

  * TRAINING   -- video_dataset.build_frame_targets(), once per frame while
                  the tf.data.Dataset is being built.
  * INFERENCE  -- cv.preproc_engines (BackgroundSub stacks the channels,
                  MedianSub removes the scene level) and
                  dl_engines.DLEngine.__call__(), which forms the batch.

Both paths MUST produce identical arrays, which is the only reason this lives
in a module of its own rather than being inlined twice.

It sits at the top level of the package, rather than next to its callers in
hsdetect.dl, because hsdetect/dl/__init__.py star-imports the tensorflow-backed
layer and model modules -- importing it from hsdetect.cv.preproc_engines would
otherwise pull tensorflow into the preprocessing import chain.

Two-channel input
-----------------
Channel 0 is the background-subtracted frame |I0 - B|, which sits at ~0..60 dK.
Channel 1 is the frame as it entered background subtraction, i.e. absolute
sensor values at ~3000 dK. Fed in unmodified, that DC level is a problem rather
than information: the backbone convolves with padding="same", so a constant
interior level becomes a step at every image border (the padding contributes 0
where the image contributes ~3000) and at the boundary of ZeroCorners' mask.
Measured on real frames, that step is ~55x the contrast of a person. BatchNorm
does not absorb it, because it absorbs a *constant* offset and the offset stops
being constant exactly at those boundaries.

The fix is to remove the scene's own level, so that 0 means "at the temperature
of the scene" rather than "absolute zero". Zero padding then extends the frame
with ambient-temperature pixels, which is the sensible extrapolation, and the
corners ZeroCorners writes read as ambient as well.

That subtraction is done per frame rather than with a fixed constant because
the scene level is a property of the deployment, not of the dataset: across the
training recordings it was measured at 2999..3022 dK, but a 15 degC room sits
near 2880 dK, and a fixed constant calibrated on the former would leave a
residual DC roughly the size of a whole person in the latter.

Subtracting it is the MedianSub engine's job, so that it happens inside the
engine chain which the training script and live inference both execute -- there
is no second code path to keep in sync, and nothing has to be carried in
metadata. This module only provides the two numpy primitives that engine uses,
plus the stacking and batching helpers.

Estimating the scene level
--------------------------
The median of the frame's valid pixels. Measured over 1543 annotated frames, a
person in the frame moves it by at most 2 dK: a median is unaffected by a tail
covering less than half the pixels, and box coverage stays below 14% in this
setup. (If coverage could exceed ~50% -- a person close enough to fill the
field of view -- the median would tip into the person's own values, and the
median of BackgroundSub's background model, which excludes detections
structurally, would be the robust alternative.)

Two kinds of pixel are excluded from the estimate:

  * exact zeros -- the mask written by ZeroCorners (the sensor's dead corners)
    or by ShrinkCenter (its zero border). Those are not measurements; a real
    reading is never 0 K.
  * everything outside the sensor's detection mask, when the caller passes one.

The two overlap by design, because whether the corners are already zeroed
depends on where the caller sits in the chain: ZeroCorners runs BEFORE
BackgroundSub in the training script and AFTER it live. The corners read
~3030..3055 dK, i.e. ABOVE the scene rather than below, so counting them biases
the estimate by ~0.5..1.1 dK. Passing the detection mask makes the estimate
independent of that ordering.

Preserving the mask
-------------------
subtract_scene_level() maps exact zeros to exact zeros, so the mask keeps
meaning the same thing in the subtracted frame. That is also what makes the
ordering difference above harmless: live, the corners still hold sensor values
when MedianSub runs and are zeroed by ZeroCorners right afterwards; in the
training script they are already zero and are left alone.

The subtraction must happen before anything resamples the frame spatially.
ShrinkCenter's INTER_AREA downscale averages across the boundary of the zeroed
corners; run on absolute dK it produces intermediate values (a half-and-half
pixel gives 0.5 * 3000 = 1500) that are neither data nor exactly zero, and
subtracting the scene level afterwards would turn them into a ring of strongly
negative values around the mask -- many times the full range of channel 0, i.e.
exactly the kind of spurious high-contrast edge this is meant to remove. Run on
already-subtracted values, the same average mixes 0 with values near 0 and
correctly stays near 0. So MedianSub belongs before ShrinkCenter in the chain.

Consequence: from BackgroundSub onwards the pipeline's frame is float32 when it
carries two channels. The purely spatial engines downstream must therefore not
cast it to uint16.
"""

import numpy as np


def estimate_scene_level(frame_raw: np.ndarray,
                         mask: np.ndarray = None) -> float:
    """Estimate a frame's own scene level, in the frame's units (dK).

    Parameters
    ----------
    frame_raw : np.ndarray
        (H, W) or (H, W, 1) frame of absolute sensor values.
    mask : np.ndarray or None
        Optional boolean/0-1 mask of the pixels that carry data, i.e. the
        sensor's detection mask. Pixels outside it are excluded, as are exact
        zeros in either case. See the module docstring for why both.

    Returns
    -------
    float or None
        The median of the valid pixels, or None if there are none -- in which
        case subtract_scene_level() leaves the frame alone rather than shifting
        it by a made-up number.
    """
    raw = np.asarray(frame_raw, dtype=np.float32)

    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]

    valid = raw != 0.0

    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != raw.shape:
            raise ValueError(f'Mask {mask.shape} does not match frame '
                             f'{raw.shape}.')
        valid &= mask.astype(bool)

    if not valid.any():
        return None

    return float(np.median(raw[valid]))


def subtract_scene_level(frame_raw: np.ndarray, offset: float) -> np.ndarray:
    """Subtract a scene level from an absolute-value frame, keeping the mask.

    Pixels that are exactly 0 stay exactly 0: they are the mask written by
    ZeroCorners or ShrinkCenter, not measurements, and after the subtraction 0
    is precisely the value that means "at the scene's temperature" -- which is
    what makes it the right thing for zero padding to extend the frame with.

    An `offset` of None returns the frame unchanged (as float32), so a frame
    whose scene level could not be estimated is passed through rather than
    corrupted.
    """
    raw = np.asarray(frame_raw, dtype=np.float32)

    if offset is None:
        return raw

    return np.where(raw == 0.0,
                    np.float32(0.0),
                    raw - np.float32(offset)).astype(np.float32)


def stack_channels(frame_bgsub: np.ndarray,
                   frame_raw: np.ndarray = None) -> np.ndarray:
    """Stack the background-subtracted frame (channel 0) and the frame that
    entered background subtraction (channel 1) into one float32 (H, W, 2)
    array.

    Stacking only -- removing channel 1's scene level is MedianSub's job, and
    happens on whichever representation that engine is handed (channel 1 of
    this array in the live chain, AnnotatedFrame.image2 in the training one).

    Returns the background-subtracted frame unchanged -- same array, same
    dtype, including None -- if `frame_raw` is None, so a pipeline or a stored
    AnnotatedVideo that carries no raw channel keeps producing single-channel
    input exactly as before.
    """
    if frame_raw is None or frame_bgsub is None:
        return frame_bgsub

    bgsub = np.asarray(frame_bgsub)
    raw = np.asarray(frame_raw)

    # Tolerate either being stored with a trailing singleton channel axis.
    if bgsub.ndim == 3 and bgsub.shape[-1] == 1:
        bgsub = bgsub[..., 0]
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]

    if bgsub.shape != raw.shape:
        raise ValueError(f'Background-subtracted frame {bgsub.shape} and raw '
                         f'frame {raw.shape} have different shapes.')

    return np.stack([bgsub.astype(np.float32),
                     raw.astype(np.float32)], axis=-1)


def make_net_input(frame: np.ndarray) -> np.ndarray:
    """Convert a preprocessed frame into the array the model is called with.

    Parameters
    ----------
    frame : np.ndarray
        (H, W) or (H, W, 1) -- background-subtracted frame only; returned as
        float32 (H, W, 1), so single-channel models behave exactly as before
        this module existed.
        (H, W, 2) -- as produced by stack_channels() and corrected by
        MedianSub. Returned as float32, unchanged otherwise.

    Returns
    -------
    np.ndarray, float32, (H, W, C)
        No batch axis -- the caller adds one.
    """
    x = np.asarray(frame, dtype=np.float32)

    if x.ndim == 2:
        x = x[..., np.newaxis]
    elif x.ndim != 3:
        raise ValueError(f'Expected a (H,W), (H,W,1) or (H,W,2) frame, '
                         f'got shape {x.shape}.')

    return x
