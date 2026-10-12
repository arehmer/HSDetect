"""Optical flow estimation for grayscale video sequences.

This module provides a small, extensible framework for computing dense optical
flow on video sequences represented as lists of ``numpy`` arrays.

The framework is organised around the abstract base class
:class:`OpticalFlowEstimator`, which defines a single entry point,
:meth:`OpticalFlowEstimator.process`. Concrete estimators implement a specific
algorithm. Currently provided:

* :class:`FarnebackOpticalFlow` -- dense flow via ``cv2.calcOpticalFlowFarneback``.
* :class:`LucasKanadeOpticalFlow` -- pyramidal Lucas-Kanade via
  ``cv2.calcOpticalFlowPyrLK``, evaluated on a regular grid and assembled into
  a dense field.

Input convention
----------------
All estimators in this module expect a *list of single-channel ``np.uint16``
frames* of identical shape ``(H, W)``. This mirrors the output of many
scientific and industrial cameras (12-bit or 16-bit sensors stored in 16-bit
containers).

Dependencies
------------
* ``numpy``
* ``opencv-python``

Install with::

    pip install opencv-python numpy
"""

from __future__ import annotations

import abc
from typing import List, Sequence

import cv2
import numpy as np

import copy 

from hsdetect.annot.annot import AnnotatedFrame,AnnotatedVideo 

__all__ = [
    "OpticalFlowEstimator",
    "FarnebackOpticalFlow",
    "LucasKanadeOpticalFlow",
]


class OpticalFlowEstimator(abc.ABC):
    """Abstract base class for dense optical flow estimators.

    Subclasses implement :meth:`process`, which consumes an ordered sequence of
    video frames and returns one dense flow field per consecutive frame pair.

    The class also provides shared input validation (:meth:`_validate_frames`)
    and the ``uint16`` to ``uint8`` conversion helper
    (:meth:`_to_uint8_global`) used by OpenCV-backed subclasses.
    """

    @abc.abstractmethod
    def process(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Compute dense optical flow for a video sequence.

        Parameters
        ----------
        frames:
            Ordered list of ``N`` video frames. Each frame must be a
            single-channel (2-D) ``np.uint16`` array, and all frames must share
            the same shape ``(H, W)``.

        Returns
        -------
        list of np.ndarray
            ``N - 1`` flow fields, one per consecutive frame pair
            ``(frames[i], frames[i + 1])``. Each field has shape ``(H, W, 2)``
            and dtype ``float32``, where channel 0 is the horizontal (x)
            displacement and channel 1 the vertical (y) displacement, in
            pixels. An empty list is returned for an empty or single-frame
            input.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_frames(frames: Sequence[np.ndarray]) -> None:
        """Validate that ``frames`` matches the expected input convention.

        Checks, in order, that the input is a sequence, that every element is a
        ``np.ndarray``, that every frame is 2-D (single-channel), that every
        frame has dtype ``np.uint16``, and that all frames share one shape.

        Parameters
        ----------
        frames:
            Candidate sequence of video frames.

        Raises
        ------
        TypeError
            If ``frames`` is not a sequence, if an element is not a
            ``np.ndarray``, or if an element does not have dtype ``np.uint16``.
        ValueError
            If a frame is not 2-D, or if frame shapes are inconsistent.
        """
        if isinstance(frames, np.ndarray) or not isinstance(frames, Sequence):
            raise TypeError(
                "frames must be a list/sequence of np.ndarray, got "
                f"{type(frames).__name__}"
            )

        if len(frames) == 0:
            return

        reference_shape = None
        for index, frame in enumerate(frames):
            if not isinstance(frame, np.ndarray):
                raise TypeError(
                    f"frames[{index}] must be a np.ndarray, got "
                    f"{type(frame).__name__}"
                )
            if frame.ndim != 2:
                raise ValueError(
                    f"frames[{index}] must be 2-D (single-channel grayscale), "
                    f"got ndim={frame.ndim} with shape {frame.shape}"
                )
            if frame.dtype != np.uint16:
                raise TypeError(
                    f"frames[{index}] must have dtype uint16, got {frame.dtype}"
                )
            if reference_shape is None:
                reference_shape = frame.shape
            elif frame.shape != reference_shape:
                raise ValueError(
                    "all frames must share the same shape; frames[0] has "
                    f"{reference_shape} but frames[{index}] has {frame.shape}"
                )

    def _fit_MinMaxScaler(self,
                          avideo : AnnotatedVideo):
        """
        Loop through whole video and get min and max pixel intensity to fit
        sklearn MinMaxScaler

        Parameters
        ----------
        avideo : AnnotatedVideo
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        pix_min = np.inf; pix_max = 0
        
        for frame in avideo:
            img = frame.image
            
            # Get min and max intensity of this frame
            frame_min = img.min((0,1))
            frame_max = img.max((0,1))
            
            # Check if new global min or max have been found and assign
            if frame_min < pix_min:
                pix_min = frame_min
            if frame_max > pix_max:
                pix_max = frame_max
                
 
    @staticmethod
    def _to_uint8_global(frames: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Convert ``uint16`` frames to ``uint8`` using per-video normalization.

        OpenCV's Farneback implementation (like most classical flow
        implementations) only accepts 8-bit single-channel input, so 16-bit
        frames must be down-converted first. This is a **lossy,
        assumption-laden step**:

        * The global minimum and maximum are computed **once over the entire
          sequence**, not per frame. A single affine mapping is then applied to
          every frame::

              scaled = (frame - global_min) / (global_max - global_min) * 255

          Using per-video rather than per-frame statistics preserves relative
          brightness across the sequence. Per-frame normalization would rescale
          each frame independently and thereby inject artificial intensity
          changes, which a brightness-constancy-based method such as Farneback
          would misinterpret as motion.
        * Up to 8 bits of intensity resolution are discarded. Low-contrast
          sequences, or sequences containing a few extreme outlier pixels (hot
          pixels, saturated specular highlights), compress the useful dynamic
          range into a handful of 8-bit codes and can degrade flow quality.
          Clip or otherwise pre-condition such data before calling this method
          if that matters.
        * The mapping is linear; no gamma correction or histogram equalization
          is applied.

        Degenerate case: if ``global_max == global_min`` the sequence has
        constant intensity and the scaling is undefined (division by zero). An
        all-zero ``uint8`` sequence is returned in that case, which yields zero
        flow -- the correct answer for a constant-intensity sequence.

        Parameters
        ----------
        frames:
            Validated sequence of 2-D ``np.uint16`` frames.

        Returns
        -------
        list of np.ndarray
            Frames of the same shapes with dtype ``np.uint8``, in the same
            order.
        """
        if len(frames) == 0:
            return []

        global_min = min(int(frame.min()) for frame in frames)
        global_max = max(int(frame.max()) for frame in frames)

        if global_max == global_min:
            # Constant-intensity sequence: the affine mapping is undefined.
            return [np.zeros(frame.shape, dtype=np.uint8) for frame in frames]

        scale = 255.0 / float(global_max - global_min)
        converted: List[np.ndarray] = []
        for frame in frames:
            scaled = (frame.astype(np.float32) - global_min) * scale
            # Round, then clip to guard against float round-off at the ends.
            converted.append(np.clip(np.rint(scaled), 0, 255).astype(np.uint8))
        return converted


    @classmethod
    def _prepare_uint8(cls, frames: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Validate a sequence and convert it to OpenCV-compatible ``uint8``.

        Convenience wrapper combining :meth:`_validate_frames` and
        :meth:`_to_uint8_global`, which is the standard preamble for any
        OpenCV-backed :meth:`process` implementation.

        Parameters
        ----------
        frames:
            Candidate sequence of video frames.

        Returns
        -------
        list of np.ndarray
            The converted ``uint8`` frames, or an empty list when fewer than
            two frames were supplied (no frame pair exists, so no conversion is
            needed). Validation runs in either case.

        Raises
        ------
        TypeError, ValueError
            Propagated from :meth:`_validate_frames`.
        """
        cls._validate_frames(frames)
        if len(frames) < 2:
            # Nothing to pair up: no flow field is defined.
            return []
        return cls._to_uint8_global(frames)


class FarnebackOpticalFlow(OpticalFlowEstimator):
    """Dense optical flow via Gunnar Farneback's polynomial expansion algorithm.

    A validating wrapper around ``cv2.calcOpticalFlowFarneback``. All tuning
    parameters of the OpenCV function are exposed as constructor arguments
    (with OpenCV's conventional defaults) and stored as instance attributes, so
    a configured estimator can be reused across sequences.

    Because Farneback requires 8-bit input, ``uint16`` frames are converted with
    per-video (global) min/max normalization; see
    :meth:`OpticalFlowEstimator._to_uint8_global` for the assumptions and the
    information loss involved.

    Attributes
    ----------
    pyr_scale : float
        Image scale (< 1) between successive pyramid levels. ``0.5`` builds a
        classic pyramid where each level is half the size of the previous one.
    levels : int
        Number of pyramid levels, including the original image. ``1`` means no
        pyramid is used.
    winsize : int
        Averaging window size. Larger values increase robustness to noise and
        allow larger displacements, at the cost of a blurrier flow field.
    iterations : int
        Number of iterations performed at each pyramid level.
    poly_n : int
        Size of the pixel neighbourhood used for the polynomial expansion.
        Typically 5 or 7.
    poly_sigma : float
        Standard deviation of the Gaussian used to smooth derivatives for the
        polynomial expansion. Roughly 1.1 for ``poly_n=5``, 1.5 for
        ``poly_n=7``.
    flags : int
        OpenCV operation flags, e.g. ``cv2.OPTFLOW_FARNEBACK_GAUSSIAN``.
    """

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        flags: int = 0,
    ) -> None:
        """Create a configured Farneback estimator.

        Parameters
        ----------
        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags:
            Passed through verbatim to ``cv2.calcOpticalFlowFarneback``. See the
            class attribute documentation for their meaning; the defaults are
            OpenCV's conventional ones.

        Raises
        ------
        ValueError
            If ``pyr_scale`` is not in ``(0, 1)``, or if ``levels``,
            ``winsize``, ``iterations`` or ``poly_n`` is not positive.
        """
        if levels != 1 and not 0.0 < pyr_scale < 1.0:
            raise ValueError(f"pyr_scale must be in (0, 1), got {pyr_scale}")
        if levels < 1:
            raise ValueError(f"levels must be >1= 1, got {levels}")
        if winsize < 1:
            raise ValueError(f"winsize must be >= 1, got {winsize}")
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        if poly_n < 1:
            raise ValueError(f"poly_n must be >= 1, got {poly_n}")

        self.pyr_scale: float = float(pyr_scale)
        self.levels: int = int(levels)
        self.winsize: int = int(winsize)
        self.iterations: int = int(iterations)
        self.poly_n: int = int(poly_n)
        self.poly_sigma: float = float(poly_sigma)
        self.flags: int = int(flags)

    def process(self,
                avideo: AnnotatedVideo,
                coords = 'xy') -> List[np.ndarray]:
        """Compute Farneback optical flow for a ``uint16`` video sequence.

        The sequence is validated, converted to ``uint8`` with per-video
        (global) min/max normalization -- a lossy step documented in
        :meth:`OpticalFlowEstimator._to_uint8_global` -- and then passed
        pairwise to ``cv2.calcOpticalFlowFarneback``.

        Parameters
        ----------
        frames:
            Ordered list of ``N`` single-channel ``np.uint16`` frames, all of
            shape ``(H, W)``.
        
        coords : str
            'xy' to return xy components of optical flow, or 'polar' to return
            magnitude and phase.
        Returns
        -------
        list of np.ndarray
            ``N - 1`` flow fields of shape ``(H, W, 2)`` and dtype ``float32``.
            ``flow[y, x, 0]`` and ``flow[y, x, 1]`` give the horizontal and
            vertical displacement in pixels that maps the pixel at ``(x, y)`` in
            ``frames[i]`` to its position in ``frames[i + 1]``. Empty and
            single-frame inputs both yield ``[]``.

        Raises
        ------
        TypeError
            If ``frames`` is not a sequence of ``np.ndarray``, or a frame does
            not have dtype ``uint16``.
        ValueError
            If a frame is not 2-D, or frame shapes are inconsistent.
        """
        
        # Copy original video (to maintain metadata)
        avideo_flow = copy.deepcopy(avideo)
        
        # Get image dimensions from first frame
        h,w = avideo[0].image.shape
        
        # Fit a global scaler to scale images to np.uint8 range
        scaler = GlobalMinMaxScaler(feature_range = (0,255),
                                    clip = True)
        
        frame_list = [frame.image for frame in avideo_flow]
        
        scaler.fit(frame_list)
        
        flow_frames_x = []
        flow_frames_y = [] 
        
        for k in range(len(avideo)):
            
            # Exception handling fro first frame
            if k == 0:
                flow_frame = copy.deepcopy(avideo_flow[k])
                flow_frame.image = np.zeros((h,w)).astype(np.float32) 
                continue
            
            # Get previous and current frame
            frame_prev = avideo_flow[k-1].image
            frame_curr = avideo_flow[k].image
            
            # Scale to 0,...,255 and convert to np.uint8
            frame_prev_uint8 = scaler.transform(frame_prev).astype(np.uint8)
            frame_curr_uint8 = scaler.transform(frame_curr).astype(np.uint8)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                frame_prev_uint8,
                frame_curr_uint8,
                None,
                self.pyr_scale,
                self.levels,
                self.winsize,
                self.iterations,
                self.poly_n,
                self.poly_sigma,
                self.flags,
            )
            
            # Write result to annotated frame
            flow_frame_x = copy.deepcopy(avideo_flow[k])
            flow_frame_y = copy.deepcopy(avideo_flow[k])
            
            flow_frame_x.image = flow[:,:,0]
            flow_frame_y.image = flow[:,:,1]
            
            # Append to list storing results
            flow_frames_x.append(flow_frame_x)
            flow_frames_y.append(flow_frame_y)
        
        if coords == 'xy':
            return AnnotatedVideo(frames = flow_frames_x),AnnotatedVideo(frames = flow_frames_y)
        
        if coords == 'polar':
            
            # Convert to magnitude and phase
            flow_frames_mag = copy.deepcopy(flow_frames_x)
            flow_frames_phase = copy.deepcopy(flow_frames_x)
            
            for kx,ky in zip(range(len(flow_frames_x)),
                             range(len(flow_frames_y))):
                
                mag, phase = displacement_to_magnitude_phase(flow_frames_x[kx].image,
                                                             flow_frames_y[ky].image)
    
                flow_frames_mag[kx].image = mag
                flow_frames_phase[kx].image = phase
                        
            return AnnotatedVideo(frames = flow_frames_mag), AnnotatedVideo(frames = flow_frames_phase)
                


class LucasKanadeOpticalFlow(OpticalFlowEstimator):
    """Optical flow via pyramidal Lucas-Kanade (``cv2.calcOpticalFlowPyrLK``).

    Unlike Farneback, OpenCV's Lucas-Kanade implementation is a **sparse**
    tracker: it follows a caller-supplied set of points from one frame into the
    next, rather than producing a flow value for every pixel. To satisfy the
    dense ``(H, W, 2)`` output contract of :class:`OpticalFlowEstimator`, this
    class tracks a *regular grid* of points covering the frame and assembles
    the results into a dense field. Two consequences follow, and both are
    important when comparing results against :class:`FarnebackOpticalFlow`:

    * **Grid density is a cost/quality trade-off.** With ``grid_step=1`` every
      pixel is tracked, which is genuinely dense but expensive -- one
      independent windowed least-squares solve per pixel. Larger steps track a
      subsampled grid and bilinearly upsample the result to full resolution,
      which is far faster but spatially smoother.
    * **Tracking can fail per point.** Lucas-Kanade reports a status flag for
      each point, and rejects those in untextured regions (via
      ``min_eig_threshold``) or that leave the frame. Failed points cannot be
      assigned a meaningful displacement, so they are filled with
      ``fill_value`` (``0.0`` by default). A zero there means *"no estimate"*,
      not *"no motion"* -- use :meth:`process_with_status` when the caller
      needs to tell those two cases apart.

    The ``uint16`` input is converted to ``uint8`` with the same lossy
    per-video normalization used by every estimator here; see
    :meth:`OpticalFlowEstimator._to_uint8_global`.

    Attributes
    ----------
    win_size : tuple of int
        Search window size at each pyramid level. Larger windows tolerate
        larger displacements and noise but blur motion boundaries.
    max_level : int
        Maximum pyramid level index. ``0`` uses the original image only;
        ``max_level=3`` uses four levels.
    max_count : int
        Termination criterion: maximum iterations per point per level.
    epsilon : float
        Termination criterion: minimum search-window displacement below which
        iteration stops.
    criteria : tuple
        The assembled OpenCV ``TermCriteria`` tuple built from ``max_count``
        and ``epsilon``, passed to ``cv2.calcOpticalFlowPyrLK``.
    flags : int
        OpenCV operation flags, e.g. ``cv2.OPTFLOW_LK_GET_MIN_EIGENVALS``.
    min_eig_threshold : float
        Minimum eigenvalue of the spatial gradient matrix. Points below this
        are treated as untrackable (typically flat, untextured regions) and
        their status is reported as failed.
    grid_step : int
        Spacing, in pixels, of the tracked point grid. ``1`` tracks every
        pixel; larger values subsample and then upsample the flow field.
    fill_value : float
        Displacement written for points whose tracking failed.
    """

    def __init__(
        self,
        win_size: tuple[int, int] = (21, 21),
        max_level: int = 3,
        max_count: int = 30,
        epsilon: float = 0.01,
        flags: int = 0,
        min_eig_threshold: float = 1e-4,
        grid_step: int = 1,
        fill_value: float = 0.0,
    ) -> None:
        """Create a configured Lucas-Kanade estimator.

        Parameters
        ----------
        win_size, max_level, flags, min_eig_threshold:
            Passed through verbatim to ``cv2.calcOpticalFlowPyrLK``, with
            OpenCV's conventional defaults.
        max_count, epsilon:
            Combined into the ``TermCriteria`` tuple that OpenCV expects,
            exposed separately so callers need not build the tuple themselves.
        grid_step:
            Spacing of the tracked point grid in pixels (``1`` = every pixel).
        fill_value:
            Displacement assigned where tracking failed.

        Raises
        ------
        TypeError
            If ``win_size`` is not a pair of integers.
        ValueError
            If ``win_size`` entries are not positive, ``max_level`` is
            negative, ``max_count`` or ``grid_step`` is not positive, or
            ``epsilon`` or ``min_eig_threshold`` is negative.
        """
        if not isinstance(win_size, (tuple, list)) or len(win_size) != 2:
            raise TypeError(
                f"win_size must be a (width, height) pair, got {win_size!r}"
            )
        if not all(isinstance(value, int) for value in win_size):
            raise TypeError(f"win_size entries must be int, got {win_size!r}")
        if not all(value > 0 for value in win_size):
            raise ValueError(f"win_size entries must be positive, got {win_size!r}")
        if max_level < 0:
            raise ValueError(f"max_level must be >= 0, got {max_level}")
        if max_count < 1:
            raise ValueError(f"max_count must be >= 1, got {max_count}")
        if epsilon < 0.0:
            raise ValueError(f"epsilon must be >= 0, got {epsilon}")
        if min_eig_threshold < 0.0:
            raise ValueError(
                f"min_eig_threshold must be >= 0, got {min_eig_threshold}"
            )
        if grid_step < 1:
            raise ValueError(f"grid_step must be >= 1, got {grid_step}")

        self.win_size: tuple[int, int] = (int(win_size[0]), int(win_size[1]))
        self.max_level: int = int(max_level)
        self.max_count: int = int(max_count)
        self.epsilon: float = float(epsilon)
        self.flags: int = int(flags)
        self.min_eig_threshold: float = float(min_eig_threshold)
        self.grid_step: int = int(grid_step)
        self.fill_value: float = float(fill_value)
        self.criteria: tuple[int, int, float] = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.max_count,
            self.epsilon,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _grid_points(self, shape: tuple[int, int]) -> np.ndarray:
        """Build the regular grid of points to track for a frame of ``shape``.

        Parameters
        ----------
        shape:
            Frame shape ``(H, W)``.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_rows * n_cols, 1, 2)`` and dtype ``float32``,
            in the ``(x, y)`` order OpenCV expects, ordered row-major so it can
            be reshaped straight back into a ``(n_rows, n_cols, ...)`` grid.
        """
        height, width = shape
        ys = np.arange(0, height, self.grid_step, dtype=np.float32)
        xs = np.arange(0, width, self.grid_step, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        points = np.stack((grid_x, grid_y), axis=-1)
        return points.reshape(-1, 1, 2).astype(np.float32)

    def _flow_for_pair(
        self, previous: np.ndarray, current: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track the grid between one frame pair and assemble a dense field.

        Parameters
        ----------
        previous, current:
            Consecutive ``uint8`` frames of identical shape ``(H, W)``.

        Returns
        -------
        flow : np.ndarray
            Dense flow of shape ``(H, W, 2)``, dtype ``float32``.
        valid : np.ndarray
            Boolean mask of shape ``(H, W)``; ``True`` where the flow came from
            a successfully tracked point, ``False`` where ``fill_value`` was
            substituted. When ``grid_step > 1`` both arrays are upsampled from
            the grid, so ``valid`` marks pixels interpolated purely from
            successful neighbours.
        """
        height, width = previous.shape
        points = self._grid_points(previous.shape)

        tracked, status, _error = cv2.calcOpticalFlowPyrLK(
            previous,
            current,
            points,
            None,
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=self.criteria,
            flags=self.flags,
            minEigThreshold=self.min_eig_threshold,
        )

        n_rows = len(range(0, height, self.grid_step))
        n_cols = len(range(0, width, self.grid_step))

        displacement = (tracked - points).reshape(n_rows, n_cols, 2)
        ok = status.reshape(n_rows, n_cols).astype(bool)

        # A failed point carries no usable displacement, so overwrite whatever
        # OpenCV left in `tracked` for it rather than propagating garbage.
        grid_flow = np.where(ok[..., None], displacement, self.fill_value)
        grid_flow = grid_flow.astype(np.float32)

        if self.grid_step == 1:
            return grid_flow, ok

        # Upsample to full resolution. Displacements are in pixels of the
        # original image and are independent of the grid spacing, so the
        # vectors are interpolated but never rescaled.
        flow = cv2.resize(
            grid_flow, (width, height), interpolation=cv2.INTER_LINEAR
        )
        # Interpolating the mask and requiring 1.0 keeps a pixel valid only if
        # every grid neighbour it drew from was itself successfully tracked.
        valid_upsampled = cv2.resize(
            ok.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
        )
        valid = valid_upsampled >= 1.0
        flow = np.where(valid[..., None], flow, self.fill_value)
        return np.ascontiguousarray(flow, dtype=np.float32), valid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, 
                avideo: AnnotatedVideo,
                coords = 'xy') -> List[np.ndarray]:
        """Compute Lucas-Kanade optical flow for a ``uint16`` video sequence.

        The sequence is validated, converted to ``uint8`` with per-video
        (global) min/max normalization -- a lossy step documented in
        :meth:`OpticalFlowEstimator._to_uint8_global` -- and each consecutive
        frame pair is then tracked on a regular grid with
        ``cv2.calcOpticalFlowPyrLK``.

        Pixels whose tracking failed are set to ``fill_value``; with the
        default of ``0.0`` they are indistinguishable from genuinely static
        pixels. Use :meth:`process_with_status` if that distinction matters.

        Parameters
        ----------
        frames:
            Ordered list of ``N`` single-channel ``np.uint16`` frames, all of
            shape ``(H, W)``.

        Returns
        -------
        list of np.ndarray
            ``N - 1`` flow fields of shape ``(H, W, 2)`` and dtype ``float32``,
            with the same channel and sign convention as
            :meth:`FarnebackOpticalFlow.process`. Empty and single-frame inputs
            both yield ``[]``.

        Raises
        ------
        TypeError
            If ``frames`` is not a sequence of ``np.ndarray``, or a frame does
            not have dtype ``uint16``.
        ValueError
            If a frame is not 2-D, or frame shapes are inconsistent.
        """
        # Copy original video (to maintain metadata)
        avideo_flow = copy.deepcopy(avideo)
        
        # Get image dimensions from first frame
        h,w = avideo[0].image.shape
        
        # Fit a global scaler to scale images to np.uint8 range
        scaler = GlobalMinMaxScaler(feature_range = (0,255),
                                    clip = True)
        
        frames = [frame.image for frame in avideo_flow]
        
        scaler.fit(frames)
        
        frames_scaled = [scaler.transform(frame).astype(np.uint8) for frame in frames]
        
        # Calculate optical flow        
        flow_res = [flow for flow, _valid in self.process_with_status(frames)]
        # flow_valid = [_valid for flow, _valid in self.process_with_status(frames)]
        
        # Write optical flow to AnnotatedVideo
        flow_frames_x = []
        flow_frames_y = []
        
        # Copy original video (to maintain metadata)
        avideo_flow = copy.deepcopy(avideo)
        
        for k in range(len(avideo)-1):
            
            # Exception handling fro first frame
            if k == 0:
                flow_frame = copy.deepcopy(avideo_flow[k])
                flow_frame.image = np.zeros((h,w)).astype(np.float32) 
                continue
            
            # Write result to annotated frame
            flow_frame_x = copy.deepcopy(avideo_flow[k])
            flow_frame_y = copy.deepcopy(avideo_flow[k])
        
            flow_frame_x.image = flow_res[k][:,:,0]
            flow_frame_y.image = flow_res[k][:,:,1]
            
            # Append to list storing results
            flow_frames_x.append(flow_frame_x)
            flow_frames_y.append(flow_frame_y)
            
        if coords == 'xy':
            return AnnotatedVideo(frames = flow_frames_x),AnnotatedVideo(frames = flow_frames_y)
        
        if coords == 'polar':
            
            # Convert to magnitude and phase
            flow_frames_mag = copy.deepcopy(flow_frames_x)
            flow_frames_phase = copy.deepcopy(flow_frames_x)
            
            for kx,ky in zip(range(len(flow_frames_x)),
                             range(len(flow_frames_y))):
                
                mag, phase = displacement_to_magnitude_phase(flow_frames_x[kx].image,
                                                             flow_frames_y[ky].image)
    
                flow_frames_mag[kx].image = mag
                flow_frames_phase[kx].image = phase
                        
            return AnnotatedVideo(frames = flow_frames_mag), AnnotatedVideo(frames = flow_frames_phase)
        
        return 

    def process_with_status(
        self, frames: List[np.ndarray]
    ) -> List[tuple[np.ndarray, np.ndarray]]:
        """Compute flow, returning per-pixel tracking validity alongside it.

        Identical to :meth:`process` but preserves the information Lucas-Kanade
        provides and the dense output contract cannot express: which pixels
        were actually tracked. This distinguishes "no motion here" from "no
        estimate here", which matters because untextured regions -- exactly
        where Lucas-Kanade's aperture problem bites -- are silently filled with
        ``fill_value`` by :meth:`process`.

        Parameters
        ----------
        frames:
            Ordered list of ``N`` single-channel ``np.uint16`` frames, all of
            shape ``(H, W)``.

        Returns
        -------
        list of tuple
            ``N - 1`` pairs ``(flow, valid)``, where ``flow`` has shape
            ``(H, W, 2)`` and dtype ``float32``, and ``valid`` is a boolean
            array of shape ``(H, W)`` that is ``True`` where the flow value
            came from a successfully tracked point.

        Raises
        ------
        TypeError, ValueError
            As for :meth:`process`.
        """
        frames_u8 = self._prepare_uint8(frames)
        if not frames_u8:
            return []

        return [
            self._flow_for_pair(previous, current)
            for previous, current in zip(frames_u8[:-1], frames_u8[1:])
        ]

    def __repr__(self) -> str:
        """Return an unambiguous representation including all parameters."""
        return (
            f"{type(self).__name__}(win_size={self.win_size}, "
            f"max_level={self.max_level}, max_count={self.max_count}, "
            f"epsilon={self.epsilon}, flags={self.flags}, "
            f"min_eig_threshold={self.min_eig_threshold}, "
            f"grid_step={self.grid_step}, fill_value={self.fill_value})"
        )


def displacement_to_magnitude_phase(dx, dy, degrees=False, wrap_positive=False):
    """Convert Cartesian displacement components to magnitude and direction.

    Parameters
    ----------
    dx, dy : array_like
        2D arrays of the x and y components of the displacement field.
        Must have the same shape.
    degrees : bool, default False
        Return the angle in degrees instead of radians.
    wrap_positive : bool, default False
        If True, map the angle to [0, 2*pi) (or [0, 360)) instead of the
        default (-pi, pi] returned by arctan2.

    Returns
    -------
    magnitude : ndarray
        Euclidean length of the displacement vector at each pixel, >= 0.
    phase : ndarray
        Angle of the vector measured counter-clockwise from the +x axis.
    """
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    if dx.shape != dy.shape:
        raise ValueError(f"shape mismatch: dx {dx.shape} vs dy {dy.shape}")

    magnitude = np.hypot(dx, dy)          # sqrt(dx**2 + dy**2), overflow-safe
    phase = np.arctan2(dy, dx)            # note the argument order: y first

    if wrap_positive:
        phase = np.mod(phase, 2.0 * np.pi)
    if degrees:
        phase = np.degrees(phase)

    return magnitude, phase

class GlobalMinMaxScaler:
    """Scales frames to [0, 1] using the min/max over all frames seen in fit()."""

    def __init__(self, feature_range=(0.0, 1.0), clip=False):
        self.lo, self.hi = feature_range
        self.clip = clip
        self.min_ = None
        self.max_ = None

    def fit(self, frames):
        self.min_ = min(np.nanmin(f) for f in frames)
        self.max_ = max(np.nanmax(f) for f in frames)
        return self

    def transform(self, frame):
        frame = np.asarray(frame, dtype=float)
        span = self.max_ - self.min_
        if span == 0:                       # constant data
            out = np.zeros_like(frame) + self.lo
        else:
            out = (frame - self.min_) / span * (self.hi - self.lo) + self.lo
        return np.clip(out, self.lo, self.hi) if self.clip else out

    def fit_transform(self, frames):
        return [self.transform(f) for f in self.fit(frames)]

    def inverse_transform(self, frame):
        frame = np.asarray(frame, dtype=float)
        return (frame - self.lo) / (self.hi - self.lo) * (self.max_ - self.min_) + self.min_

    def save(self, path):
        np.savez(path, min_=self.min_, max_=self.max_,
                 lo=self.lo, hi=self.hi, clip=self.clip)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        s = cls(feature_range=(float(d["lo"]), float(d["hi"])), clip=bool(d["clip"]))
        s.min_, s.max_ = float(d["min_"]), float(d["max_"])
        return s



def _make_synthetic_sequence(
    n_frames: int = 5,
    height: int = 128,
    width: int = 160,
    shift_per_frame: int = 3,
) -> List[np.ndarray]:
    """Build a synthetic ``uint16`` sequence of a block translating rightwards.

    Parameters
    ----------
    n_frames:
        Number of frames to generate.
    height, width:
        Frame dimensions in pixels.
    shift_per_frame:
        Horizontal displacement, in pixels, applied between consecutive frames.

    Returns
    -------
    list of np.ndarray
        ``n_frames`` single-channel ``np.uint16`` frames of shape
        ``(height, width)``.
    """
    frames: List[np.ndarray] = []
    for i in range(n_frames):
        frame = np.full((height, width), 1000, dtype=np.uint16)
        x0 = 20 + i * shift_per_frame
        frame[40:90, x0:x0 + 40] = 45000
        frames.append(frame)
    return frames



if __name__ == "__main__":
    SHIFT = 3
    sequence = _make_synthetic_sequence(n_frames=5, shift_per_frame=SHIFT)

    print(f"input frames : {len(sequence)} x {sequence[0].shape} {sequence[0].dtype}")
    print(f"true motion  : dx={SHIFT:+d}, dy=+0 per frame pair")

    # Both estimators satisfy the same interface, so they are interchangeable.
    estimators: List[OpticalFlowEstimator] = [
        FarnebackOpticalFlow(winsize=25, levels=4, poly_n=7, poly_sigma=1.5),
        LucasKanadeOpticalFlow(win_size=(21, 21), max_level=3, grid_step=2),
    ]

    for estimator in estimators:
        print(f"\n{estimator}")
        flow_fields = estimator.process(sequence)
        print(f"  flow fields: {len(flow_fields)}")

        for i, flow in enumerate(flow_fields):
            # Sample a band straddling the block's leading (right) edge. The
            # block interior is untextured, so both methods recover little
            # motion there -- the aperture problem -- and the estimate is only
            # meaningful where there is an intensity gradient, i.e. near edges.
            leading_x = 20 + i * SHIFT + 40
            edge_band = flow[50:80, leading_x - 5:leading_x + 5]

            # Average only over pixels that carry an actual estimate. Lucas-
            # Kanade leaves untrackable pixels at fill_value, so an
            # unconditional mean would mostly measure the zero fill rather
            # than the recovered motion.
            estimated = np.linalg.norm(edge_band, axis=2) > 0.5
            if estimated.any():
                mean_dx = edge_band[..., 0][estimated].mean()
                mean_dy = edge_band[..., 1][estimated].mean()
                summary = f"mean_dx={mean_dx:+.2f} mean_dy={mean_dy:+.2f}"
            else:
                summary = "mean_dx=n/a mean_dy=n/a"

            magnitude = np.linalg.norm(flow, axis=2)
            print(
                f"  pair {i}->{i + 1}: shape={flow.shape} dtype={flow.dtype} "
                f"{summary} over {estimated.sum():4d}/{estimated.size} edge px, "
                f"max|flow|={magnitude.max():.2f}"
            )

    # Lucas-Kanade can additionally report which pixels it actually tracked,
    # separating "no motion" from "no estimate".
    lk = estimators[1]
    assert isinstance(lk, LucasKanadeOpticalFlow)
    flow, valid = lk.process_with_status(sequence)[0]
    print(
        f"\nLK tracking status: {valid.sum()} / {valid.size} pixels tracked "
        f"({100.0 * valid.mean():.1f}%)"
    )

    estimator = estimators[0]

    # Edge cases: empty and single-frame inputs return [] rather than raising.
    print(f"empty input  -> {estimator.process([])}")
    print(f"single frame -> {estimator.process([sequence[0]])}")

    # Degenerate case: a constant-intensity sequence normalizes to all-zero
    # frames, and therefore yields exactly zero flow.
    constant = [np.full((32, 32), 7000, dtype=np.uint16) for _ in range(3)]
    constant_flows = estimator.process(constant)
    print(
        f"constant seq -> {len(constant_flows)} fields, "
        f"max|flow|={max(float(np.abs(f).max()) for f in constant_flows):.2f}"
    )

    # Invalid inputs raise descriptive errors.
    for bad in (
        [sequence[0].astype(np.float32)],          # wrong dtype
        [np.zeros((8, 8, 3), dtype=np.uint16)],    # not single-channel
        [sequence[0], np.zeros((64, 64), np.uint16)],  # mismatched shapes
    ):
        try:
            estimator.process(bad)
        except (TypeError, ValueError) as error:
            print(f"rejected     -> {type(error).__name__}: {error}")
