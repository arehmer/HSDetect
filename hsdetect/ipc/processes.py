import time
from pathlib import Path
import warnings
import pickle as pkl
import numpy as np
import cv2

from hspy.tparray import TPArray

import hsdetect
from hsdetect.cv.detectors import build_detector
from hsdetect.cv.tracktors import build_tracktor
from hsdetect.cv.inf_engines import InferenceEngine

from hsdetect import build_engine

from hspy.ipc.processes import RProcess_R1, RWProcess_R1

from hsdetect.cv.posture_engines import PostureNetEngine



class CVProcess(RWProcess_R1):
    """
    Multiprocessing version of ProposalDetectorThread (ProcessorThread.py).
    Reads frames from read_buffer, runs object detection, writes the
    result (original data + detections) into write_buffer.
    """

    def __init__(self,
                 name: str,
                 model_dict: dict,
                 write_buffer,
                 read_buffer,
                 **kwargs):
        """
        Parameters
        ----------
        model_dict : dict
            Kwargs passed to build_detector(**model_dict) or 
            build_tracktor(**model_dict )inside
            _setup(). The detector/tracktor is built inside the child process
            rather than being constructed in the parent and pickled
            across, since detector/model objects frequently hold
            resources (loaded weights, a GPU/inference context, ...)
            that either aren't picklable or shouldn't be duplicated
            across a process boundary.
        write_buffer, read_buffer : Queue / SharedMemoryQueue

        Returns
        -------
        None.

        """

        self.model_dict = model_dict

        super().__init__(name=name,
                         read_buffer=read_buffer,
                         write_buffer=write_buffer,
                         **kwargs)

    def _setup(self):
        
        # Check if the provided model is a detector or tracker by name
        # and build the model (load weights / params, allocate whatever
        # runtime context it needs) here, inside the child process
        name = self.model_dict['name']  
        
        if name in hsdetect.cv.detectors.get_classes():
            print(f'Provided model {name} identified as detector.')
            self.model = build_detector(**self.model_dict)
        elif name in hsdetect.cv.tracktors.get_classes():
            print(f'Provided model {name} identified as tracktor.')
            self.model = build_tracktor(**self.model_dict)
        else:
            print(f'Provided model {name} could not be identified.')
            return None

        self.tparray = TPArray(SensorType=self.model.SensorType)

        self.num_pix = len(self.tparray._pix)

        self.t0 = time.time()

    def _target(self):

        # Get result from upstream process
        data = self.read_buffer.get()

        # Check success flag of upstream process
        if data['success'] is True:

            if 'pix_dK' in data.keys():
                frame = data['pix_dK']
            elif 'frame' in data.keys():
                frame = data['frame']

            # Reshape if not the proper size
            if frame.ndim == 1:
                frame = frame[0:self.num_pix]
                frame = frame.reshape(self.tparray._npsize)

            try:
                data_processed = self.model.process({'frame': frame,
                                                     'idx': data['image_id']},
                                                    update_bg = True)
                data.update(data_processed)

            except Exception as e:
                print(f"[{self.name}] {e}")

                # Set success flag to False in case of failure
                data['success'] = False

        else:
            # If upstream process failed, set success flag to False
            data['success'] = False

        return data

    def stop(self):
        self._exit.set()
        
        
class InferenceProcess(RWProcess_R1):
    """
    Class for running a hsdetect.cv.inference engine as a process..
    Reads frames from read_buffer, runs object detection, writes the
    result (original data + detections) into write_buffer.
    """
    
    def __init__(self,
                 name: str,
                 inference_engine_dir: Path,
                 write_buffer,
                 read_buffer,
                 **kwargs):
        
        self.inference_engine_dir = Path(inference_engine_dir)
        self.inference_engine = None
        self._internal_frame_idx = 0
        
        super().__init__(name=name,
                         read_buffer=read_buffer,
                         write_buffer = write_buffer,
                         **kwargs)
        
    def _setup(self,):
        
        # Build the model from the directory given
        # 1. Look for a .dict file that contains the inference engine para-
        # meters
        for file in self.inference_engine_dir.iterdir():
            
            if file.suffix == '.dict':
                
                with open(file,'rb') as io:
                    inf_engine_params = pkl.load(io)
                    try:
                        self.inference_engine = build_engine(**inf_engine_params)
                        print(f'Built inference engine with engines: {self.inference_engine.engine_names}.')
                    except:
                        pass
                
        if self.inference_engine is None:
            raise RuntimeError('Was not able to build inference engine from ' + \
                               f'{self.inference_engine_dir}.')
        
            
    def _target(self):

        # Get result from upstream process
        data = self.read_buffer.get()
        
        
        # If success flag is set, pass frame to inference engine
        if data['success'] == True:
            
            # Get frame 
            frame = data['frame'].reshape((120,160))
            
            try:
                # Pass frame to inference engine
                inference = self.inference_engine(frame)
                
                
                # Add result to dict
                data['inference'] = inference
                
                # Set success flag to True
                data['success'] = True
                
            except Exception as e:
                print(f"[{self.name}] {e}")
                
                # Set success flag to False in case of failure
                data['success'] = False

        return data


class PostureInferenceProcess(RWProcess_R1):
    """
    Class for running a hsdetect.cv.posture_engines.PostureNetEngine as a
    process. Reads raw frames from read_buffer, runs background removal,
    the posture CNN and its post-processing, and writes the result
    (original data + posture annotations) into write_buffer.

    The posture counterpart of InferenceProcess. It differs in two ways,
    both of which follow from PostureNetEngine being self-contained:

      * There is no engine CHAIN. PostureNetEngine does its own background
        estimation and removal, so this process hands it the raw frame and
        there is no BackgroundSub / ZeroCorners / decode stage to wire up.
      * The engine is built from a parameter dict rather than from a saved
        pipeline, although a directory holding a pickled '.dict' is
        accepted too, the way InferenceProcess takes one.

    As in CVProcess, the engine is constructed inside the child process and
    not pickled across: PostureNetEngine holds a loaded torch model, which
    is both expensive and awkward to send over a process boundary.

    The first frames are consumed by the background estimate and produce no
    annotations -- see PostureNetEngine.update_background(). They must show
    the EMPTY room.
    """

    def __init__(self,
                 name: str,
                 write_buffer,
                 read_buffer,
                 engine_dict: dict = None,
                 engine_dir: Path = None,
                 ArrayType: int = None,
                 SensorType: int = None,
                 **kwargs):
        """
        Parameters
        ----------
        engine_dict : dict
            Parameters of the engine, e.g.

                {'name': 'PostureNetEngine',
                 'SensorType': SensorTypes['HTPA160x120DR1_L3K95_0K8'],
                 'model_path': '.../CNNDetection/checkpoints/best.pt'}

            Everything PostureNetEngine takes may be set here (bg_frames,
            motion_gap, alarm_motion, background, device, ...).
        engine_dir : Path
            Alternative to engine_dict: a directory holding a pickled
            '.dict' of those same parameters, as written by
            PostureNetEngine.save(). Mirrors InferenceProcess.
        ArrayType, SensorType : int
            Used only to reshape a flat incoming frame, as CVProcess does.
            Optional: without either, the engine reshapes the frame itself
            from the geometry its checkpoint declares.
        """

        if engine_dict is None and engine_dir is None:
            raise ValueError('Either engine_dict or engine_dir must be '
                             'provided.')

        self.engine_dict = engine_dict
        self.engine_dir = None if engine_dir is None else Path(engine_dir)

        if ArrayType is not None:
            self._tparray_kwargs = {'ArrayType': ArrayType}
        elif SensorType is not None:
            self._tparray_kwargs = {'SensorType': SensorType}
        else:
            self._tparray_kwargs = None

        self.engine = None

        super().__init__(name=name,
                         read_buffer=read_buffer,
                         write_buffer=write_buffer,
                         **kwargs)

    def _setup(self):

        params = dict(self.engine_dict) if self.engine_dict is not None \
            else self._params_from_dir()

        # Built directly rather than through build_engine(), which walks and
        # imports every module of hsdetect to find the one that knows this
        # engine -- tensorflow included. The posture path needs torch only,
        # and keeping it independent of the detector path is the point of
        # this process. A dict naming some other engine still goes through
        # the registry.
        engine_name = params.pop('name', 'PostureNetEngine')

        if engine_name == 'PostureNetEngine':
            self.engine = PostureNetEngine(**params)
        else:
            self.engine = build_engine(name=engine_name, **params)

        print(f'[{self.name}] {engine_name} built from '
              f'{self.engine.model_path}.')

        # Optional, for reshaping a flat frame the way CVProcess does.
        if self._tparray_kwargs is not None:
            self.tparray = TPArray(**self._tparray_kwargs)
            self.num_pix = len(self.tparray._pix)
        else:
            self.tparray = None
            self.num_pix = None

        self.t0 = time.time()

    def _params_from_dir(self) -> dict:
        """Load the engine parameters from the first '.dict' in engine_dir."""

        for file in sorted(self.engine_dir.iterdir()):
            if file.suffix == '.dict':
                with open(file, 'rb') as io:
                    return pkl.load(io)

        raise RuntimeError(f'No .dict with engine parameters found in '
                           f'{self.engine_dir}.')

    def _target(self):

        # Get result from upstream process. Through _get(), not
        # read_buffer.get(): a bare get() blocks forever once the stage
        # upstream has ended, so stop() would never be noticed.
        data = self._get()

        # Upstream failed (a dropped frame, or the final 'device_lost'
        # item): pass it on untouched so the viewer can report it.
        if data.get('success') is not True:
            data['success'] = False
            return data

        if 'frame' in data.keys():
            frame = data['frame']
        elif 'pix_dK' in data.keys():
            frame = data['pix_dK']
        else:
            frame = data['pix']

        # Drop whatever the packet carries behind the pixels and reshape,
        # as CVProcess does. Skipped without a TPArray, in which case the
        # engine reshapes a flat frame of the right length itself.
        if frame.ndim == 1 and self.tparray is not None:
            frame = frame[0:self.num_pix]
            frame = frame.reshape(self.tparray._npsize)

        try:
            # One call does background removal, the affine, the CNN and the
            # post-processing, and returns the annotations.
            annotations = self.engine({'frame': frame,
                                       'idx': data.get('image_id')})

            last = self.engine.last or {}

            # The annotation DataFrame, empty while the background is still
            # being estimated and on a frame classified 'no_person'.
            data['posture'] = annotations

            # The per-frame result itself, which is there even when no row
            # was emitted -- the viewer needs it either way.
            data['posture_result'] = {key: last.get(key) for key in
                                      ('tag', 'conf', 'E_k', 'probs',
                                       'majority', 'motion', 'alarm')}

            # The background-removed frame the network actually saw.
            data['frame_nobg'] = last.get('frame_nobg')
            data['background_established'] = self.engine.background_established

        except Exception as e:
            print(f"[{self.name}] {e}")

            # Set success flag to False in case of failure
            data['success'] = False

        return data

    def stop(self):
        self._exit.set()


class ImshowPostureProcess(RProcess_R1):
    """
    Multiprocessing viewer for PostureInferenceProcess's output. Reads from
    read_buffer and draws, in its own process, the background-removed frame
    the CNN saw with the posture result burned into the corner.

    The posture counterpart of ImshowProcess. It renders the way
    CNNDetection/renderCNN.py renders, so that what you see live is
    comparable with the imagesCNN/*.png that project writes offline:

      * a FIXED colour scale in dK above background (0..60 by default), not
        a per-frame min-max stretch, so the same temperature is the same
        colour in every frame - the whole point of the fixed input affine;
      * integer NEAREST upscaling, so a sensor pixel stays a visible square
        and nothing is interpolated into existence;
      * the tag and its confidence, the majority tag and the motion value
        top left, and a large red ALARM top right when the gate fires.

    Boxes are drawn only for annotations SMALLER than the whole frame. The
    posture CNN classifies the frame rather than localising in it, so its
    annotation covers everything and a rectangle around the border would
    only be visual noise.
    """

    def __init__(self,
                 name: str,
                 read_buffer,
                 ArrayType: int = None,
                 SensorType: int = None,
                 window_name: str = 'Posture stream',
                 vmin: float = 0.0,
                 vmax: float = 60.0,
                 scale: int = 6,
                 show_ek: bool = True,
                 colormap: str = 'INFERNO',
                 **kwargs):
        """
        vmin, vmax : the fixed colour scale in dK above background. The
                     defaults are renderCNN.py's DEFAULT_VMIN / DEFAULT_VMAX.
        scale      : integer upscaling factor; 6 puts a 160x120 sensor at
                     960x720, about renderCNN.py's default output width.
        show_ek    : also burn in the expected criticality E[k].
        colormap   : name of a cv2.COLORMAP_* constant. 'INFERNO' matches
                     the colour recipe of the project's own PNG converters.
        """

        if ArrayType is None and SensorType is not None:
            self._tparray_kwargs = {'SensorType': SensorType}
        elif ArrayType is not None and SensorType is None:
            self._tparray_kwargs = {'ArrayType': ArrayType}
        else:
            raise Exception('Either ArrayType or SensorType must be provided.')

        self.window_name = window_name
        self.vmin = vmin
        self.vmax = vmax
        self.scale = int(scale)
        self.show_ek = show_ek
        self.colormap = colormap

        super().__init__(name=name,
                         read_buffer=read_buffer,
                         **kwargs)

    def _setup(self):

        # TPArray construction and window creation both happen here,
        # inside the child process.
        self.tparray = TPArray(**self._tparray_kwargs)
        self.num_pix = len(self.tparray._pix)

        # Resolved here and not in __init__ so that an OpenCV build without
        # this particular colormap falls back in the child process, where
        # the window lives, rather than raising in the parent.
        self._cmap = getattr(cv2, f'COLORMAP_{self.colormap.upper()}',
                             cv2.COLORMAP_HOT)

        self.t0 = time.time()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def _target(self):

        # Get result from upstream process
        data = self._get()

        # Nothing to draw: a failed acquisition attempt, or the final
        # 'device_lost' item the client hands over before it ends. Keep the
        # window responsive instead of raising on a missing frame.
        if not data.get('success', True):
            if data.get('device_lost'):
                print(f"[{self.name}] Upstream reports the device as lost - "
                      f"no further frames.")
            cv2.waitKey(1)
            return

        frame_nobg = data.get('frame_nobg')

        if frame_nobg is None:
            # The background is still being estimated, so there is no
            # contrast frame yet. Show the raw frame, stretched, with a
            # banner - the first frames have to show the EMPTY room, and
            # somebody standing in them is the one mistake worth catching
            # while it is still happening.
            image = self._render_raw(data)
            self._draw_label(image, 'estimating background...\n'
                                    'keep the room empty')
        else:
            image = self._render_contrast(frame_nobg)
            self._draw_posture(image, data)

        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)

    def _render_contrast(self, frame_nobg):
        """Fixed-range normalise, colour-map and integer-upscale one NoBG
        frame -- renderCNN.py::renderFrame()'s recipe."""

        span = self.vmax - self.vmin if self.vmax > self.vmin else 1.0
        norm = np.clip((np.asarray(frame_nobg, dtype=np.float64) - self.vmin)
                       / span, 0.0, 1.0)

        image = cv2.applyColorMap((norm * 255).astype(np.uint8), self._cmap)

        return self._upscale(image)

    def _render_raw(self, data):
        """Min-max stretch of the raw frame, for the background phase only.

        A fixed scale is meaningless here: these are absolute sensor values
        around 3000 dK, not contrast above a background that does not exist
        yet.
        """

        if 'frame' in data.keys():
            frame = data['frame']
        elif 'pix_dK' in data.keys():
            frame = data['pix_dK']
        else:
            frame = data['pix']

        if frame.ndim == 1:
            frame = frame[0:self.num_pix]
            frame = frame.reshape(self.tparray._npsize)

        frame = cv2.normalize(frame.astype(np.float32), None, 0, 255,
                              cv2.NORM_MINMAX).astype(np.uint8)

        return self._upscale(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

    def _upscale(self, image):
        """Integer NEAREST upscale: a sensor pixel stays a visible square."""
        sf = max(1, self.scale)
        return np.repeat(np.repeat(image, sf, axis=0), sf, axis=1)

    def _draw_posture(self, image, data):
        """Burn the posture result, any sub-frame box, and the ALARM in."""

        result = data.get('posture_result') or {}

        tag = result.get('tag')
        conf = result.get('conf')
        motion = result.get('motion')

        if tag is None:
            return

        # Line 1: the per-frame tag and its confidence. Line 2: the majority
        # tag over the voting window. Line 3: the motion value, '--' while
        # it is still warming up. As renderCNN.py labels its frames.
        label = f"{tag}  {conf:.2f}"
        if self.show_ek and result.get('E_k') is not None:
            label += f"  E[k]={result['E_k']:.2f}"
        label += f"\nmaj {result.get('majority')}"
        label += ("\nmotion  --" if motion is None
                  else f"\nmotion {motion:.1f}")

        self._draw_label(image, label)

        self._draw_boxes(image, data.get('posture'))

        if result.get('alarm'):
            self._draw_alarm(image)

    def _draw_boxes(self, image, annotations):
        """Draw only annotations that are smaller than the whole frame.

        A whole-frame box is what the posture CNN emits by design, and
        outlining the entire image says nothing. A smaller one means the
        engine was configured to localise, and then it is worth seeing.
        """

        if annotations is None or len(annotations) == 0:
            return

        sf = max(1, self.scale)
        height, width = image.shape[0] // sf, image.shape[1] // sf

        for idx in annotations.index:
            box = annotations.loc[[idx]]

            x, y = int(box['xtl'].item()), int(box['ytl'].item())
            w = int(box['xbr'].item()) - x
            h = int(box['ybr'].item()) - y

            if w >= width and h >= height:
                continue

            cv2.rectangle(image, (sf * x, sf * y), (sf * (x + w), sf * (y + h)),
                          (0, 255, 255), 1)

    def _draw_label(self, image, text):
        """White text with a black outline in the top-left corner.

        Drawn twice, thick black then thin white, which is how a stroke is
        done in OpenCV -- it has no equivalent of the stroke_width
        renderCNN.py's Pillow drawLabel() uses, and plain white text is
        unreadable over the bright end of the colour map.
        """

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4 * max(1, self.scale) / 6.0
        thickness = max(1, int(round(font_scale * 2)))
        line_height = int(28 * font_scale) + 10
        margin = max(4, 2 * self.scale)

        for i, line in enumerate(text.split('\n')):
            org = (margin, margin + line_height * (i + 1))
            cv2.putText(image, line, org, font, font_scale, (0, 0, 0),
                        thickness + 2, cv2.LINE_AA)
            cv2.putText(image, line, org, font, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)

    def _draw_alarm(self, image):
        """Large red ALARM in the top-right corner, opposite the label so
        the two never overlap -- renderCNN.py::drawAlarm()."""

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2 * max(1, self.scale) / 6.0
        thickness = max(2, int(round(font_scale * 3)))
        text = 'ALARM'

        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale,
                                              thickness + 2)
        margin = max(6, 2 * self.scale)
        org = (image.shape[1] - text_w - margin, margin + text_h)

        cv2.putText(image, text, org, font, font_scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
        cv2.putText(image, text, org, font, font_scale, (0, 0, 255),
                    thickness, cv2.LINE_AA)

    def _on_idle(self):
        """Called by RProcess_R1.run() while no item arrives. Keeps the
        window responsive, so it still redraws and reacts to the window
        manager while the stream has not started yet or the device is
        gone."""
        try:
            cv2.waitKey(1)
        except cv2.error:
            pass

    def _teardown(self):
        # Window is destroyed here, inside the child process, once the loop
        # has exited.
        cv2.destroyWindow(self.window_name)

    def stop(self):
        self._exit.set()
