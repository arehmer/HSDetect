"""
inference.py

Constructs the missing inference-side pipeline: background subtraction ->
network (ONNX) -> decode -> background update, wired together so the SAME
per-frame step works for:

  1. Live/streaming data: call .step(frame) once per incoming raw frame.
  2. A whole AnnotatedVideo: .process_video(video) loops .step() over
     every frame in order, exactly as if they'd arrived live.

Background update convention at inference time
------------------------------------------------
During TRAINING, background.py's update_bg() is fed GROUND-TRUTH boxes
(see Section 3.2 of Vandersteegen et al., and how BackgroundModel.
_process_avideo() was originally sketched with a use_gt flag). At
INFERENCE time there is no ground truth -- so update_bg() is fed the
model's OWN decoded detections from that same step instead. This is a
self-referential loop: each frame's background is updated using what the
model just predicted, then that updated background is used to
background-subtract the NEXT incoming frame.

This module does not compute detection accuracy metrics (F1/AP against
ground truth, matching the paper's Table 1) -- it only produces
detections. See the note at the bottom of the accompanying explanation
for how that would build on top of this.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import logging


try:
    from ..misc.annot import AnnotatedVideo, AnnotatedFrame
except:
    from hsdetect.annot.annot import AnnotatedVideo, AnnotatedFrame
    
from hsdetect import build_engine, Engine, Payload, SAME

logger = logging.getLogger(__name__)

IS_ENGINE_MODULE = True

def build_inference_engine(**kwargs):
    name = kwargs.pop('name')
    if name == 'InferenceEngine':
        return InferenceEngine(**kwargs)
    else:
        raise TypeError(f'Unknown Inference Engine {name}!')


import ast       
def get_classes():
    with open(__file__, 'r') as f:
        tree = ast.parse(f.read(), filename=__file__)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


def _engine_param_name(entry):
    """Name of one entry of a params['engines'] list, which is a parameter
    dict after save(), but may be an already-built Engine, since build_engine()
    accepts either."""
    if isinstance(entry,dict):
        return entry.get('name')
    return getattr(entry,'name',None)


class InferenceEngine(Engine):
    """Wraps a trained YOLOv2Tiny model and any number of pre- or post-
    processing engines into a frame-by-frame inference engine.
    """
    
    name = 'InferenceEngine'

    # A chain is itself an engine: it takes the frame the sensor delivers
    # and produces detections. `emits` is replaced per instance by
    # _check_chain(), with whatever this particular chain ends on.
    accepts = Payload.FRAME_RAW
    emits = Payload.BOXES
    
    def __init__(self,**attr_dict:dict):

        
        # Store the list of engines as an attribute
        self.engines = attr_dict.pop('engines',None)  
    
        # Make sure every passed engine is a valid engien
        self._check_engines()
        
        # Bookkeeping
        self._frame_idx = 0  

                
    @property
    def engines(self):
        return self._engines
    
    @engines.setter
    def engines(self,engines:list):
        
        # Type check
        if not isinstance(engines, list):
            raise TypeError(f'Expected engines to be {list}, but received{type(engines)}.')
        
        # List for storing built_engines
        built_engines = []
        
        for engine in engines:

            if isinstance(engine,dict):
                built_engines.append(build_engine(**engine))
            elif isinstance(engine,Engine):
                built_engines.append(engine)
            else:
                raise Exception(f'Engine {engine} not recognized.')
        
        # Assign built engines to member attribute
        self._engines = built_engines
        
        # Build or rebuild the inference pipeline
        self._build_inference_pipeline()

    @property
    def engine_names(self) -> list:
        """Names of this pipeline's engines, in the order they are executed.

        The names are the engines' `name` class attributes, which is also the
        key `Engine.save()` stores them under -- so the same name identifies an
        engine both in a built pipeline and in a saved parameter dict.
        """
        return list(self._inference_pipeline.keys())

    def get_engine(self, name: str) -> Engine:
        """Return the engine called `name`.

        Use this instead of indexing `self.engines`: a positional index
        silently returns the WRONG engine as soon as an engine is inserted into
        or removed from the chain, whereas a wrong name raises here, naming
        every engine the pipeline actually has.

        Parameters
        ----------
        name : str
            An engine's `name`, e.g. 'BackgroundSub' or
            'YOLO_EnDecode_Engine'. See `engine_names`.

        Raises
        ------
        TypeError
            If `name` is not a string. Deliberately not an index: this method
            exists to replace positional access, and silently accepting an int
            would defeat that.
        KeyError
            If no engine of that name is in the pipeline. The message lists
            `engine_names`, since the usual cause is a renamed or reordered
            chain.
        """
        if not isinstance(name, str):
            raise TypeError(f'Engine name must be {str}, received '
                            f'{type(name)}. Engines are looked up by name, '
                            f'not by position -- see self.engine_names.')

        try:
            return self._inference_pipeline[name]
        except KeyError:
            # from None: the bare KeyError('BackgroundSub') of the dict lookup
            # adds nothing to the message built here.
            raise KeyError(f"No engine named '{name}' in this pipeline. "
                           f"Available engines, in execution order: "
                           f"{self.engine_names}.") from None

    def __getitem__(self, name: str) -> Engine:
        """`inf_engine['BackgroundSub']` -- shorthand for get_engine().

        Note that this does NOT make the pipeline iterable: get_engine()
        rejects the integer indices Python's legacy iteration protocol would
        pass, with the same TypeError iterating this object raised before this
        method existed. Iterate `self.engines` instead.
        """
        return self.get_engine(name)

    @staticmethod
    def get_engine_params(params: dict, name: str) -> dict:
        """Look an engine up by name in a SAVED parameter dict, i.e. before
        any engine has been built.

        The counterpart to get_engine(), for the other side of the pickle.
        `Engine.name` is a class attribute and `save()` stores it under the
        'name' key, so one and the same name identifies an engine in a built
        pipeline and in `InferenceEngine.save()`'s output -- which is what lets
        a caller drop positional indices into params['engines'] entirely.

        The entry is returned AS IS, not copied, so that writing to it writes
        through to `params`:

            InferenceEngine.get_engine_params(params,'DLEngine')['dl_model']                 = remapped_path
            inf_engine = build_engine(**params)

        That in-place edit is the whole point -- handing back a copy would turn
        such a remap into a silent no-op, which is a worse failure than the
        positional indexing this replaces.

        Parameters
        ----------
        params : dict
            A dict as produced by `InferenceEngine.save()`, i.e. carrying an
            'engines' list of per-engine parameter dicts. The whole dict, not
            its 'engines' value, so that callers never touch the structure.
        name : str
            The engine's `name`, e.g. 'DLEngine'.

        Returns
        -------
        dict
            That engine's parameter dict, still part of `params`.

        Raises
        ------
        TypeError
            If `params` is not a dict, `name` is not a string, or the entry
            found is an already-built Engine rather than a parameter dict (use
            get_engine() on the built pipeline for that).
        KeyError
            If `params` has no 'engines' key, or no engine of that name is in
            it -- the message lists the names that are.
        ValueError
            If several entries carry that name, since returning the first
            would silently hide the others.
        """
        if not isinstance(name, str):
            raise TypeError(f'Engine name must be {str}, received '
                            f'{type(name)}. Engines are looked up by name, '
                            'not by position.')

        if not isinstance(params, dict):
            raise TypeError(f'Expected a saved parameter dict ({dict}), '
                            f'received {type(params)}.')

        try:
            engines = params['engines']
        except KeyError:
            raise KeyError("Parameter dict carries no 'engines' key, so it is "
                           'not an InferenceEngine.save() result. Its keys '
                           f'are: {sorted(params.keys())}.') from None

        # Entries are parameter dicts after save(), but build_engine() also
        # accepts already-built Engines in that list, so _engine_param_name()
        # reads the name off either and the type check below produces the
        # better message.
        matches = [entry for entry in engines
                   if _engine_param_name(entry) == name]

        if not matches:
            raise KeyError(f"No engine named '{name}' in this parameter dict. "
                           'Engines it carries, in execution order: '
                           f'{[_engine_param_name(entry) for entry in engines]}.')

        if len(matches) > 1:
            raise ValueError(f"{len(matches)} engines named '{name}' in this "
                             'parameter dict -- a name has to identify one '
                             'engine, since the built pipeline is keyed by '
                             'it. Give them distinct names.')

        if not isinstance(matches[0],dict):
            raise TypeError(f"Engine '{name}' in this parameter dict is an "
                            f'already-built {type(matches[0])}, not a '
                            'parameter dict. Build the pipeline and use '
                            'InferenceEngine.get_engine() instead.')

        return matches[0]

    def __call__(self, data, frame_idx = None, conf_min = 0.9, **kwargs):

        # Check type of passed data
        if isinstance(data,AnnotatedVideo):
            
            video_res = {}
            
            for aframe in data:
                frame = aframe.image
                frame_idx = aframe.index
                
                inf_res = self.step(frame,
                                    frame_idx)
                
                # The single-channel masked frame is the one worth storing
                # next to the boxes; a chain without one (no ZeroCorners)
                # falls back to the frame it was handed. Both are resolved by
                # payload, so reordering the chain does not break this.
                processed = (self._output_for(inf_res, Payload.FRAME_MASKED)
                             or self._output_for(inf_res, Payload.FRAME_RAW))

                frame = processed['frame']
                idx = processed['idx']
                df_annot = inf_res[self._boxes_source]
                if len(df_annot)!=0:
                    df_annot = df_annot.loc[df_annot['conf'] >= conf_min]
                
                video_res[idx] = (frame,df_annot)
            
            # Convert to AnnotatedVideo
            aframes = []
            for idx, res in video_res.items():
                aframe = AnnotatedFrame(idx,res[0],res[1])
                aframes.append(aframe)
                
            avideo = AnnotatedVideo()
            avideo.from_AnnotatedFrames(frames=aframes)
            return avideo
        
        else:
            frame = data
            return self.step(frame,
                             frame_idx,
                             **kwargs)
        
        
    # def process(self,data,**kwargs):
    #     if isinstance(data,dict):
    #         return self.step(frame = data['frame'],
    #                          idx = data['idx'],
    #                          **kwargs)
    #     elif isinstance(data,AnnotatedVideo):
    #         return self._process_avideo(data,**kwargs)
    #     else:
    #         raise NotImplementedError()

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
            if isinstance(attr, property):
                
                # Derived from the engines when built, not a parameter
                if attr_name == 'engine_names':
                    continue
                
                # Retrieve the value of the property attr
                attr_val = getattr(self, attr_name)
                
                # Exeption for list of engines
                if attr_name == 'engines':
                    
                    attr_dict['engines'] = []
                    engines = attr_val
                    
                    for engine in engines:
                        attr_dict['engines'].append(engine.save())
                
                else:
                
                    # If the property has a save() method, save the return value of
                    # that method
                    if hasattr(attr_val, "save") and callable(attr_val.save):
                        attr_dict[attr_name] = attr_val.save()
                    else:
                        attr_dict[attr_name] = attr_val
                
        # Append name
        attr_dict['name'] = self.name
                
        return attr_dict

    def step(self,
             frame : np.ndarray,
             frame_idx : int, 
             **kwargs):
        
        # db output
        # print(f'Frame index {self._frame_idx}.')
        
        # If ground truth annotations are provided, these will be used for 
        # updating the background engine
        gt = kwargs.pop('gt', None)
        
        # If index is provided (e.g. because a video with indexed frames is 
        # processed) use that index, otherwise use the internal counter
        frame_idx = self._frame_idx if frame_idx is None else frame_idx
        
        # Initialize a dictionary for storing the input, intermediate results,
        # and final output of the inference pipeline
        inference = {}
        inference['input'] = {'frame':frame, 
                              'idx': frame_idx}    # the first element is the raw frame
               
        # Run all engines after each other in the order that they were given
        for engine_name,engine in self._inference_pipeline.items():
            
            # print(f'Processing engine: {engine_name}')
            # Obtain the input to the engine from self.IOmapping
            engine_input = self.IOmapping[engine_name]
            
            # Feed the input to the engine, and store the engines output under
            # its name 
            try:
                inference[engine_name] = engine(inference[engine_input])
            except Exception:
                logger.exception("Engine %s failed", engine_name)
                raise
        
        
        
        # If there is an engine that requires updating (i.e. BackgroundSub)
        # do that here
        self.update_engines(inference,gt)
        
        # for engine_name in self._engines_requiring_update:
        #     self._inference_pipeline[engine_name].update_step()
            
        # if gt_annot is not None:
        #     self._inference_pipeline[engine_name]
        
        # Increment the internal counter (whether it is used or not)
        self._frame_idx += 1
            
        return inference
    
    def update_engines(self,
                       inference : dict,
                       gt = None):
        """Run the chain's backward edges after a forward pass.

        Which engines have one, what they are handed and which method is
        called is declared by the engines themselves, in Engine.feedback, and
        validated by _check_chain(). This method used to name BackgroundSub,
        ZeroCorners and YOLO_EnDecode_Engine literally, which made it correct
        for exactly one chain order and silently wrong for any other.

        `gt` replaces the chain's own detections when ground-truth boxes are
        given, which is how the background is updated while preparing
        training data.
        """
        for engine in self.engines:

            spec = engine.feedback

            if spec is None:
                continue

            args = []

            if spec.frame:
                payload = inference[self.IOmapping[engine.name]]
                args.append(payload['frame'] if isinstance(payload, dict)
                            else payload)

            if spec.boxes:
                args.append(gt if gt is not None
                            else inference[self._boxes_source])

            getattr(engine, spec.method)(*args)

    def _process_avideo(self,
                        avideo : AnnotatedVideo,
                        **kwargs) -> AnnotatedVideo:
        """
        Function for processing an AnnotatedVideo.

        Parameters
        ----------
        video : AnnotatedVideo
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        AnnotatedVideo
            DESCRIPTION.

        """
        
        gt_update = kwargs.pop('gt_update',False)
        
        # List for storing results for each frame
        frame_results = []
        
        # Loop over whole video
        for aframe in avideo:
            
            # Get frame
            frame = aframe.image
            
            # Get annotations
            if gt_update:
                gt = aframe.annotations
            else:
                gt = None
            
            
            # Pass frame to inference pipeline via step()
            inference_result = self.step(frame,
                                         gt = gt)
            
            # The final result is the output of the last engine
            last_engine_name = list(self._inference_pipeline.keys())[-1]
            frame_result = inference_result[last_engine_name]
            
            # Build an AnnotatedFrame from the last result
            frame_results.append(frame_result)
            
        # Put the result together to an AnnotatedVideo. Some filters are acausal,
        # therefore check index of each frame, keep only the latest result
        # for every frame if multiple exist.
        avideo_dict = {}
        for result in frame_results:
            avideo_dict[result['idx']] = AnnotatedFrame(result['idx'],
                                                        result['frame'],
                                                        result.pop('annotations',None))
        
        return AnnotatedVideo(list(avideo_dict.values()))
            
            
            
            
    
    def _output_for(self, inference : dict, payload : Payload):
        """The output of the LAST engine of this chain that emitted `payload`,
        or None if no engine does.

        Asking what a result IS, rather than which engine produced it, is what
        lets a caller keep working when the chain is reordered or an engine is
        swapped for another that emits the same thing.
        """
        found = None

        for engine in self.engines:
            if self._engine_outputs.get(engine.name) is payload:
                found = inference.get(engine.name)

        return found

    def _build_inference_pipeline(self) -> None:

        # The inference pipeline is just an ordered dict of engines
        self._inference_pipeline = {engine.name : engine for engine in self.engines}

        # Work out what reaches each engine, and reject a chain that cannot
        # work, before a single frame is pushed through it
        self._check_chain()

    def _check_chain(self) -> None:
        """Work out what reaches each engine, and reject a chain that cannot
        work.

        Everything here comes from the engines' own declarations
        (Engine.accepts, Engine.emits, Engine.feedback), never from engine
        names, so that a chain in a different order gets checked rather than
        silently mis-wired. Two things are checked:

        1. The forward flow. The chain is handed a FRAME_RAW; every engine is
           asked whether it accepts what the engine before it emitted. This is
           what catches a chain putting BackgroundSub before ZeroCorners:
           BackgroundSub needs the dead corners zeroed, and says so in
           `accepts`. Such a chain used to build happily and then feed
           update_bg() a None frame on every step, so the background never
           became valid, subtract() returned None forever, and the chain
           produced no detections at all -- with no exception anywhere.

        2. The backward edges. An engine declaring a Feedback needs the chain
           to actually produce what that feedback is fed with, and needs the
           method it names to exist.

        An engine declaring neither `accepts` nor `emits` is untyped: it is
        assumed to pass its payload on unchanged, so that an engine without a
        contract can still sit in a chain.
        """
        names = [engine.name for engine in self.engines]

        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f'Chain contains more than one engine called {duplicates}. '
                'Engines are addressed by name, so one would hide the other. '
                f'Chain: {" -> ".join(names)}.')

        # Which engine's output each engine is handed. The chain is linear;
        # 'input' is the frame step() was called with.
        mapping = {names[0]: 'input'}
        for curr, prev in zip(names[1:], names[:-1]):
            mapping[curr] = prev
        self.IOmapping = mapping

        payload = self.accepts if isinstance(self.accepts, Payload) \
            else Payload.FRAME_RAW
        inputs, outputs = {}, {}

        for position, engine in enumerate(self.engines):

            accepted = engine.accepted_payloads()

            if accepted and payload not in accepted:
                source = ('the frame the chain is called with' if position == 0
                          else "'" + names[position - 1] + "'")
                raise ValueError(
                    f"Engine '{engine.name}' at position {position} of this "
                    f"chain accepts {', '.join(str(x) for x in accepted)}, "
                    f'but {source} provides {payload}. '
                    f'Chain: {" -> ".join(names)}.')

            inputs[engine.name] = payload

            emitted = payload if engine.emits in (None, SAME) else engine.emits
            outputs[engine.name] = emitted
            payload = emitted

        self._engine_inputs = inputs
        self._engine_outputs = outputs
        self.emits = payload

        # The engine the backward edges are fed from: the last one in the
        # chain that produces boxes.
        boxes_source = None
        for engine in self.engines:
            if outputs[engine.name] is Payload.BOXES:
                boxes_source = engine.name
        self._boxes_source = boxes_source

        for engine in self.engines:

            spec = engine.feedback

            if spec is None:
                continue

            if not callable(getattr(engine, spec.method, None)):
                raise ValueError(
                    f"Engine '{engine.name}' declares the feedback method "
                    f"'{spec.method}', which it does not have.")

            if spec.frame and not inputs[engine.name].is_frame:
                raise ValueError(
                    f"Engine '{engine.name}' is fed back the payload it "
                    f'receives, which has to be a frame, but it receives '
                    f'{inputs[engine.name]}. Chain: {" -> ".join(names)}.')

            if spec.boxes and boxes_source is None:
                raise ValueError(
                    f"Engine '{engine.name}' is updated with this chain's "
                    f'detections, but no engine in the chain emits '
                    f'{Payload.BOXES}. Chain: {" -> ".join(names)}.')

    def _check_engines(self):
        
        for engine in self.engines:
            
            # Check if engine has name attribute
            if not hasattr(engine,'name'):
                raise ValueError(f'{engine} has no attribute "name".')
                
            # Check if engine has a call function
            if not hasattr(engine,'__call__'):
                raise ValueError(f'{engine} has no attribute "__call__"')
            
            if not callable(engine.__call__):
                raise ValueError(f'engine.call is not callable but {type(engine.call)}')
                


# class InferenceEngine:
#     """Wraps a trained YOLOv2Tiny model + a BackgroundModel into a single
#     stateful, frame-by-frame detector.

#     Parameters
#     ----------
#     model : YOLOv2Tiny
#         A model instance already built at some fixed input_shape.
#     anchors : array-like of (w, h) pairs
#         Same anchors the model was trained with.
#     background_model : BackgroundModel
#         A BackgroundModel instance sized for this model's input_shape
#         (i.e. background_model.htpa.h/.w must match model.input_shape[:2]).
#         Pass a fresh, not-yet-initialized instance to start a new stream;
#         pass an already-warmed-up instance to resume one.
#     num_classes, conf_threshold, nms_iou_threshold, max_detections :
#         Forwarded to decode_predictions() on every step.
#     """

#     def __init__(
#         self,
#         model,
#         anchors,
#         background_engine: BackgroundEngine,
#         num_classes: int = 1,
#         conf_threshold: float = 0.5,
#         nms_iou_threshold: float = 0.3,
#         max_detections: int = 100,
#     ):
#         self.model = model
#         self.anchors = np.asarray(anchors, dtype=np.float32)
#         self.bg_engine = background_engine
#         self.num_classes = num_classes
#         self.conf_threshold = conf_threshold
#         self.nms_iou_threshold = nms_iou_threshold
#         self.max_detections = max_detections

#         self.img_h, self.img_w, _ = model.input_shape
#         self._frame_counter = 0

#     def step(self, raw_frame: np.ndarray) -> FrameResult:
#         """Process ONE new raw (uint16) frame:
#           1. Background-subtract it (if the background isn't established
#              yet, no detection is possible this step -- see below).
#           2. Run the model and decode its output into pixel-space boxes.
#           3. Update the background using THIS step's own detections
#              (not ground truth -- see module docstring).

#         Returns a FrameResult with whatever detections were found (empty
#         if the background wasn't established yet).
#         """
#         idx = self._frame_counter
#         self._frame_counter += 1

#         subtracted = self.bg_engine.subtract(raw_frame)

#         if subtracted is None:
#             # Background not established yet (per N_init). No detection is
#             # possible this step -- but we still call update_bg() so the
#             # background keeps progressing toward being established. No
#             # detections exist yet to exclude by, so pass an empty list.
#             self.bg_engine.update_bg(raw_frame, [])
#             return FrameResult(index=idx, detections=[], background_established=False)

#         net_input = subtracted[np.newaxis, :, :, np.newaxis].astype(np.float32)
#         y_pred = self.model(net_input, training=False)[0]  # drop batch dim

#         detections = decode_predictions(
#             y_pred, self.anchors, self.img_h, self.img_w,
#             num_classes=self.num_classes,
#             conf_threshold=self.conf_threshold,
#             nms_iou_threshold=self.nms_iou_threshold,
#             max_detections=self.max_detections,
#         )

#         # Update background using the model's OWN detections (inference-time
#         # convention -- see module docstring).
#         boxes_for_update = [(d.xtl, d.ytl, d.xbr, d.ybr) for d in detections]
#         self.bg_engine.update_bg(raw_frame, boxes_for_update)

#         return FrameResult(index=idx, detections=detections, background_established=True)
    
#     def process(self,data,**kwargs):
#         if isinstance(data,dict):
#             return self._process_frame(frame = data['frame'],
#                                        idx = data['idx'],
#                                        **kwargs)
#         elif isinstance(data,AnnotatedVideo):
#             return self._process_avideo(data,**kwargs)
#         else:
#             raise NotImplementedError()

#     def _process_frame(self,
#                        frame:np.ndarray,
#                        idx:int,
#                        **kwargs:dict) -> dict:
        
#         # Get keyword arguments
#         gt_annot = kwargs.pop('gt_annot',None)
        
#         update_bg = kwargs.pop('update_bg',False)
#         use_gt = kwargs.pop('use_gt',True)
        
#         # Subtract background from frame (returns background-subtracted frame
#         # or None, if background has not yet been established)
#         bg_sub = self.bg_engine.subtract(frame)
        
#         if bg_sub is None:
#             # If background is not established yet, just update the background
#             # using the current frame and return 
#             self.bg_engine.update_bg(frame, [])
#             return AnnotatedFrame(index=idx,
#                                   annotations=None,
#                                   image = frame,
#                                   metadata = {'background_established':False})
        
#         # If background has been established, convert input frame to proper shape
#         net_input = bg_sub[np.newaxis, :, :, np.newaxis].astype(np.float32)
        
#         # Evaluate network on the input frame
#         y_pred = self.model(net_input, training=False)[0]   # drop batch dim
        
#         # Decode the raw network outputs
#         detections = decode_predictions(
#             y_pred, self.anchors, self.img_h, self.img_w,
#             num_classes=self.num_classes,
#             conf_threshold=self.conf_threshold,
#             nms_iou_threshold=self.nms_iou_threshold,
#             max_detections=self.max_detections,
#         )
        
#         # Convert detections to pd.DataFrame   
#         bboxes = [(d.xtl, d.ytl, d.xbr, d.ybr, d.class_id, d.conf) for d in detections]

#         annot_index = pd.Index(data = range(len(bboxes)),
#                                name = 'id')
#         columns = ['xtl', 'ytl', 'xbr', 'ybr', 'label', 'score']
        
#         annot_df = pd.DataFrame(index = annot_index,
#                                 data = bboxes,
#                                 columns = columns)
        
#         # Update background     
#         self.bg_engine.update_bg(frame,
#                                  annot_df)
        
#         # Return the result
#         return AnnotatedFrame(index=idx,
#                               annotations=annot_df,
#                               image = bg_sub,
#                               metadata = {'background_established':True})
          
#     def _process_avideo(self,
#                         video:AnnotatedVideo,
#                         **kwargs):
        
#         # Check if inputs have proper type
#         if not isinstance(video,AnnotatedVideo):
#             raise TypeError('video is not type AnnotatedVideo')
                     
#         # Initialize list to store video with detection results in
#         annotated_frames = []

#         for frame in video:
            
#             # Get frame, annotations and index
#             img = frame.image
#             img = img.astype(np.uint16)

#             idx = frame.index
            
#             annotations = frame.annotations
#             bboxes = [(a.xtl, a.ytl, a.xbr, a.ybr) for _,a in annotations.iterrows()]
            
#             # Call process_frame() on frame
#             annot_frame = \
#                 self._process_frame(img,
#                                     idx,
#                                     gt_annot = annotations,
#                                     **kwargs)
            
#             # Append returned AnnotatedFrame 
#             annotated_frames.append(annot_frame)
                       
#         # In the end return an annotated video
#         return AnnotatedVideo(frames = annotated_frames)
        
#     def process_video(self, video) -> List[FrameResult]:
#         """Run .step() over every frame of an AnnotatedVideo, IN ORDER
#         (background state is sequential/stateful, so order matters --
#         same constraint as BackgroundModel itself, discussed earlier for
#         why sequences must be processed independently and in temporal
#         order rather than shuffled).

#         Ground-truth annotations on `video`'s frames are NOT used here
#         (this only produces detections, mirroring true inference-time
#         behavior with no access to ground truth) -- pair the returned
#         FrameResults with frame.annotations yourself if you want to
#         compute detection accuracy metrics.
#         """
#         results = []
#         for frame in video.frames:           
#             result = self.step(frame.image.astype(np.uint16))
#             results.append(result)
#         return results


# See test_inference.py for a runnable self-test -- it can't live in this
# file's own __main__ block, since the module-level `from background
# import BackgroundModel` above must succeed before any test-only mocking
# of hspy/hsod could run (same constraint background.py itself has).
