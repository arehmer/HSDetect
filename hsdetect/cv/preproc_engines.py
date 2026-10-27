# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 09:54:07 2023

@author: Rehmer
"""

# from .gradient import  ConvolutionFilter

import numpy as np
import cv2
import copy 
import pandas as pd 
import warnings

from hsdetect.annot.annot import AnnotatedVideo, AnnotatedFrame
from hspy.tparray import TPArray

from hsdetect import Engine

IS_ENGINE_MODULE = True

def build_preprocessing_engine(**kwargs):
    name = kwargs.pop('name')
    if name == 'PreprocEngAWA' or name == 'PreprocEngAWA_CV':
        return PreprocEngAWA(**kwargs)
    elif name == 'ZeroCorners':
        return ZeroCorners(**kwargs)
    elif name == 'BackgroundSub':
        return BackgroundSub(**kwargs)
    else:
        raise Exception('Preprocessing Engine ' + name + ' not known!')


import ast       
def get_classes():
    with open(__file__, 'r') as f:
        tree = ast.parse(f.read(), filename=__file__)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


class PreprocEngine(Engine):
    
    def __init__(self,SensorType:int):
        self.SensorType = SensorType
        self._tparray = TPArray(SensorType = self.SensorType)
    
    @property
    def SensorType(self):
        return self._SensorType
    @SensorType.setter
    def SensorType(self,SensorType:int):
        self._SensorType = SensorType
        
    def save(self):
        """
        Overrides standard save method, which only stores attributes, but 
        is not capable of storing objects which have a save() method of 
        their own.

        Returns
        -------
        None.

        """
        
        attr_dict = {}
        
        for attr_name in dir(self):
            # get the attribute *on the class*, not the instance
            attr = getattr(self.__class__, attr_name, None)
        
            # check if it's a property
            if isinstance(attr, property):
                
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
    
    def __call__(self,frames,frame_index,**kwargs):
        """ Call function only calls process. call() ist for compatibility with
        dl models, process() for compatibility with cv models. """
        

        # Create two numpy arrays for the processed frames and index filled with
        # zeros
        frame_res = np.zeros(frames.shape, frames.dtype)
        frame_index_res = np.zeros(frame_index.shape, frame_index.dtype)
        
        # Create a mapping from frame index to location index. Needed if 
        # processing engine has an internal buffer and does not return
        # frames in the same order that it obtained them
        idx_to_loc = {k:idx for k,idx in zip( range(len(frame_index)),frame_index)}
        
        for frame, idx in zip(frames,frame_index):
            
            # Pass current frame to engine, which returns a processed frame,
            # possibly not the current one, be aware of the index!
            frame_proc, idx_proc = self._process_frame(frame = frame,
                                                       idx = idx,
                                                       **kwargs)
            
            # Map idx_proc to destined location in frame_res
            k = idx_to_loc[idx_proc]
            
            # Write frame_proc and idx_proc to output arrays
            frame_res[k, ...] = frame_proc
            frame_index_res[k] = idx_proc
            
        return frame_res, frame_index_res

    
    def step(self,data:dict,**kwargs):
        return self._process_frame(frame = data['frame'],
                                   idx = data['idx'],
                                   **kwargs)
    
    def process(self,data,**kwargs):
        if isinstance(data,dict):
            return self._process_frame(frame = data['frame'],
                                       idx = data['idx'],
                                       **kwargs)
        elif isinstance(data,AnnotatedVideo):
            return self._process_avideo(data,**kwargs)
        else:
            raise NotImplementedError()
        

class PreprocEngAWA(PreprocEngine):
    
    name = 'PreprocEngAWA'
    
    def __init__(self,**attr_dict):
        
        super(PreprocEngAWA,self).__init__(attr_dict['SensorType'])
        
        self.K = attr_dict.pop('K',None)
        self.S = attr_dict.pop('S',None)
        self.a = attr_dict.pop('a',None)
        self.eps = attr_dict.pop('eps',None)
        
        

        self.cv_preproc_engine = \
            cv2.htpa.PreprocEngAWA(self.SensorType, self.K, self.S, 
                                   self.a, self.eps)
            
    @property
    def K(self):
        return self._K
    @K.setter
    def K(self,K:int):
        self._K = K
        
    @property
    def S(self):
        return self._S
    @S.setter
    def S(self,S:int):
        self._S = S
        
    @property
    def a(self):
        return self._a
    @a.setter
    def a(self,a:float):
        self._a = a
        
    @property
    def eps(self):
        return self._eps
    @eps.setter
    def eps(self,eps:float):
        self._eps = eps
        
    def __call__(self,frames,frame_index,**kwargs):
        
        frames_res, frame_index_res = \
            super(PreprocEngAWA, self).__call__(frames, frame_index, **kwargs)
            
        # Depending on the filter order, the last K frames will be missing
        # from frame_res and frame_index_res!
        # Therefore the last K frames with be zero, which is fine.
        # The last K entries in frame_index_res though must be fixed to have a
        # steadily increasing index:
        for k in range(self.K):
            frame_index_res[-k-1] =  frame_index[-k-1]
            
        return frames_res, frame_index_res

    
    def _process_frame(self,
                       frame:np.ndarray,
                       idx:int,
                       **kwargs : dict) -> dict:
                
        frame = frame.astype(np.uint16)
        
        idx_proc, frame_proc = self.cv_preproc_engine.process(frame,
                                                              idx)
        
        return frame_proc, idx_proc
        
        # # # Create an AnnotatedFrame
        # # annot_frame =  AnnotatedFrame(index = idx_filt,
        # #                               image = img_proc,
        # #                               annotations = annotations)
                
        # # Prepare dictionary for return
        # return_dict = {}    
        # return_dict.update(kwargs)          # keep passed kwargs
        # return_dict['frame'] = img_proc     # overwrite frame with result
        # return_dict['idx'] = idx_filt       # overwrite idx with result
                
        # return return_dict
    
    def _process_avideo(self,
                        video:AnnotatedVideo,
                        **kwargs):
        
        # Check if inputs have proper type
        if not isinstance(video,AnnotatedVideo):
            raise TypeError('video is not type AnnotatedVideo')

        # Initialize list to store video with detection results in
        filt_frames = {}

        for frame in video:
            
            # Get frame and index
            img = frame.image
            img = img.astype(np.uint16)

            idx = frame.index
            
            # Call process_frame() on frame
            return_dict = self._process_frame(img,
                                              idx)
            
            # Add annotated frame to dictionary by using index of processed
            # image as key. this will overwrite existing entries and keep 
            # every entry at its original position
            filt_frames[return_dict['idx']] = return_dict['frame']
                        
        # After processing a whole video, reset the internal filter buffer
        # by creating the C++ object completely new
        self.reset()
        
        # Create a new AnnotatedVideo with the filtered frames (and the 
        # annotations of the original AnnotatedVideo)
        filtered_aframes = []
        for idx,np_frame in filt_frames.items():
            aframe = AnnotatedFrame(idx,
                                    np_frame,
                                    video[idx].annotations)
            
            filtered_aframes.append(aframe)
        
        # In the end return an annotated video
        return AnnotatedVideo(filtered_aframes)
    

    
    def reset(self):
        """
        
        """
                
        self.cv_preproc_engine = \
            cv2.htpa.PreprocEngAWA(self.SensorType, self.K, self.S, 
                                   self.a, self.eps)
 
class BackgroundSub(PreprocEngine):
    """Maintains and updates a running background image, and performs
    background subtraction on incoming frames.

    Parameters
    ----------
    SensorType : int
        Encodes the type of HTPA.
    decay : float
        Exponential-moving-average decay factor used in the background
        update (paper uses decay = 0.99).        
    thresh_update : int
        Threshold applied when updating the background in update_bg(). Pixels 
        in the bakground subtracted image that exceed this threshold 
        are NOT included in updating the background, even if they are not 
        overlapping with a detection.
    N_init : int
        Number of consecutive images needed to establish the background image 
        bg. Before the background image bg is established, subtract() will 
        return None instead of a background subtracted frame.
    bg_mask : np.ndarray
        Mask [0,255] with same dimension of sensor frame. Marks pixels for
        which the background has already been established (=255) and for which
        pixels no background estimate exists yet (=0)
    box_pad : int
        Pixels which overlap with an existing detection are excluded when
        updating the background. As a precausion, the bounding boxes of 
        detections are padded in every direction by box_pad pixels.
    """
    
    name = 'BackgroundSub'
    
    def __init__(self,SensorType:int,**attr_dict):
        
        self.SensorType = SensorType
        
        self.decay = attr_dict.pop('decay',0)
        self.thresh_update = attr_dict.pop('thresh_update',1)
        self.N_init = attr_dict.pop('N_init',1)
        self.bg = attr_dict.pop('bg',None)
        self.bg_mask = attr_dict.pop('bg_mask',None)
        self.box_pad = attr_dict.pop('box_pad',0)
        
        # NOTE: previously `htpa = TPArray(...)` was a local variable and
        # discarded -- update_bg() needs the sensor's (w, h), so this is
        # now stored on self. Assumes TPArray exposes `.w`/`.h` attributes,
        # matching the C++ `_tparray.w` / `_tparray.h` used in BackSub::update_bg.
        self.htpa = TPArray(SensorType=SensorType)

        # Internal bookkeeping state mirroring the C++ BackSub class's
        # private `_n_init` / `_bg_valid` members, needed by update_bg().
        # Not exposed as properties (pure internal state, not user config).
        self._n_init = 0
        self._bg_valid = False

    
    @property
    def SensorType(self):
        return self._SensorType
    
    @SensorType.setter
    def SensorType(self,SensorType:int):
        self._SensorType = SensorType

    @property
    def decay(self):
        return self._decay
    
    @decay.setter
    def decay(self,decay:int):
        self._decay = decay     

    @property
    def thresh_update(self):
        return self._thresh_update
    
    @thresh_update.setter
    def thresh_update(self,thresh_update:int):
        self._thresh_update = thresh_update  

    @property
    def N_init(self):
        return self._N_init
    
    @N_init.setter
    def N_init(self,N_init:int):
        self._N_init = N_init     

    @property
    def bg(self):
        return self._bg
    
    @bg.setter
    def bg(self,bg:int):
        self._bg = bg 
        
    @property
    def bg_mask(self):
        return self._bg_mask
    
    @bg_mask.setter
    def bg_mask(self,bg_mask:int):
        self._bg_mask = bg_mask    

    @property
    def box_pad(self):
        return self._box_pad
    
    @box_pad.setter
    def box_pad(self,box_pad:int):
        self._box_pad = box_pad       

    # ---- Step 1: background construction -------------------------------    
    def step(self,data:dict,**kwargs):
        return self._process_frame(frame = data['frame'],
                                   idx = data['idx'],
                                   **kwargs)
    
    def process(self,data,**kwargs):
        if isinstance(data,dict):
            return self._process_frame(frame = data['frame'],
                                       idx = data['idx'],
                                       **kwargs)
        if isinstance(data,AnnotatedVideo):
            return self._process_avideo(data,**kwargs)
        elif isinstance(data,AnnotatedFrame):
            return self._process_aframe(data,**kwargs)
        else:
            raise NotImplementedError()


    # def _normalize_detections(self, detections) -> List[BoundingBox]:
    #     """Accept either a pandas DataFrame with xtl/ytl/xbr/ybr columns
    #     (as passed by _process_avideo(), i.e. frame.annotations) or a
    #     list/iterable of (xtl, ytl, xbr, ybr) tuples (matching this
    #     module's BoundingBox type and the _enlarge_boxes/_boxes_to_mask
    #     helpers), and normalize to a plain list of tuples.
    #     """
    #     if hasattr(detections, "columns"):  # duck-type pandas.DataFrame
    #         cols = ["xtl", "ytl", "xbr", "ybr"]
    #         return list(detections[cols].itertuples(index=False, name=None))
    #     return list(detections)
    

            
    def update_bg(self, 
                  current_frame: np.ndarray, 
                  detections : pd.DataFrame) -> None:
        """Update the running background estimate using the current frame
        and its associated detections (or ground-truth boxes).

        This is a line-by-line port of the reference C++ implementation
        (BackSub::update_bg), reproducing its exact behavior:

          1. Build a detection mask: 255 for pixels overlapping any
             (box_pad-enlarged) detection box, 0 otherwise. Enlarged boxes
             are clipped to [0, w-1] / [0, h-1] -- NOTE: this reproduces
             the C++ code's clipping exactly, including its off-by-one
             behavior (clipping to w-1/h-1 rather than w/h means the very
             last row/column is never fully covered by a clipped box).
          2. First call ever (no background established yet): bg is set
             directly to the current frame, bg_mask to the negated
             detection mask, and the method returns early -- no EMA
             blending happens on this call.
          3. Subsequent calls: pixels whose absolute difference from the
             current background estimate exceeds `thresh_update` are
             ALSO excluded from the update (in addition to detected
             pixels) -- this guards against undetected foreground.
          4. bg_mask accumulates (bitwise OR) which pixels have ever been
             validly observed -- once established, a pixel stays marked
             established.
          5. A background candidate is built: excluded pixels keep the
             old background value, valid pixels take the current frame's
             value (matches Algorithm 1's B_hat construction, extended
             with the diff-threshold exclusion).
          6. The background is updated via EMA:
             bg = decay*bg + (1-decay)*bg_candidate, rounded and clipped
             to uint16 (matching cv::Mat::convertTo's saturate_cast
             rounding behavior).
          7. `_n_init`/`_bg_valid` bookkeeping is updated exactly as in
             the C++ version. NOTE: with N_init=1, this logic (faithfully
             reproduced from the C++) never sets `_bg_valid=True` -- the
             validity condition can only trigger from the second call
             onward. Flagging this in case it's not the intended
             behavior for your use case.

        Parameters
        ----------
        current_frame : np.ndarray, dtype uint16, shape (h, w)
            Raw current frame (NOT background-subtracted).
        detections : pandas.DataFrame (xtl/ytl/xbr/ybr columns) or
                     list of (xtl, ytl, xbr, ybr) tuples
            Detections or ground-truth boxes for this frame, used to
            exclude foreground pixels from the background update.
        """
        if current_frame is None:
            return
        
        if not isinstance(current_frame, np.ndarray):
            raise TypeError(f'Frame is type {type(current_frame)} instead of {np.ndarray}.')
        if current_frame.dtype != np.uint16:
            current_frame = current_frame.astype(np.uint16)

        h, w = self.htpa.height, self.htpa.width
        # boxes = self._normalize_detections(detections)

        # ---- 2a. Build the detection mask (255 = overlaps a detection) ----
        det_mask = np.zeros((h, w), dtype=np.uint8)
        for idx, detection in detections.iterrows():
            xtl, ytl, xbr, ybr = detection[['xtl','ytl','xbr','ybr']]
            # Faithful port of the C++ clipping: max(0, ... - pad) and
            # min(w-1, ... + pad) / min(h-1, ... + pad) -- NOT min(w, ...).
            xtl_c = max(0, int(xtl) - self.box_pad)
            ytl_c = max(0, int(ytl) - self.box_pad)
            xbr_c = min(w - 1, int(xbr) + self.box_pad)
            ybr_c = min(h - 1, int(ybr) + self.box_pad)
            det_mask[ytl_c:ybr_c, xtl_c:xbr_c] = 255

        # ---- Exception: first call ever -- no valid background yet ----
        if not self._bg_valid and self._n_init == 0:
            self.bg = current_frame.copy()
            self.bg_mask = np.bitwise_not(det_mask)
            self._n_init = 1
            return

        # ---- 2b. Exclude pixels whose diff from the current bg exceeds
        #      thresh_update, in addition to detected pixels ----
        abs_diff = np.abs(current_frame.astype(np.int32) - self.bg.astype(np.int32))
        diff_mask = np.where(abs_diff > self.thresh_update, np.uint8(255), np.uint8(0))
        det_mask = np.bitwise_or(det_mask, diff_mask)

        # ---- 3. Accumulate which pixels have ever been validly observed ----
        self.bg_mask = np.bitwise_or(self.bg_mask, np.bitwise_not(det_mask))

        # ---- 4. Background candidate: excluded pixels keep old bg,
        #      valid (det_mask == 0) pixels take the current frame ----
        excluded = det_mask.astype(bool)
        bg_cand = np.where(excluded, self.bg, current_frame)

        # ---- EMA update, with saturate_cast-style round + clip to uint16 ----
        bg_float = self.decay * self.bg.astype(np.float32) + (1.0 - self.decay) * bg_cand.astype(np.float32)
        self.bg = np.clip(np.round(bg_float), 0, 65535).astype(np.uint16)

        # ---- n_init / bg_valid bookkeeping (exact port, two sequential ifs) ----
        if not self._bg_valid and self._n_init < self.N_init - 1:
            self._n_init += 1
        if not self._bg_valid and self._n_init == self.N_init - 1:
            self._bg_valid = True
        
        return

    # ---- Step 2: background subtraction (network input) ----------------

    def subtract(self, current_frame: np.ndarray) -> np.ndarray:
        """Produce the background-subtracted frame that gets fed to the
        detector: I0 - B. This is the actual model input.
        """
        
        # Check input type
        if not isinstance(current_frame,np.ndarray):
            raise TypeError(f'Frame is type {type(current_frame)} instead of {np.ndarray}.')
        
        if current_frame.dtype != np.uint16:
            raise TypeError(f'Frame is dtype {current_frame.dtype} instead of {np.uint16}.')
        
        # Check if background has been established (per N_init, not just
        # "has bg ever been assigned" -- bg is set on the very first
        # update_bg() call, before N_init images have actually been seen).
        if not self._bg_valid:
            warnings.warn(f'Background not established, returning zeros')
            return np.zeros(current_frame.shape,
                            current_frame.dtype)
        
        # Subtract
        frame_bgsub = current_frame.astype(np.float32) - self.bg
        
        # Return
        return abs(frame_bgsub).astype(np.uint16)
    
    
    def reset(self):
        """
        Resets bg, bg_mask, _n_init and _bg_valid to initial values
        .
        Returns
        -------
        None.

        """
        
        self.bg = None
        self.bg_mask = None
        self._n_init = 0
        self._bg_valid = False
     
    def _process_frame(self,
                       frame:np.ndarray,
                       idx:int,
                       detections : pd.DataFrame = pd.DataFrame(),
                       **kwargs:dict) -> dict:    
        
        
        # Subtract background from current frame
        backsub_frame = self.subtract(frame)
        
        # Update background
        self.update_bg(frame,
                        detections)
        
        # Prepare dictionary for return
        # return_dict = {}    
        # return_dict.update(kwargs)                # keep passed kwargs
        # return_dict['frame'] = backsub_frame      # overwrite frame with result
        # return_dict['idx'] = idx                  # overwrite frame with result
        # return_dict['bg'] = self.bg               # overwrite frame with result
        # return_dict['bg_mask'] = self.bg_mask     # overwrite frame with result        
                
        return backsub_frame, idx
        
    def _process_aframe(self,
                       frame : AnnotatedFrame,
                       **kwargs : dict) -> dict:
        
        # Copy given AnnotatedFrame to not change original
        frame_cp = copy.deepcopy(frame)
        
        # Get image from AnnotatedFrame
        frame_img = frame_cp.image
        
        if not frame_img.dtype is np.uint16:
            frame_img = frame_img.astype(np.uint16)
        
        backsub_frame = self.subtract(frame_img)
        
        # If the background has not been established yet, subtract() returns
        # None. Handle that gracefully
        if backsub_frame is None:
            backsub_frame = np.zeros((self.htpa.height,self.htpa.width),
                                     np.uint16)
            frame_cp.metadata = {}
            frame_cp.metadata['no_cv'] = True
        else:
            frame_cp.metadata = {}
            frame_cp.metadata['no_cv'] = False
        
        frame_cp.image = backsub_frame
        
        # Prepare dictionary for return
        return_dict = {'bg' : self.bg,
                       'bg_mask' : self.bg_mask,
                       'annot_frame': frame_cp}
        
        return return_dict  
    
    
    def _process_avideo(self,
                        video:AnnotatedVideo,
                        **kwargs):
        
        # Check if inputs have proper type
        if not isinstance(video,AnnotatedVideo):
            raise TypeError('video is not type AnnotatedVideo')

        # Get keyword arguments
        use_gt = kwargs.pop('use_gt',True)
        
        # # Copy the original video. Returned video will contain background-
        # # subracted images 
        # video_copy = copy.deepcopy(video)        
        
        # Initialize list to store video with detection results in
        bg_sub_frames = []

        for frame in video:
            
            # Call process_frame() on frame
            return_dict = \
                self._process_aframe(frame)
            
            # Append returned AnnotatedFrame 
            bg_sub_frames.append(return_dict['annot_frame'])
            
            # Use the given (ground truth) annotations for background updating
            if use_gt:
                detections = return_dict['annot_frame'].annotations
            # Use the proposals for background updating
            else:
                ValueError('Set "use_gt=True" to use annotations of ' +\
                           'AnnotatedVideo for background updating.')
                
            self.update_bg(frame.image,
                           detections)
            
        # Reset memory components of BackgroundEngine, otherwise background and
        # background mask will carry over to next video
        self.reset()
        
        # Deepcopy the original video to carry-over metadata to the background
        # subtracted video as well
        video_bg = copy.deepcopy(video)
        
        # Replace only the frames with the background subtracted frames
        video_bg.frames = bg_sub_frames
        
        # In the end return an annotated video
        return video_bg
    
class ZeroCorners(PreprocEngine):
        
    name = 'ZeroCorners'
        
    def __init__(self,**attr_dict):
        
        super().__init__(attr_dict['SensorType'])
               
        self._tparray = TPArray(SensorType = self.SensorType)
        
            
    def _process_frame(self,
                       frame:np.ndarray,
                       idx:int,
                       **kwargs) -> dict:
        
        if frame is None:
            return_dict = {}
            return_dict.update(kwargs)          # keep passed kwargs
            return_dict['frame'] = frame
            return_dict['idx'] = idx
            
            return return_dict
        
        frame = frame.astype(np.uint16)
        
        # Obtain detection mask [0,1] and convert to boolean
        detection_mask = self._tparray._detection_mask.astype(bool)
        
        # Set corners (detection_mask = 0) to zero
        frame[~detection_mask] = 0
        
        # Create an AnnotatedFrame
        # annot_frame =  AnnotatedFrame(index = idx,
        #                               image = frame,
        #                               annotations = annotations)
                
        # Prepare dictionary for return
        return_dict = {}    
        return_dict.update(kwargs)          # keep passed kwargs
        return_dict['frame'] = frame
        return_dict['idx'] = idx
                
        return return_dict
    
    def _process_avideo(self,
                        video:AnnotatedVideo,
                        **kwargs):
        
        # Check if inputs have proper type
        if not isinstance(video,AnnotatedVideo):
            raise TypeError('video is not type AnnotatedVideo')

        # Initialize list to store video with detection results in
        zeroed_frames = {}

        for frame in video:
            
            # Get frame and index
            img = frame.image
            img = img.astype(np.uint16)

            idx = frame.index
            
            # Call process_frame() on frame
            return_dict = self._process_frame(img,
                                              idx)
            
            # Add annotated frame to dictionary by using index of processed
            # image as key. this will overwrite existing entries and keep 
            # every entry at its original position
            zeroed_frames[return_dict['idx']] = return_dict['frame']
                        
        # After processing a whole video, reset the internal filter buffer
        # by creating the C++ object completely new
        self.reset()
        
        # Create a new AnnotatedVideo with the filtered frames (and the 
        # annotations of the original AnnotatedVideo)
        zeroed_aframes = []
        for idx,np_frame in zeroed_frames.items():
            aframe = AnnotatedFrame(idx,
                                    np_frame,
                                    video[idx].annotations)
            
            zeroed_aframes.append(aframe)
        
        # In the end return an annotated video
        return AnnotatedVideo(zeroed_aframes)
        
        
    def reset(self):
        """
        ZeroCorners has no memory components to reset      
        """        
        return None
    
    # def process(self,**kwargs)->cv2.htpa.DataFrame:
    #     '''
    #     Applies pre_process() method of Cpp class PreprocEngAWA

    #     Parameters
    #     ----------
    #     frame : np.ndarray
    #         Frame to apply proposal generator to.
    #     idx : int
    #         Index of that frame.
    #     video: pd.DataFrame
    #         Video or sequence of frames as pd.DataFrame to apply proposal
    #         generator to

    #     Returns
    #     -------
    #     bboxes : TYPE
    #         DESCRIPTION.

    #     '''

    #     img = kwargs.pop('frame',None)
    #     img_idx = kwargs.pop('idx',None)
    #     video = kwargs.pop('video',None)
        
    #     # Check if single frame or video is provided
    #     if isinstance(img,np.ndarray) and isinstance(img_idx,int):
            
    #         # Cast image to proper type just to be sure
    #         img = img.astype(np.uint16)
            
    #         # Apply preprocessing engine to the single given frame
    #         idx_filt, img_proc = self.cv_preproc_engine.process(img, img_idx)
            
    #         return img_proc,idx_filt
                        
            
    #     elif isinstance(video,pd.DataFrame):
    #         # Copy video to not overwrite the original
    #         video_proc = video.copy()
            
    #         # If a video is provided, loop over the whole video and apply
    #         # preprocessing engine frame-wise
    #         for idx in video.index:
    #             img = video.loc[idx,self._tparray._pix].values
    #             img = img.reshape(self._tparray._npsize).astype(np.uint16)
        
    #             # Call the get_proposal() method of the Cpp class
    #             idx_filt, img_proc = self.cv_preproc_engine.process(img,
    #                                                                 idx)
                
    #             video_proc.loc[idx_filt,self._tparray._pix] = img_proc.flatten()
            
    #         # After processing a whole video, reset the internal filter buffer
    #         # by creating the C++ object completely new
    #         self.reset()
            
    #         return video_proc
        
    #     elif isinstance(video,AnnotatedVideo):
            
    #         # Initialize list to store pre-preocessed video in
    #         video_proc = {}
            
    #         # Loop over all frames
    #         for frame in video:
                
    #             img = frame.image
    #             annot = frame.annotations.copy()
    #             idx = frame.index
                
    #             # Apply OpenCV instance 
    #             idx_filt, img_proc = self.cv_preproc_engine.process(img,
    #                                                                 idx)
                
    #             frame_proc = AnnotatedFrame(idx_filt,
    #                                         img_proc,
    #                                         annot)
                
    #             # Append processed frame to dictionary. That way frames which
    #             # are filtered later due to filter buffer delays are automatically
    #             # overwritten with the newest version
    #             video_proc[idx_filt] = frame_proc
                
    #         # Unpack all annotated Frames into a list
    #         video_proc_list = [video_proc[k] for k in video_proc.keys()]
            
    #         # Make an AnnotatedVideo from that list
    #         avideo = AnnotatedVideo()
    #         avideo.from_AnnotatedFrames(video_proc_list)
            
    #         # After processing a whole video, reset the internal filter buffer
    #         # by creating the C++ object completely new
    #         self.reset()
            
    #         return avideo


        