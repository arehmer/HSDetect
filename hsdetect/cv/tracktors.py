# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:30:01 2023

@author: rehmer
"""
import time
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import json 
import time

import hsdetect
from hsdetect.cv.detectors import Detector,build_detector
from hsdetect.cv.trackers import Tracker,build_tracker
from . import misc
from hsdetect.annot.annot import AnnotatedFrame, AnnotatedVideo



def build_tracktor(**kwargs):
    name = kwargs.pop('name')
    if name == 'HSv00':
        return HSv00(**kwargs)
    if name == 'HSv01':
        return HSv01(**kwargs)
    if name == 'DummyTracktor':
        return DummyTracktor(**kwargs)

import ast       
def get_classes():
    with open(__file__, 'r') as f:
        tree = ast.parse(f.read(), filename=__file__)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

class Tracktor():
    
    def __init__(self,
                 SensorType:int,
                 **attr_dict:dict):
        
        self.SensorType = SensorType  

    @property
    def SensorType(self):
        return self._SensorType
    
    @SensorType.setter
    def SensorType(self,SensorType:int):
        self._SensorType = SensorType
        
        
    # def track(self,**kwargs):
        
    #     """
    #     Applys tracktor either frame by frame or loops over a whole sequence
    #     that is given as a pd.DataFrame.

    #     Parameters
    #     ----------
    #     img : np.ndarray
    #         Frame to apply tracktor to.
    #     img_idx : int
    #         Index of that frame.
    #     video: pd.DataFrame
    #         Video or sequence of frames as pd.DataFrame to apply tracktor to

    #     Returns
    #     -------
    #     None.

    #     """
        
    #     img = kwargs.pop('img',None)
    #     img_idx = kwargs.pop('img_idx',None)
    #     video = kwargs.pop('video',None)
        
        
    #     # Check if single frame or video is provided
    #     if isinstance(img,np.ndarray) and isinstance(img_idx,int):
        
    #         # Cast image to proper type
    #         img = img.astype(np.uint16)
        
    #         # Apply proposal generator to the single given frame
    #         det_cv, img_pre  = self.cv_tracktor.track(img,img_idx)

    #         # Convert to pd.DataFrame
    #         det = misc.cv2_dfTrack_to_pd_DataFrame(det_cv)
                        
    #         # Name index "id" for compatiblity with all other classes
    #         det.index.rename('id',inplace=True)
            
    #         # Filter for positive detections
    #         det = det.loc[det['label']==1]
            
                      
    #         # Convert to AnnotatedFrame type
    #         return AnnotatedFrame(index = img_idx,
    #                               image = img_pre,
    #                               annotations = det)

        
    #     elif isinstance(video,pd.DataFrame):
    #         # If a video is provided, loop over the whole video and apply
    #         # tracktor frame-wise
            
    #         # Initialize a list in which to store the tracks for each frame
    #         bboxes = []
            
    #         # Pointer to tparray attribute of detector
    #         tparray = self.detector.tparray
            
    #         imgs = []
    #         idxs = []
            
    #         # Convert all frames to the proper shape
    #         for idx in video.index:
                        
    #             img = video.loc[idx,tparray._pix].values
    #             img = img.reshape(tparray._npsize).astype(np.uint16)
        
        
    #             imgs.append(img)
    #             idxs.append(idx)
            
    #         i = 0

    #         for idx in idxs:
                
    #             # Call the track() method of the Cpp class
    #             time.sleep(0.01)
    #             df_bboxes, _ = self.cv_tracktor.track(imgs[i],idx)
    #             time.sleep(0.01)
    #             i = i+1
    #             # Convert to pd.DataFrame
    #             df_bboxes = pd.DataFrame(data = df_bboxes.get_content(),
    #                                       columns = df_bboxes.get_headers())
                                
    #             # Append found boxes, if any                
    #             if len(df_bboxes)>0:
    #                 bboxes.append(df_bboxes)

    #         # Concatenate all DataFrames, if any
    #         if len(bboxes)>0:
    #             bboxes = pd.concat(bboxes)
    #         else:
    #         # Else set bboxes to an empty DataFrame
    #             bboxes = df_bboxes
            
    #         # Cast datatypes
    #         bboxes = bboxes.astype(self.detector.cv_datatypes)
            
    #         # Set original index back
    #         bboxes = bboxes.set_index('idx',drop=True)
            
    #         # Tracktor sets the index equal to the tracks, which makes sense
    #         # from a frame-wise point of view, but not in a video since indices
    #         # will not be unique anymore
    #         bboxes = bboxes.reset_index(drop=True)

    #         # Name index "id" for compatiblity with all other classes
    #         bboxes.index.rename('id',inplace=True)
            

    #         return bboxes
           

    def process(self,data,**kwargs):
        if isinstance(data,dict):
            return self._process_frame(frame = data['frame'],
                                       idx = data['idx'],
                                       **kwargs)
        elif isinstance(data,AnnotatedVideo):
            return self._process_avideo(data,**kwargs)
        else:
            raise NotImplementedError()
            
            
    def _process_frame(self,
                        frame:np.ndarray,
                        idx:int,
                        **kwargs:dict) -> dict:
        
        # Get keyword arguments
        gt_annot = kwargs.pop('gt_annot',None)
        update_bg = kwargs.pop('update_bg',True)
        use_gt = kwargs.pop('use_gt',False)
        
        frame = frame.astype(np.uint16)
        
        det_cv, frame_pre = self.cv_tracktor.process(frame,
                                                     idx)
        
        # Convert to pd.DataFrame
        det = misc.cv2_dfTrack_to_pd_DataFrame(det_cv)
                    
        # Name index "id" for compatiblity with all other classes
        det.index.rename('id',inplace=True)
        
        # Create an AnnotatedFrame from the detection result
        annot_frame =  AnnotatedFrame(index = idx,
                                      image = frame_pre,
                                      annotations = det)
        
        # Prepare dictionary for return
        return_dict = {'frame_pre':frame_pre,
                       'idx_pre':None,
                       'annot_frame':  annot_frame}
        
        # Add background to return dict if BackSub is applied
        proposal_engine = self.detector.proposal_engine
        if isinstance(proposal_engine, hsdetect.cv.proposal_engines.BackSub):
            return_dict['bg'] = \
                proposal_engine.cv_proposal_engine._bg
            return_dict['bg_mask'] = \
                proposal_engine.cv_proposal_engine._bg_mask
                
        # If the ProposalEngine uses background subtraction, update the 
        # background
        proposal_engine = self.detector.proposal_engine
        
        if isinstance(proposal_engine, hsdetect.cv.proposal_engines.BackSub):
            if update_bg:
                # Use the given annotations background updating
                if use_gt:
                    detections = gt_annot
                # Use the predictions for background updating
                else:
                    detections = return_dict['annot_frame'].annotations
                    
                proposal_engine.update_bg(return_dict['frame_pre'],
                                          detections)
        
        return return_dict 
            
    def _process_avideo(self,
                        video:AnnotatedVideo,
                        **kwargs):
        
        # Check if inputs have proper type
        if not isinstance(video,AnnotatedVideo):
            raise TypeError('video is not type AnnotatedVideo')

        # Initialize list to store video with detection results in
        annotated_frames = []

        for frame in video:
            
            # Get frame and index
            img = frame.image
            img = img.astype(np.uint16)

            idx = frame.index
                        
            return_dict = self._process_frame(img,
                                              idx,
                                              gt_annot = frame.annotations,
                                              **kwargs)
            
            # Append returned AnnotatedFrame 
            annotated_frames.append(return_dict['annot_frame'])
                       
        # In the end return an annotated video
        return AnnotatedVideo(frames = annotated_frames)
        
      
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

        

class HSv00(Tracktor):
    
    name = 'HSv00'
    
    def __init__(self,**attr_dict):
        
        super(HSv00,self).__init__(attr_dict['SensorType'])
        
        self.detector = attr_dict.pop('detector',None)
        self.tracker = attr_dict.pop('tracker',None)
        self.s_active = attr_dict.pop('s_active',0)                             # minimal score an active track needs to have to not get a strike
        self.s_init = attr_dict.pop('s_init',0.5)                               # minimal score a detection needs to have to initialize a track
        self.IoU_active = attr_dict.pop('IoU_active',0.7)                       # minimal IoU between a new detection and an active track to be considered corresponding
        self.IoU_init = attr_dict.pop('IoU_init',0.1)                           # maximal IoU between a new detection and any active track for the detection to consitute a new track
        self.strikes_lim = attr_dict.pop('strikes_lim',30)                      # maximal number of strikes (detections below score_lim_active) in a row before track is deleted
        self.score_lim_del = attr_dict.pop('score_lim_del',-1.5)                # If any track gets a score below this value, it is instantly deleted whatever the number of strikes
        self.alpha = attr_dict.pop('alpha',2)                                   #  Weight for pixels in blocks in qmodel that are outside the mask from 0, ..., 10
        
        self.initialize_cpp_instance()
        
    def initialize_cpp_instance(self):
        
        self.cv_tracktor = cv2.htpa.HSv00(self.detector.cv_detector,
                                          self.tracker.cv_tracker,
                                          self.s_active,
                                          self.s_init,
                                          self.IoU_active,
                                          self.IoU_init,
                                          self.strikes_lim,
                                          self.score_lim_del,
                                          self.alpha)
  
    
    
    @property
    def detector(self):
        return self._detector
    @detector.setter
    def detector(self,detector):
        if detector is None:
            self._preproc_engine = None
        elif isinstance(detector, dict):
            self._detector = build_detector(**detector)
        elif isinstance(detector,Detector):
            self._detector = detector
        else:
            Exception('Detector not recognized.')
    
    @property
    def tracker(self):
        return self._tracker
    @tracker.setter
    def tracker(self,tracker):
        if tracker is None:
            self._tracker = None
        elif isinstance(tracker, dict):
            self._tracker = build_tracker(**tracker)
        elif isinstance(tracker,Tracker):
            self._tracker = tracker
        else:
            Exception('Detector not recognized.')
    
    @property
    def s_active(self):
        return self._s_active
    @s_active.setter
    def s_active(self,s_active):
        self._s_active = s_active
        
    @property
    def s_init(self):
        return self._s_init
    @s_init.setter
    def s_init(self,s_init):
        self._s_init = s_init
    
    @property
    def IoU_active(self):
        return self._IoU_active
    @IoU_active.setter
    def IoU_active(self,IoU_active):
        self._IoU_active = IoU_active
    
    @property
    def IoU_init(self):
        return self._IoU_init
    @IoU_init.setter
    def IoU_init(self,IoU_init):
        self._IoU_init = IoU_init
    
    @property
    def strikes_lim(self):
        return self._strikes_lim
    @strikes_lim.setter
    def strikes_lim(self,strikes_lim):
        self._strikes_lim = strikes_lim
        
    @property
    def score_lim_del(self):
        return self._score_lim_del
    @score_lim_del.setter
    def score_lim_del(self,score_lim_del):
        self._score_lim_del = score_lim_del
    
    @property
    def alpha(self):
        return self._alpha
    @alpha.setter
    def alpha(self,alpha):
        self._alpha = alpha

    def save_json(self,path:Path):
        """
        Saves the dictionary containing all the attributes of the Tracktor as a
        JSON file.

        Parameters
        ----------
        path : Path
            Absolute path including file-name of the JSON file.

        Returns
        -------
        None.

        """
        
        attr_dict = self.save()
        
        with open(path, 'w') as fp:
            json.dump(attr_dict, fp)
            
        return None

class HSv01(Tracktor):
    
    name = 'HSv01'
    
    def __init__(self,**attr_dict):
        
        super(HSv01,self).__init__(attr_dict['SensorType'])
        
        self.detector = attr_dict.pop('detector',None)
        self.tracker = attr_dict.pop('tracker',None)
        self.s_active = attr_dict.pop('s_active',0)                             # minimal score an active track needs to have to not get a strike
        self.s_init = attr_dict.pop('s_init',0.5)                               # minimal score a detection needs to have to initialize a track
        self.N_init = attr_dict.pop('N_init',3)
        self.IoU_active = attr_dict.pop('IoU_active',0.7)                       # minimal IoU between a new detection and an active track to be considered corresponding
        self.IoU_init = attr_dict.pop('IoU_init',0.1)                           # maximal IoU between a new detection and any active track for the detection to consitute a new track
        self.IoU_redundant = attr_dict.pop('IoU_redundant',0.85)                # maximal IoU allowed between two tracks before they are determined to be tracking the same object   
        self.strikes_lim = attr_dict.pop('strikes_lim',30)                      # maximal number of strikes (detections below score_lim_active) in a row before track is deleted
        self.score_lim_del = attr_dict.pop('score_lim_del',-1.5)                # If any track gets a score below this value, it is instantly deleted whatever the number of strikes
        self.alpha = attr_dict.pop('alpha',2)                                   #  Weight for pixels in blocks in qmodel that are outside the mask from 0, ..., 10
        
        self.initialize_cpp_instance()
        
    def initialize_cpp_instance(self):
        
        self.cv_tracktor = cv2.htpa.HSv01(self.detector.cv_detector,
                                          self.tracker.cv_tracker,
                                          self.s_active,
                                          self.s_init,
                                          self.N_init,
                                          self.IoU_active,
                                          self.IoU_init,
                                          self.IoU_redundant,
                                          self.strikes_lim,
                                          self.score_lim_del,
                                          self.alpha)
        
  
    
    
    @property
    def detector(self):
        return self._detector
    @detector.setter
    def detector(self,detector):
        if detector is None:
            self._preproc_engine = None
        elif isinstance(detector, dict):
            self._detector = build_detector(**detector)
        elif isinstance(detector,Detector):
            self._detector = detector
        else:
            Exception('Detector not recognized.')
    
    @property
    def tracker(self):
        return self._tracker
    @tracker.setter
    def tracker(self,tracker):
        if tracker is None:
            self._tracker = None
        elif isinstance(tracker, dict):
            self._tracker = build_tracker(**tracker)
        elif isinstance(tracker,Tracker):
            self._tracker = tracker
        else:
            Exception('Detector not recognized.')
    
    @property
    def s_active(self):
        return self._s_active
    @s_active.setter
    def s_active(self,s_active):
        self._s_active = s_active
        
    @property
    def s_init(self):
        return self._s_init
    @s_init.setter
    def s_init(self,s_init):
        self._s_init = s_init
        
    @property
    def N_init(self):
        return self._N_init
    @N_init.setter
    def N_init(self,N_init:int):
        self._N_init = N_init

    @property
    def IoU_active(self):
        return self._IoU_active
    @IoU_active.setter
    def IoU_active(self,IoU_active):
        self._IoU_active = IoU_active
        
    @property
    def IoU_init(self):
        return self._IoU_init
    @IoU_init.setter
    def IoU_init(self,IoU_init):
        self._IoU_init = IoU_init
        
    @property
    def IoU_redundant(self):
        return self._IoU_redundant
    @IoU_redundant.setter
    def IoU_redundant(self,IoU_redundant):
        self._IoU_redundant = IoU_redundant
    
    @property
    def strikes_lim(self):
        return self._strikes_lim
    @strikes_lim.setter
    def strikes_lim(self,strikes_lim):
        self._strikes_lim = strikes_lim
        
    @property
    def score_lim_del(self):
        return self._score_lim_del
    @score_lim_del.setter
    def score_lim_del(self,score_lim_del):
        self._score_lim_del = score_lim_del
    
    @property
    def alpha(self):
        return self._alpha
    @alpha.setter
    def alpha(self,alpha):
        self._alpha = alpha

    def save_json(self,path:Path):
        """
        Saves the dictionary containing all the attributes of the Tracktor as a
        JSON file.

        Parameters
        ----------
        path : Path
            Absolute path including file-name of the JSON file.

        Returns
        -------
        None.

        """
        
        attr_dict = self.save()
        
        with open(path, 'w') as fp:
            json.dump(attr_dict, fp)
            
        return None


    
class DummyTracktor(Tracktor):
    
    name = 'DummyTracktor'
    
    
    @property
    def detector(self):
        return self._detector
    @detector.setter
    def detector(self,detector):
        if detector is None:
            self._preproc_engine = None
        elif isinstance(detector, dict):
            self._detector = build_detector(**detector)
        elif isinstance(detector,Detector):
            self._detector = detector
        else:
            Exception('Detector not recognized.')
    
    
    def __init__(self,**attr_dict):
        
        
        self.detector = attr_dict.pop('detector',None)

        self.cv_tracktor = cv2.htpa.DummyTracktor(self.detector.cv_MultiScaleDetector)
        
