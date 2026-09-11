# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 09:54:07 2023

@author: Rehmer
"""

# from .gradient import  ConvolutionFilter

import numpy as np
import cv2
import copy 

from hsdetect.annot.annot import AnnotatedVideo, AnnotatedFrame
from hspy.tparray import TPArray

def build_preprocessing_engine(**kwargs):
    name = kwargs.pop('name')
    if name == 'PreprocEngAWA' or name == 'PreprocEngAWA_CV':
        return PreprocEngAWA(**kwargs)
    elif name == 'ZeroCorners':
        return ZeroCorners(**kwargs)
    else:
        raise Exception('Preprocessing Engine ' + name + ' not known!')

import ast       
def get_classes():
    with open(__file__, 'r') as f:
        tree = ast.parse(f.read(), filename=__file__)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

class PreprocEngine():
    
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

    
    def _process_frame(self,
                       frame:np.ndarray,
                       idx:int,
                       annotations = None) -> dict:
                
        frame = frame.astype(np.uint16)
        
        idx_filt, img_proc = self.cv_preproc_engine.process(frame,
                                                            idx)
        
        # Create an AnnotatedFrame
        annot_frame =  AnnotatedFrame(index = idx_filt,
                                      image = img_proc,
                                      annotations = annotations)
                
        # Prepare dictionary for return
        return_dict = {'frame_pre':img_proc,
                       'idx_pre':idx_filt,
                       'annot_frame':  annot_frame}
        
        return return_dict
    
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
                                              idx,
                                              frame.annotations)
            
            # Add annotated frame to dictionary by using index of processed
            # image as key. this will overwrite existing entries and keep 
            # every entry at its original position
            filt_frames[return_dict['idx_pre']] = return_dict['annot_frame']
                        
        # After processing a whole video, reset the internal filter buffer
        # by creating the C++ object completely new
        self.reset()
        
                
        # Deepcopy the original video to carry-over metadata to the background
        # subtracted video as well
        video_filt = copy.deepcopy(video)
        
        # Replace only the frames with the background subtracted frames
        video_filt.frames = list(filt_frames.values())
        
        # In the end return an annotated video
        return video_filt
    

    
    def reset(self):
        """
        
        """
                
        self.cv_preproc_engine = \
            cv2.htpa.PreprocEngAWA(self.SensorType, self.K, self.S, 
                                   self.a, self.eps)
 
    
class ZeroCorners(PreprocEngine):
        
    name = 'ZeroCorners'
        
    def __init__(self,**attr_dict):
        
        super().__init__(attr_dict['SensorType'])
               
        self._tparray = TPArray(SensorType = self.SensorType)
        
            
    def _process_frame(self,
                       frame:np.ndarray,
                       idx:int,
                       annotations = None) -> dict:
                
        frame = frame.astype(np.uint16)
        
        # Obtain detection mask [0,1] and convert to boolean
        detection_mask = self._tparray._detection_mask.astype(bool)
        
        # Set corners (detection_mask = 0) to zero
        frame[~detection_mask] = 0
        
        # Create an AnnotatedFrame
        annot_frame =  AnnotatedFrame(index = idx,
                                      image = frame,
                                      annotations = annotations)
                
        # Prepare dictionary for return
        return_dict = {'frame_pre':None,
                       'idx_pre':idx,
                       'annot_frame':  annot_frame}
        
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
                                              idx,
                                              frame.annotations)
            
            # Add annotated frame to dictionary by using index of processed
            # image as key. this will overwrite existing entries and keep 
            # every entry at its original position
            zeroed_frames[return_dict['idx_pre']] = return_dict['annot_frame']
                        
        # After processing a whole video, reset the internal filter buffer
        # by creating the C++ object completely new
        self.reset()
        
        # Deepcopy the original video to carry-over metadata to the background
        # subtracted video as well
        video_zeroed = copy.deepcopy(video)
        
        # Replace only the frames with the background subtracted frames
        video_zeroed.frames = list(zeroed_frames.values())
        
        # In the end return an annotated video
        return video_zeroed
        
        
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


        