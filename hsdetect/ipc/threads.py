# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 08:34:59 2023

@author: rehmer
"""

from threading import Thread, Condition
from queue import Queue
import time

from hspytools.ipc.threads_base import RWThread, RWThread_R1
from hspytools.tparray import TPArray
from hsdetect.cv.tracktors import Tracktor
from hsdetect.cv.detectors import Detector

# Only for debugging:
import numpy as np
from pathlib import Path


class ProcessingThread(RWThread_R1):
    
    def __init__(self,
                 name : str,
                 detector:Detector,
                 write_buffer:Queue,
                 read_buffer:Queue,
                 **kwargs):
        
        self.detector = detector
        
        self.tparray = TPArray(SensorType = detector.SensorType)
        
        self.num_pix = len(self.tparray._pix)
        self.t0 = time.time()
        
        
        # Call parent class
        super().__init__(name = name,
                         read_buffer = read_buffer,
                         write_buffer = write_buffer,
                         **kwargs)
        
    def _target(self):
        
        # Get result from upstream thread
        data = self.read_buffer.get()
        
        
        # Check success flag of upstream thread
        if data['success'] is True:
            
            # If not, perform tracking
            if 'pix_dK' in data.keys():
                frame = data['pix_dK']
            elif 'frame' in data.keys():
                frame = data['frame']
            
            # Reshape if not the proper size
            if frame.ndim == 1:
                frame = frame[0:self.num_pix]
                frame = frame.reshape(self.tparray._npsize)
            
            # For debugging: print image_id
            # print(result['image_id'])
            # Apply Tracktor
            try:
                data_processed  = self.detector.process({'frame' : frame,
                                                         'idx' : data['image_id']})
                data.update(data_processed)
                       
            except Exception as e:
                print(f"[{self.name}] {e}")  
                
                # Set success flag to False in case of failure
                data['success'] = False
                
        else:
            
            # If upstream thread failed, set success flag to False
            data['success'] = False
        
        # self.image_id = self.image_id + 1
        
        # Return dict
        return data

# class Tracktor_Thread_R1(RWThread_R1):
    
#     def __init__(self,
#                  name : str,
#                  tracktor:Tracktor,
#                  write_buffer:Queue,
#                  read_buffer:Queue,
#                  **kwargs):
        
#         self.tracktor = tracktor
        
#         self.tparray = TPArray(SensorType = tracktor.SensorType)
        
#         self.num_pix = len(self.tparray._pix)
#         self.t0 = time.time()
        
        
#         # Call parent class
#         super().__init__(name = name,
#                          read_buffer = read_buffer,
#                          write_buffer = write_buffer,
#                          **kwargs)
        
#     def _target(self,data:dict):
        
#         # print('Executed tracktor thread: ' + str(time.time()-self.t0) )
        
               
#         # Check success flag of upstream thread
#         if data['success'] is True:
            
#             # If not, perform tracking
#             frame = data['pix_dK']
            
#             # Reshape if not the proper size
#             if frame.ndim == 1:
#                 frame = frame[0:self.num_pix]
#                 frame = frame.reshape(self.tparray._npsize)
            
#             # For debugging: print image_id
#             # print(result['image_id'])
#             # Apply Tracktor
#             try:
                
#                 annot_frame  = self.tracktor.track(img_idx = data['image_id'],
#                                                      img = frame)
                           
#                 data['frame_proc'] = annot_frame.image
#                 data['bboxes'] = annot_frame.annotations
                
                
#                 if len(data['bboxes'])!=0:
#                     print(data['bboxes'][['xtl','ytl','w','h','score','confirmed','image_id']])
                        
#             except:
#                 print("Exception in Tracktor.")
                
#                 # Set success flag to False in case of failure
#                 data['success'] = False
                
#                 # np.save(Path.cwd() / 'debug' / ('frame_'+str(result['image_id'])),frame)
#         else:
            
#             # If upstream thread failed, set success flag to False
#             data['success'] = False
        
#         # self.image_id = self.image_id + 1
        
#         # Return dict of frame and bounding boxes
#         return data


# class Tracktor_Thread(RWThread):
    
#     def __init__(self,
#                  ArrayType,
#                  tracktor:Tracktor,
#                  write_buffer:Queue,
#                  write_condition:Condition,
#                  read_buffer:Queue,
#                  read_condition:Condition,
#                  **kwargs):
        
#         self.tracktor = tracktor
        
#         self.tparray = TPArray(ArrayType = ArrayType)
        
#         self.num_pix = len(self.tparray._pix)
#         self.t0 = time.time()
        
        
#         # Call parent class
#         super().__init__(target = self._target_function,
#                          read_buffer = read_buffer,
#                          read_condition = read_condition,
#                          write_buffer = write_buffer,
#                          write_condition = write_condition,
#                          **kwargs)
        
#     def _target_function(self):
        
#         # print('Executed tracktor thread: ' + str(time.time()-self.t0) )
        
#         # Get result from upstream thread
#         result = self.read_buffer.get()
               
#         # Check success flag of upstream thread
#         if result['success'] is True:
            
#             # If not, perform tracking
#             frame = result['frame']
            
#             # Reshape if not the proper size
#             if frame.ndim == 1:
#                 frame = frame[0:self.num_pix]
#                 frame = frame.reshape(self.tparray._npsize)
            
#             # For debugging: print image_id
#             # print(result['image_id'])
#             # Apply Tracktor
#             try:
                
#                 annot_frame  = self.tracktor.track(img_idx = result['image_id'],
#                                                      img = frame)
                           
#                 result['frame_proc'] = annot_frame.image
#                 result['bboxes'] = annot_frame.annotations
                
#                 if len(result['bboxes'])!=0:
#                     print(result['bboxes'][['xtl','ytl','w','h','score','confirmed','image_id']])
                        
#             except:
#                 print("Exception in Tracktor.")
                
#                 # Set success flag to False in case of failure
#                 result['success'] = False
                
#                 # np.save(Path.cwd() / 'debug' / ('frame_'+str(result['image_id'])),frame)
#         else:
            
#             # If upstream thread failed, set success flag to False
#             result['success'] = False
        
#         # self.image_id = self.image_id + 1
        
#         # Return dict of frame and bounding boxes
#         return result
    

# class Detector_Thread(RWThread):
    
#     def __init__(self,
#                  ArrayType,
#                  detector:Detector,
#                  write_buffer:Queue,
#                  write_condition:Condition,
#                  read_buffer:Queue,
#                  read_condition:Condition,
#                  **kwargs):
        
#         self.detector = detector
        
#         self.tparray = TPArray(ArrayType = ArrayType)
        
#         self.num_pix = len(self.tparray._pix)
#         self.t0 = time.time()
        
        
#         # Call parent class
#         super().__init__(target = self._target_function,
#                          read_buffer = read_buffer,
#                          read_condition = read_condition,
#                          write_buffer = write_buffer,
#                          write_condition = write_condition,
#                          **kwargs)
        
#     def _target_function(self):
        
#         # print('Executed tracktor thread: ' + str(time.time()-self.t0) )
        
#         # Get result from upstream thread
#         result = self.read_buffer.get()
               
#         # Check success flag of upstream thread
#         if result['success'] is True:
            
#             # If not, perform tracking
#             frame = result['frame']
            
#             # Reshape if not the proper size
#             if frame.ndim == 1:
#                 frame = frame[0:self.num_pix]
#                 frame = frame.reshape(self.tparray._npsize)
            
#             # For debugging: print image_id
#             # print(result['image_id'])
#             # Apply Tracktor
#             try:
#                 bboxes, frame  = self.detector.detect(result['image_id'],
#                                                       frame)
                
#                 # Keep only boxes with label 1
#                 bboxes = bboxes.loc[bboxes['label']==1]
                
#                 result['frame_proc'] = frame
#                 result['bboxes'] = bboxes
#                 if len(bboxes)!=0:
#                     # print(bboxes)
#                     print(bboxes[['w','h','score']])
                        
#             except Exception as e:
#                 print("Exception in Detector.")
#                 print(e)
                
#                 # Set success flag to False in case of failure
#                 result['success'] = False
                
#                 # np.save(Path.cwd() / 'debug' / ('frame_'+str(result['image_id'])),frame)
#         else:
            
#             # If upstream thread failed, set success flag to False
#             result['success'] = False
        
#         # self.image_id = self.image_id + 1
        
#         # Return dict of frame and bounding boxes
#         return result


    
# class Detector_Thread(RWThread):
    
#     def __init__(self,
#                   width:int,
#                   height:int,
#                   detector:Detector,
#                   write_buffer:Queue,
#                   read_buffer:Queue,
#                   **kwargs):
        
#         self.detector = detector
        
#         self.tparray = TPArray(width, height)
        
#         self.num_pix = len(self.tparray._pix)
#         self.t0 = time.time()
        
#         self.image_id = 0
        
#         self.s_init = kwargs.pop('s_init',1)
        
#         # Call parent class
#         super().__init__(target = self._target_function,
#                           read_buffer = read_buffer,
#                           write_buffer = write_buffer,
#                           **kwargs)
        
#     def _target_function(self):
        
#         raise Exception('Target function needs to be updated, alike to Tracktor_Thread!')
        
#         # Get result from upstream thread
#         upstream_dict = self.read_buffer.get()
#         frame = upstream_dict['frame']
        
#         # Reshape if not the proper size
#         if frame.ndim == 1:
#             frame = frame[0:self.num_pix]
#             frame = frame.reshape(self.tparray._npsize)
        
#         # Apply Detector
#         try:
#             bboxes, frame = self.detector.predict(img = frame,
#                                                   img_idx = self.image_id)
#         except:
#             print("Exception in Tracktor.")
#             bboxes = self.tracktor.track_record
        
#         bboxes = bboxes.loc[bboxes['score']>self.s_init]
            
#         self.image_id = self.image_id + 1
        
#         # print('Executed detection thread: ' + str(time.time()-self.t0))
        
#         # Return dict of frame and bounding boxes
#         return {'frame':frame,'bboxes':bboxes}