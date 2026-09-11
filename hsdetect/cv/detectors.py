import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from pathlib import Path

# from sklearn.model_selection import cross_val_score

# from hspytools.tparray import TPArray

from collections import defaultdict


from hspytools.cv.seg import OtsuSeg

from hsdetect.cv.misc import IoU 
from . import misc



from hspy.tparray import TPArray

import hsdetect

from hsdetect.cv.preproc_engines import PreprocEngine,build_preprocessing_engine
from hsdetect.cv.postproc_engines import PostprocEngine,build_postprocessing_engine
from hsdetect.cv.proposal_engines import ProposalEngine,build_proposal_engine, BackSub
from hsdetect.cv.feature_engines import FeatureEngine,build_feature_engine
from hsdetect.cv.classifier_engine import ClassifierEngine,build_classifier_engine

from hsdetect.annot.annot import AnnotatedFrame,AnnotatedVideo


def build_detector(**kwargs):
    name = kwargs.pop('name')
    if name == 'SubDetector':
        return SubDetector(**kwargs)
    elif name == 'OneScaleDetector':
        return OneScaleDetector(**kwargs)
    if name == 'TwoScaleDetector':
        return TwoScaleDetector(**kwargs)
    if name == 'ProposalDetector':
        return ProposalDetector(**kwargs)
    else:
        raise TypeError('Unknown Detector Type!')
        
import ast       
def get_classes():
    with open(__file__, 'r') as f:
        tree = ast.parse(f.read(), filename=__file__)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

# def get_classes():
#     all_classes = []
#     for name, obj in inspect.getmembers(sys.modules[__name__]):
#         if inspect.isclass(obj):
#             all_classes.append(obj)
#     return dir(sys.modules[__name__])
            

# Detector base class
class Detector():
    
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
    
# SubDetector, consisting of a feature engine and a classifier engine. To be 
# used within other detector classes, that provide pre-processing and proposal
# engines
class SubDetector(Detector):
    
    name = 'SubDetector'
    
    def __init__(self,**attr_dict):
        
        # Call constructor of parent class (Detector)
        SensorType = attr_dict.pop('SensorType',None)
        super(SubDetector, self).__init__(SensorType,
                                          **attr_dict)
        
        # -------------------------- Attributes -------------------------------
        self.winSize = attr_dict.pop('winSize',None)
        
        ################## Engines ###########################################
        self.feature_engine = attr_dict.pop('feature_engine',None)
        self.classifier_engine = attr_dict.pop('classifier_engine',None)
        
        # ----------------------- Initialize Cpp instance --------------------
        if hasattr(self.classifier_engine, 'cv_classifier_engine'):
            self.initialize_cpp_instance()

    def initialize_cpp_instance(self):
               
        self.cv_detector = cv2.htpa.SubDetector(self.winSize,
                                                self.feature_engine.cv_feature_engine,
                                                self.classifier_engine.cv_classifier_engine)
    

    @property
    def winSize(self):
        return self._winSize
    @winSize.setter
    def winSize(self,winSize:tuple):
        if not isinstance(winSize,tuple):
            raise TypeError('winSize must be a tuple (w,h)')
        
        self._winSize = winSize

    @property
    def feature_engine(self):
        return self._feature_engine
    @feature_engine.setter
    def feature_engine(self,feature_eng):
        if feature_eng is None:
            self._feature_engine = None
        elif isinstance(feature_eng,dict):
            self._feature_engine = build_feature_engine(**feature_eng)
        elif isinstance(feature_eng,FeatureEngine):
            self._feature_engine = feature_eng
        else:
            raise Exception('Feature engine not recognized.')
        
    @property
    def classifier_engine(self):
        return self._classifier_engine    
    @classifier_engine.setter
    def classifier_engine(self,classifier_eng):
        if classifier_eng is None:
            self._classifier_engine = None
        elif isinstance(classifier_eng, dict):
            self._classifier_engine = build_classifier_engine(**classifier_eng)
        elif isinstance(classifier_eng,ClassifierEngine):
            self._classifier_engine = classifier_eng
        else:
            raise Exception('Classifier engine not recognized.')
            
    
    def process(self, video:AnnotatedVideo()):
        raise NotImplementedError('Subdetector has no process method!')

    
class OneScaleDetector(Detector):

    name = 'OneScaleDetector'
    
    def __init__(self,**attr_dict):
        
        # Call constructor of parent class (Detector)
        SensorType = attr_dict.pop('SensorType',None)
        super(OneScaleDetector, self).__init__(SensorType,
                                          **attr_dict)
        
        ################## Engines ###########################################
        self.preproc_engine = attr_dict.pop('preproc_engine',None)
        self.proposal_engine = attr_dict.pop('proposal_engine',None)
        self.postproc_engine = attr_dict.pop('postproc_engine',None)
        
        # ------------------------- SubDetector ------------------------------
        self.SubDetector = attr_dict.pop('SubDetector',None)
        
        self.initialize_cpp_instance()
        
    
    def initialize_cpp_instance(self):
               
        self.cv_detector = \
            cv2.htpa.OneScaleDetector(self.SensorType,
                                      self.SubDetector.cv_detector,
                                      self.preproc_engine.cv_preproc_engine,
                                      self.proposal_engine.cv_proposal_engine,
                                      self.postproc_engine.cv_postproc_engine)
    
        
    @property
    def preproc_engine(self):
        return self._preproc_engine  
    @preproc_engine.setter
    def preproc_engine(self,preproc_eng):
        if preproc_eng is None:
            self._preproc_engine = None
        elif isinstance(preproc_eng, dict):
            self._preproc_engine = build_preprocessing_engine(**preproc_eng)
        elif isinstance(preproc_eng,PreprocEngine):
            self._preproc_engine = preproc_eng
        else:
            raise Exception('Preprocessing engine not recognized.')
    
    @property
    def proposal_engine(self):
        return self._proposal_engine  
    @proposal_engine.setter
    def proposal_engine(self,prop_eng):
        if prop_eng is None:
            self._proposal_engine = None
        elif isinstance(prop_eng, dict):
            self._proposal_engine = build_proposal_engine(**prop_eng)
        elif isinstance(prop_eng,ProposalEngine):
            self._proposal_engine = prop_eng
        else:
            raise Exception('Proposal engine not recognized.')
            
    @property
    def postproc_engine(self):
        return self._postproc_engine
    @postproc_engine.setter
    def postproc_engine(self,postproc_eng):
        if postproc_eng is None:
            self._postproc_engine = None
        elif isinstance(postproc_eng, dict):
            self._postproc_engine = build_postprocessing_engine(**postproc_eng)
        elif isinstance(postproc_eng,PostprocEngine):
            self._postproc_engine = postproc_eng
        else:
            raise Exception('Post-processing engine not recognized.')

    @property
    def SubDetector(self):
        return self._SubDetector
    @SubDetector.setter
    def SubDetector(self,SubDet):
        if SubDet is None:
            self._SubDetector = None
        elif isinstance(SubDet, dict):
            self._SubDetector = build_detector(**SubDet)
        elif isinstance(SubDet,SubDetector):
            self._SubDetector = SubDet
        else:
            raise Exception('SubDetector not recognized.')
    
    def _process_frame(self,
                       frame:np.ndarray,
                       idx:int,
                       **kwargs:dict) -> dict:
        
        # Get keyword arguments
        gt_annot = kwargs.pop('gt_annot',None)
        update_bg = kwargs.pop('update_bg',False)
        use_gt = kwargs.pop('use_gt',True)
        
        frame = frame.astype(np.uint16)
        
        det_cv, frame_pre = self.cv_detector.process_frame(frame,
                                                           idx,
                                                           True,
                                                           True)
        
        # Convert to pd.DataFrame
        det = misc.cv2_to_pd_DataFrame(det_cv)
                    
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
        if isinstance(self.proposal_engine, hsdetect.cv.proposal_engines.BackSub):
            return_dict['bg'] = \
                self.proposal_engine.cv_proposal_engine._bg
            return_dict['bg_mask'] = \
                self.proposal_engine.cv_proposal_engine._bg_mask
                
        # If the ProposalEngine uses background subtraction, update the 
        # background
        if isinstance(self.proposal_engine, hsdetect.cv.proposal_engines.BackSub):
            if update_bg:
                # Use the given annotations background updating
                if use_gt:
                    detections = frame.annotations
                # Use the predictions for background updating
                else:
                    detections = return_dict['annot_frame'].annotations
                    
                self.proposal_engine.update_bg(return_dict['frame_pre'],
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
            
            # Call process_frame() on frame
            return_dict = \
                self._process_frame(img,
                                    idx,
                                    frame.annotations,
                                    **kwargs)
            
            # Append returned AnnotatedFrame 
            annotated_frames.append(return_dict['annot_frame'])
            

            
        # In the end return an annotated video
        return AnnotatedVideo(frames = annotated_frames)

class ProposalDetector(OneScaleDetector): # Formerly DummyDetector
    
    name = 'ProposalDetector'
    
    def __init__(self,**attr_dict):
        
        # Call constructor of parent class (Detector)
        SensorType = attr_dict.pop('SensorType',None)
        super(OneScaleDetector, self).__init__(SensorType,
                                          **attr_dict)
        
        ################## Engines ###########################################
        self.preproc_engine = attr_dict.pop('preproc_engine',None)
        self.proposal_engine = attr_dict.pop('proposal_engine',None)
        
        self.SubDetector = None
        self.postproc_engine = None
    
    def _process_frame(self,
                       frame:np.ndarray,
                       idx:int) -> dict:
        
        frame_pre, idx_pre = self.preproc_engine.process(frame = frame,
                                                         idx = idx)

        
        annot_frame = self.proposal_engine.process(frame = frame_pre,
                                                   idx = idx_pre)
        return_dict = {'frame_pre':frame_pre,
                       'idx_pre': idx_pre,
                       'annot_frame':  annot_frame}
                
        if isinstance(self.proposal_engine, hsdetect.cv.proposal_engines.BackSub):
            return_dict['bg'] = \
                self.proposal_engine.cv_proposal_engine._bg
            return_dict['bg_mask'] = \
                self.proposal_engine.cv_proposal_engine._bg_mask
        
        return return_dict
        
        
        
    
    
    
class TwoScaleDetector(Detector):

    name = 'TwoScaleDetector'

    def __init__(self,**attr_dict:dict):
        raise NotImplementedError('Message AR: Implement Wrapper for TwoScaleDetector')
    
    

# class Detector():
    
#     def __init__(self,**attr_dict):
        
        
#         self._width = attr_dict.pop('width',None)
#         self._height = attr_dict.pop('height',None)
        
#         self._SensorType = attr_dict.pop('SensorType',None)
#         self._tparray = TPArray(SensorType = self.SensorType)
        
#         # self._dwsize = attr_dict.pop('dwsize',None)
#         self._IoU_dw = attr_dict.pop('IoU_dw',None)

#         self._class_labels = attr_dict.pop('class_labels',
#                                            {'person':1})

#         ################## Engines ###########################################
#         self.preproc_engine = attr_dict.pop('preproc_engine',None)
#         self.proposal_generator = attr_dict.pop('proposal_generator',None)
#         self.feature_engine = attr_dict.pop('feature_engine',None)
#         self.classifier_engine = attr_dict.pop('classifier_engine',None)
#         self.postproc_engine = attr_dict.pop('postproc_engine',None)

#         ################# Training Data ######################################
#         self._Xy = attr_dict.pop('Xy',(None,None))
#         self._annot_Xy = attr_dict.pop('annot_Xy',None)
        
        
#         # self._annot_mgr = AnnotMgr(self.tparray)
    
#     @property
#     def width(self):
#         return self._width
#     @width.setter
#     def width(self,w):
#         self._width = w
    
#     @property
#     def height(self):
#         return self._height
#     @height.setter
#     def height(self,h):
#         self._height = h

#     @property
#     def SensorType(self):
#         return self._SensorType
#     @SensorType.setter
#     def SensorType(self,SensorType):
#         self._SensorType = SensorType

#     @property
#     def tparray(self):
#         return self._tparray  
#     @tparray.setter
#     def tparray(self,tparray):
#         if tparray is None:
#             self._tparray = None
#         elif isinstance(tparray, dict):
#             self._tparray = TPArray(**tparray)
#         elif isinstance(tparray,TPArray):
#             self._tparray = tparray
#         else:
#             raise Exception('Type not recognized.')
    
#     # @property
#     # def dwsize(self):
#     #     return self._dwsize
#     # @dwsize.setter
#     # def dwsize(self,tuple_val):
#     #     self._dwsize = tuple_val
        
#     @property
#     def IoU_dw(self):
#         return self._IoU_dw
#     @IoU_dw.setter
#     def IoU_dw(self,float_val):
#         self._IoU_dw = float_val  
    
#     @property
#     def class_labels(self):
#         return self._class_labels
#     @class_labels.setter
#     def class_labels(self,class_labels:dict):
#         self._class_labels = class_labels
    
#     @property
#     def preproc_engine(self):
#         return self._preproc_engine  
#     @preproc_engine.setter
#     def preproc_engine(self,preproc_eng):
#         if preproc_eng is None:
#             self._preproc_engine = None
#         elif isinstance(preproc_eng, dict):
#             self._preproc_engine = build_preprocessing_engine(**preproc_eng)
#         elif isinstance(preproc_eng,PreprocEngine):
#             self._preproc_engine = preproc_eng
#         else:
#             raise Exception('Preprocessing engine not recognized.')
    
#     @property
#     def proposal_generator(self):
#         return self._proposal_generator  
#     @proposal_generator.setter
#     def proposal_generator(self,prop_gen):
#         if prop_gen is None:
#             self._proposal_generator = None
#         elif isinstance(prop_gen, dict):
#             self._proposal_generator = build_proposal_generator(**prop_gen)
#         elif isinstance(prop_gen,ProposalGenerator):
#             self._proposal_generator = prop_gen
#         else:
#             raise Exception('Proposal generator not recognized.')
                
#     @property
#     def feature_engine(self):
#         return self._feature_engine
#     @feature_engine.setter
#     def feature_engine(self,feature_eng):
#         if feature_eng is None:
#             self._feature_engine = None
#         elif isinstance(feature_eng,dict):
#             self._feature_engine = build_feature_engine(**feature_eng)
#         elif isinstance(feature_eng,FeatureEngine):
#             self._feature_engine = feature_eng
#         else:
#             raise Exception('Feature engine not recognized.')

#     @property
#     def postproc_engine(self):
#         return self._postproc_engine
#     @postproc_engine.setter
#     def postproc_engine(self,postproc_eng):
#         if postproc_eng is None:
#             self._postproc_engine = None
#         elif isinstance(postproc_eng, dict):
#             self._postproc_engine = build_postprocessing_engine(**postproc_eng)
#         elif isinstance(postproc_eng,PostprocEngine):
#             self._postproc_engine = postproc_eng
#         else:
#             raise Exception('Post-processing engine not recognized.')
        
#     @property
#     def classifier_engine(self):
#         return self._classifier_engine    
#     @classifier_engine.setter
#     def classifier_engine(self,classifier_eng):
#         if classifier_eng is None:
#             self._classifier_engine = None
#         elif isinstance(classifier_eng, dict):
#             self._classifier_engine = build_classifier_engine(**classifier_eng)
#         elif isinstance(classifier_eng,ClassifierEngine):
#             self._classifier_engine = classifier_eng
#         else:
#             raise Exception('Classifier engine not recognized.')
    
#     @property
#     def Xy(self):
#         return self._Xy
#     @Xy.setter
#     def Xy(self,Xy):
        
#         # Get existing training data
#         Xy_old = self.Xy
        
#         # Exception if no data to append to exists yet
#         if (Xy_old[0] is None) or (Xy_old[1] is None):
#             X = Xy[0]
#             y = Xy[1]
#         else:
#             # Append new data to existing data
#             X = np.vstack([Xy_old[0],Xy[0]])
#             y = np.vstack([Xy_old[1],Xy[1]])
        
#         # Save 
#         self._Xy = (X,y)
        
#     @property
#     def annot_Xy(self):
#         return self._annot_Xy
#     @annot_Xy.setter
#     def annot_Xy(self,annot_Xy):
        
#         # Get existing annotations
#         annot_Xy_old = self.annot_Xy
        
#         # Exception if no data to append to exists yet
#         if annot_Xy_old is not None:
#             # Append new data to existing data
#             annot_Xy = pd.concat([annot_Xy_old,annot_Xy])
        
#         # Reset and rename index to 'id'. So that id always corresponds to
#         # the row in self.Xy             
#         annot_Xy.reset_index(drop=True,inplace=True)
#         annot_Xy.index.rename('id',inplace=True)
        
#         self._annot_Xy = annot_Xy 

#     def clear_data(self):
#         """
#         Reset the attributes storing estimation data, i.e. deleting any data
#         that has been passed to the detector previously

#         Returns
#         -------
#         None.

#         """
        
#         self._annot_Xy = None
#         self._Xy = (None,None)               

#     def save(self):
#         """
#         Function for saving all attributes of the detector to a dictionary
#         and writing it to a file.

#         Parameters
#         ----------

#         Returns
#         -------
#         None.

#         """
        
#         # Get all names of properties of the instance by doing
#         properties = []
#         for d in dir(self):
#             if isinstance(getattr(type(self), d, None), property):
#                 properties.append(d)
                
        
#         # Save all properties to an attr_dict
#         # Some properties have their own save-method. Use that, where available
#         attr_dict = {}
        
#         # Loop over property keys
#         for p in properties:
            
#             # Get property value
#             prop = getattr(self,p) 
            
#             # Check if prop has a save method
#             save_method = getattr(prop, "save", None)
            
#             # Check if its a callable method
#             if callable(save_method):
#                 # If its a callable save method, call it
#                 attr_dict[p] = {}
#                 attr_dict[p] = save_method()
#             else:
#                 attr_dict[p] = prop
                
#         # Get all class attributes as well
#         class_dict = {}
#         for attribute in self.__class__.__dict__.keys():
#             # Check if it's a built-in type
#             if (attribute[:2] != '__') and attribute not in attr_dict :
#                 # Check if its a method
#                 value = getattr(self.__class__, attribute)
#                 if not callable(value):
#                     # If not append to dict
#                     class_dict[attribute] = value
        
#         # Concatenate both
#         attr_dict.update(class_dict)
            
#         return attr_dict     

#     def IoU_with_dw(self,annot:pd.DataFrame)->pd.DataFrame:
        
#         # Calculate the IoU with the desired detection window size 
#         annot['IoU_dw'] = np.nan
                   
#         for a in annot.index:
#             tl = (annot.loc[a,'xtl'],annot.loc[a,'ytl'])
#             br = (annot.loc[a,'xbr'],annot.loc[a,'ybr'])
            
#             br_det = (tl[0]+self.dwsize[0],tl[1]+self.dwsize[1])
            
#             annot.loc[a,'IoU_dw'] = IoU((tl,br),
#                                      (tl,br_det))
            
#         # Cast corners of bounding box to integer
#         annot = annot.astype({'xtl':int, 'ytl':int, 'xbr':int,
#         'ybr':int})
            
#         return annot

#     # def add_border(self,annot):
        
#     #     # Shift the box to account for adding a border around the frame
#     #     annot[['xtl','ytl','xbr','ybr']] = annot[['xtl','ytl','xbr','ybr']] + \
#     #         self.border
        
        
#     #     # Add a frame by shifting the upper left and bottom right corner
#     #     for coord in ['xtl','ytl']:
#     #         annot[coord] = annot[coord] - self.border
        
#     #     for coord in ['xbr','ybr']:
#     #         annot[coord] = annot[coord] + self.border
            
#     #     # Then remove all boxes that are outside of the frame
#     #     annot = annot.loc[(annot['xtl']>=0) &\
#     #                       (annot['ytl']>=0) &\
#     #                       (annot['xbr']<=self.tparray._width+2*self.border) &\
#     #                       (annot['ybr']<=self.tparray._height+2*self.border)]
                          
#     #     return annot

#     def prep_Xy(self,annot_video:AnnotatedVideo,
#                      **kwargs:dict)->tuple:
#         """
#         Parameters
#         ----------
#         frame_dict : dict
#             DESCRIPTION.
#         annot : pd.DataFrame
#             DESCRIPTION.
#         **kwargs : dict
#             DESCRIPTION.

#         Returns
#         -------
#         tuple
#             DESCRIPTION.

#         """
        
        
#         # Save the extracted regression matrix and target vector as 
#         # well as the annotation to the model's attributes
#         # save = kwargs.pop('save',False)
#         # masks_dict = kwargs.pop('masks',False)
#         # Calculate the IoU of each annotation with the detection window dw
#         # annot = self.IoU_with_dw(annot)
               
#         # Go over all remaining annotations. Extract the content of the 
#         # bounding boxes, apply pre-processing and feature extraction steps
#         X = []
#         y = []
        
#         # Loop over all frames in the provided dictionary
#         for frame in annot_video.frames:

#             # Get the frame as numpy array and the corresponding annotations
#             # as pd.DataFrame
#             image = frame.image
#             annot = frame.annotations
#             mask = frame.mask
            
#             # Filter for annotations that were assigned to this detector in the
#             # first place
#             annot = annot.loc[annot['detector_idx'] == self.detectorID]
            
#             # If no annotations exist, continue
#             # Skip if no annotations for this frame
#             if len(annot)==0:
#                 continue
            
#             # Otherwise convert the annotations to the cv2.htpa.DataFrame
#             # type
#             cv_annot = self._pandas_to_htpa(annot)
            
#             # Pass to the cv2.htpa class to calculate feature vector
#             x = self.feature_engine.get_features(img = image,
#                                                  mask = mask,
#                                                  bboxes = cv_annot)
            
#             # Append feature vectors to regression matrix
#             X.append(x)
            
#             # Append labels to target vector
#             target = annot['label'].values.flatten()
#             y.append(target)
            
#             # Mirror annotations of frame
#             annot_frame_lr = annot.copy()
#             annot_frame_lr['xtl'] = - annot_frame_lr['xtl'] + self.tparray._width 
#             annot_frame_lr['xbr'] = - annot_frame_lr['xbr'] + self.tparray._width 

#             annot_frame_lr['xtl'] = annot_frame_lr['xtl'] - annot_frame_lr['w']
#             annot_frame_lr['xbr'] = annot_frame_lr['xbr'] + annot_frame_lr['w']

#             cv_annot_frame_lr = self._pandas_to_htpa(annot_frame_lr)
            
#             # Mirror frame
#             image_lr = np.fliplr(image).copy()
            
#             # Mirror mask
#             mask_lr = np.fliplr(mask).copy()
            
#             # Get features of mirrored data
#             x_mirror = self.feature_engine.get_features(img = image_lr,
#                                                         mask = mask_lr,
#                                                         bboxes = cv_annot_frame_lr)
            
#             X.append(x_mirror)
#             y.append(target)
        
#         # Check that neither X nor y are empty and concatenate
#         if len(X)!=0 and len(y)!=0:
#             X = np.vstack(X).astype(np.float32)
#             y = np.hstack(y).reshape((-1,1)).astype(np.float32)
#         else:
#             # If X and y are empty, convert them to an empty numpy array
#             n_features = self.feature_engine.cv_feature_engine.get_n_features()
#             X = np.array(X).reshape((-1,n_features)).astype(np.float32)
#             y = np.array(y).reshape((-1,1)).astype(np.float32)
        
#         # if save:
#         #     # Save regression matrix and target vector
#         #     self.Xy = (X,y)
            
#         #     # Save metadata for annotations to dataset
#         #     self.annot_Xy = annot
        
#         return (X,y)

#     def prep_X(self,frame_dict:dict,
#                     bboxes:pd.DataFrame,
#                     **kwargs:dict)->tuple:
        
#         # Calculate the IoU of each annotation with the detection window dw
#         bboxes = self.IoU_with_dw(bboxes)
        
#         # Filter out all annotations that are below the defined limit
#         bboxes = bboxes.loc[bboxes['IoU_dw']>=self.IoU_dw]
        
#         # If desired, add a border to all remaining annotations. If a border
#         # is added to the box, it has already been added to the frame!
#         # So the coordinates of the box have to be shifted before the box is 
#         # enlarged
#         # bboxes_border = self.add_border(bboxes.copy())
        
#         # bboxes_border is only used to get the the content of the orginally
#         # proposed bounding box plus border !
        
        
#         # Go over all remaining annotations. Extract the content of the 
#         # bounding boxes, apply pre-processing and feature extraction steps
#         X = []
        
#         for a_idx in bboxes.index:
            
#             # Get the frame id
#             frame_id = int(bboxes.loc[a_idx,'image_id'])
            
#             frame = frame_dict[frame_id]
            
#             # Extract the bounding box            
#             bbox = self._get_bbox(frame, bboxes, a_idx)
                        
#             # Pre-process the bounding box
#             bbox = self.preprocess_bbox(bbox)
            
#             # Pass to the feature extraction method
#             X.append(self.feature_engine.get_features(bbox))
        
        
#         # Add observations to the training data
#         if len(X) != 0:
#             X = np.vstack(X)
        
#         return X,bboxes

#     # def relabel_annot(self,annot):   
#     #     """
#     #     Coming from CVAT each class gets a random label_id that might even 
#     #     differ between CVAT projects. This function gives each class defined 
#     #     by a string, e.g. 'person', the label_id defined in self.class_labels

#     #     Parameters
#     #     ----------
#     #     annot : TYPE
#     #         DESCRIPTION.

#     #     Raises
#     #     ------
#     #     Exception
#     #         DESCRIPTION.

#     #     Returns
#     #     -------
#     #     annot : TYPE
#     #         DESCRIPTION.

#     #     """
        
        
#     #     if len(self.class_labels.keys())>1:
#     #         raise Exception("This code needs to be adapted for multiple classes.")
        
#     #     # Give the labels in annotations the label_id used by the detector
#     #     for label,label_id in self.class_labels.items():
#     #        annot.loc[annot['label_name']==label,'label'] = int(label_id)

#     #     # All remaining labels get a 0
#     #     annot.loc[annot['label_name'] != label,'label'] = 0
        
#     #     return annot
        
#     def predict_cvat(self,img):
#         """
#         Receives the bounding boxes containing the detected objects and
#         converts them into a list of rectangle shapes that CVAT is able
#         to incorporate in its annotations

#         Parameters
#         ----------
#         img : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         shapes : TYPE
#             DESCRIPTION.

#         """
#         # Detect blobs (assumed persons)
#         bboxes = self.predict(img)
        
#         # empty list for cvat labeled shapes
#         shapes = []
        
#         # Parse into a rectangle object that CVAT understands
#         for b in bboxes.index:
#             shapes.append(cvataa.rectangle(1,[bboxes.loc[b]['xtl'].item(),
#                                               bboxes.loc[b]['ytl'].item(),
#                                               bboxes.loc[b]['xbr'].item(),
#                                               bboxes.loc[b]['ybr'].item()]))
        
#         return shapes
    
#     def _get_bbox(self,frame,annot,a_idx):
        
#         # Extract the bounding box from the annotations
#         xtl = annot.loc[a_idx,'xtl']
#         ytl = annot.loc[a_idx,'ytl']
#         xbr = annot.loc[a_idx,'xbr']
#         ybr = annot.loc[a_idx,'ybr']
    
#         # Get content of the bounding box
#         bbox = frame[ytl:ybr,xtl:xbr]
        
#         return bbox
        
#     def fit(self,**kwargs):
        
#         # Load prepared regression matrix and target vector
#         X = self.Xy[0]
#         y = self.Xy[1]
        
#         score = self.classifier_engine.fit(X,
#                                            y.flatten())
        
#         return score

#     def propose(self,**kwargs)->pd.DataFrame:
#         """
#         Applies only the pre-processing engine and proposal-generator to the 
#         provided image or video

#         Parameters
#         ----------
#         img : np.ndarray
#             Frame to apply detector to.
#         img_idx : int
#             Index of that frame.
#         video: pd.DataFrame
#             Video or sequence of frames as pd.DataFrame to apply detector to
#         pre_process: bool
#             If true, preprocessing engine is applied to each frame. If false, 
#             no preprocessing is applied.

#         Returns
#         -------
#         None.

#         """
        
#         img = kwargs.pop('img',None)
#         img_idx = kwargs.pop('img_idx',None)
#         video = kwargs.pop('video',None)
        
#         # pre_process = kwargs.pop('pre_process',True)
        
#         # Check if single frame or video is provided
#         if isinstance(img,np.ndarray) and isinstance(img_idx,int):
        
#             # Cast image to proper type just to be sure
#             img = img.astype(np.uint16)
            
#             # Apply detector to the single given frame
#             raise Exception("Not yet implemented!")
            
#         elif isinstance(video,pd.DataFrame):
            
#             # Apply preprocessing-engine to whole video
#             video_proc = self.preproc_engine.process(video=video)
            
#             # Apply proposal-generator to the preprocessed video
#             proposals = self.proposal_generator.get_proposals(video=video_proc)
            
#         return proposals

#     def predict(self,**kwargs):
                
#         # %% Apply the proposal generator to the video
#         # print('Get proposals')
#         bbox_prop = self.prop_generator.get_proposals(**kwargs)
       
#         # Pass the video and the generated bbox proposals to get_frames() to 
#         # get all frames with bbox proposals in a dictionary 
#         # print('Get frames')
#         frame_dict = self.get_frames(annot=bbox_prop.copy(),
#                                      **kwargs)

                    
#         # Apply feature engine
#         X,bbox_prop = self.prep_X(frame_dict, bbox_prop.copy())
        
#         # Pass features to classifier
#         if len(bbox_prop)!=0:
#             y_pred = self.classifier_engine.decision_function(X)
    
#             # return label and score
#             bbox_prop['conf_score'] = y_pred
            
#         # Apply postprocessing 
#         bbox_prop = self.postproc_engine.process(bbox_prop,self.class_labels)
        
#         return bbox_prop

#     def get_frames(self,annot:pd.DataFrame,**kwargs)->dict:
            
#         """
#         Fetches the frame to every annotation from the provided video
#         ----------
#         name : TYPE
#             DESCRIPTION.
#         det_height : TYPE
#             DESCRIPTION.
#         **kwargs : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """
        
#         frame = kwargs.pop('frame',None)
#         image_id = kwargs.pop('image_id',None)
#         video = kwargs.pop('video',None)              
        
#         # If a single frame is provided, this function is superfluous
#         # because its purpose is to extract only those  frames from a sequence 
#         # for which annotations exist 
#         if frame is not None:
                            
#             return {image_id:frame}
        
#         elif video is not None:
        
#             # Initialize a dictionary to store frames in
#             frame_dict = {}
            
#             # Loop over every annotation
#             for a in annot.index:
                
#                 # Get id of the frame in the video
#                 image_id = annot.loc[a,'image_id'].astype(int)
                                
#                 # Get the frame
#                 frame = video.loc[image_id,self.tparray._pix].values
#                 frame = frame.reshape(self.tparray._npsize)
                                            
#                 frame_dict[image_id] = frame
    
#             return frame_dict



# class SingleScaleDetector_CV(Detector):
 
#     name = 'SingleScaleDetector_CV'
        
#     def __init__(self,**kwargs:dict):
#         """
        
#         Parameters
#         ----------
#         tparray : TPArray
#             DESCRIPTION.        
#         dwsize : tuple
#             DESCRIPTION.
#         IoU_dw : float
#             DESCRIPTION.
#         IoU_dw : int
#             DESCRIPTION.
#         feature_engine : 
#             DESCRIPTION
#         preproc_engine : 
#             DESCRIPTION
#         **kwargs : dict
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """
        
#         # Call the initialization method of the parent class first
#         super(SingleScaleDetector_CV,self).__init__(**kwargs)
        
#         # Set attributes
#         self.detectorID = kwargs.pop('detectorID',None)
#         self.dwsize = kwargs.pop('dwsize',None)
        
#         # Two cases must be distinguished: The passed classifier is already
#         # fitted, then it has an attribute cv_classifier_engine and the 
#         # OpenCV implementation of the SingleScale Detector can be 
#         # initialized
#         if not isinstance(self.classifier_engine,type(None)):
#             if 'cv_classifier_engine' in self.classifier_engine.__dict__:
#                 self._init_cv_single_scale_detector()
            
#         # If the classifier is not fitted, cv2.htpa.SingleScaleDetector
#         # is initialized when fit() is called
    
#     @property
#     def detectorID(self):
#         return self._detectorID
#     @detectorID.setter
#     def detectorID(self,ID:int):
#         if not isinstance(ID,int):
#             raise Exception('detectorID needs to be an integer.')
#         self._detectorID = ID
        
#     @property
#     def dwsize(self):
#         return self._dwsize
#     @dwsize.setter
#     def dwsize(self,dwsize:tuple):
#         if not isinstance(dwsize,tuple):
#             raise Exception('dwsize needs to be a tuple.')
#         self._dwsize = dwsize


    
#     def _init_cv_single_scale_detector(self):
#         '''
#         Initializes OpenCV implementation of the SingleScale Detector

#         Returns
#         -------
#         None.

#         '''
#         win_size = self.dwsize
#         cv_feature_engine = self.feature_engine.cv_feature_engine
#         cv_classifier_engine = self.classifier_engine.cv_classifier_engine
        
#         self.cv_SingleScaleDetector = \
#             cv2.htpa.SubDetector(win_size,
#                                  cv_feature_engine,
#                                  cv_classifier_engine)
    
#     def fit(self,**kwargs):
#         '''
#         Calls fit method of classifier engine

#         Parameters
#         ----------
#         **kwargs : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         '''
              
#         # Load prepared regression matrix and target vector
#         X = self.Xy[0]
#         y = self.Xy[1]
        
#         # Exception: X and y are None or empty, in this case the classifier
#         # can't be fitted
#         no_data = False
#         if X is None or y is None:
#             no_data = True
        
#         if X is not None:
#             if len(X) == 0:
#                 no_data = True
        
#         if y is not None:
#             if len(y) == 0:
#                 no_data = True
                
#         if no_data:
#             raise Warning("No data. Classifier can't be fittet.")
        
#         # In all other cases, fit the classifier
#         score = self.classifier_engine.fit(X,
#                                            y.flatten())
        
#         # If fit was successful, initialize cv2.htpa.SingleScaleDetector
#         if 'cv_classifier_engine' in self.classifier_engine.__dict__:
#             self._init_cv_single_scale_detector()
            
#         return score
        
#     def predict(self,**kwargs):
        
#         # Write here to differentiate between prediction for a single frame
#         # or a whole video
        
#         frame = kwargs.pop('frame',None)
#         bbox_prop = kwargs.pop('bbox_prop',None)
#         image_id = kwargs.pop('image_id',None)
#         video = kwargs.pop('video',None)
        
#         bbox_pred = self.cv_SingleScaleDetector.predict(frame,
#                                                         bbox_prop,
#                                                         image_id)
        
#         # Write cv.htpa.DataFrame to pandas DataFrame
#         df_cols = bbox_pred.columns
#         df_content = [[eval('box.'+col) for col in df_cols] 
#                       for box in bbox_pred.rows]
        

#         df = pd.DataFrame(data = df_content,
#                           columns= df_cols)
            
#         return df
    
#     def _pandas_to_htpa(self, df_pd:pd.DataFrame)->cv2.htpa.DataFrame:
#         '''
#         Convert pandas DataFrame to cv.htpa.DataFrame

#         Parameters
#         ----------
#         df_pd : pd.DataFrame
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         '''
#         df_htpa = cv2.htpa.DataFrame()
        
#         for idx in df_pd.index:
#             df_htpa.add_row(idx, 
#                             df_pd.loc[idx,'xtl'],
#                             df_pd.loc[idx,'ytl'],
#                             df_pd.loc[idx,'xbr'],
#                             df_pd.loc[idx,'ybr'],
#                             df_pd.loc[idx,'image_id'],
#                             df_pd.loc[idx,'score'],
#                             df_pd.loc[idx,'track'],
#                             df_pd.loc[idx,'strikes'],
#                             df_pd.loc[idx,'label'],
#                             df_pd.loc[idx,'IoU_max'],
#                             df_pd.loc[idx,'IoU_idxmax'],
#                             df_pd.loc[idx,'detector_idx'])
        
#         return df_htpa
    
    
#     def save(self):
#         """
#         Function for saving all attributes of the detector to a dictionary
#         and writing it to a file.

#         Parameters
#         ----------

#         Returns
#         -------
#         None.

#         """
        
#         # Get all names of properties of the instance by doing
#         properties = []
#         for d in dir(self):
#             if isinstance(getattr(type(self), d, None), property):
#                 properties.append(d)
                
        
#         # Save all properties to an attr_dict
#         # Some properties have their own save-method. Use that, where available
#         attr_dict = {}
        
#         # Loop over property keys
#         for p in properties:
            
#             # Get property value
#             prop = getattr(self,p) 
            
#             # Check if prop has a save method
#             save_method = getattr(prop, "save", None)
            
#             # Check if its a callable method
#             if callable(save_method):
#                 # If its a callable save method, call it
#                 attr_dict[p] = {}
#                 attr_dict[p] = save_method()
#             else:
#                 # Check if it's a property containing training data.
#                 # Training data is not saved because it takes to much storage
#                 if not (p == 'Xy') and not(p == 'annot_Xy'):
#                     attr_dict[p] = prop
                
#         # Get all class attributes as well
#         class_dict = {}
#         for attribute in self.__class__.__dict__.keys():
#             # Check if it's a built-in type
#             if (attribute[:2] != '__') and attribute not in attr_dict :
                
#                 # Get the value of the attribute
#                 value = getattr(self.__class__, attribute)
                
#                 # Check if its a method
#                 if not callable(value):
                    
#                     # If not, save                    
#                     class_dict[attribute] = value
        
#         # Concatenate both
#         attr_dict.update(class_dict)
            
#         return attr_dict   


# class MultiScaleDetector_CV(Detector):
    
#     name = 'MultiScaleDetector_CV'
    
#     def __init__(self,**kwargs:dict):
        
#         # Initialize the attributes of this detector itself
#         super().__init__(**kwargs)
        
#         # Initialize sub-detectors as an empty list
#         self.detectors = kwargs.pop('detectors',())
        
#         # If all of the passed SingleScale detectors already contain
#         # trained classifier, initialize the OpenCV implementation of the
#         # MultiScaleDetector
#         if len(self.detectors) != 0:
#             if all(['cv_SingleScaleDetector' in detector.__dict__
#                     for detector in self.detectors]):
#                 self._init_cv_multi_scale_detector()
        
#         # If not, the cv.htpa.MultiScaleDetector will be initialized after
#         # the fit() method has been successfully applied
        
#         self.cv_datatypes = {'idx':'int32','xtl':'int32','ytl':'int32',
#                              'xbr':'int32','ybr':'int32','w':'int32',
#                              'h':'int32','score':'float64','track':'int32',
#                              'label':'int32','strikes':'int32',
#                              'IoU_max':'float64','IoU_idxmax':'int32',
#                              'detector_idx':'int32','image_id':'int32'}
                
#     @property
#     def detectors(self):
#         return self._detectors 
#     @detectors.setter
#     def detectors(self,detectors:tuple):
#         # Check if detectors is a tuple
#         if not isinstance(detectors,tuple):
#             raise Exception('Detectors must be a tuple of instances of Detector or '\
#                             'a list of dict.')
        
#         # Initialize each detector in list
#         _detectors = ()
#         for det in detectors:
#             if isinstance(det,SingleScaleDetector_CV):
#                 _detectors = _detectors + (det,)
#             elif isinstance(det,dict):
#                 _detectors = _detectors + (build_detector(**det),)
                
#         self._detectors = _detectors
        
        
#     def _init_cv_multi_scale_detector(self):
#         '''
#         Initializes OpenCV implementation of the MultiScale Detector

#         Returns
#         -------
#         None.

#         '''
        
#         cv_tparray = cv2.htpa.TPArray(self.tparray.SensorType)
        
#         # cv_detectors = cv2.htpa.DetectorList(len(self.detectors))
        
#         # for detector in self.detectors:
#         #     cv_detectors.add_detector(detector.cv_SingleScaleDetector)
        
#         cv_preproc_engine = self.preproc_engine.cv_preproc_engine
#         cv_proposal_generator = self.proposal_generator.cv_proposal_generator
#         cv_postproc_engine = self.postproc_engine.cv_postproc_engine
        
        
        
#         # At the moment the cv2.htpa.MultiScaleDetector class is not able to 
#         # handle a variable number of detectors
        
#         if len(self.detectors)==1:
            
#             cv_detector1 = self.detectors[0].cv_SingleScaleDetector
            
#             self.cv_MultiScaleDetector = \
#                 cv2.htpa.MultiScaleDetector(cv_tparray,
#                                             cv_detector1,
#                                             cv_preproc_engine,
#                                             cv_proposal_generator, 
#                                             cv_postproc_engine)
        
        
#         elif len(self.detectors)==2:
        
#             cv_detector1 = self.detectors[0].cv_SingleScaleDetector
#             cv_detector2 = self.detectors[1].cv_SingleScaleDetector
            
#             self.cv_MultiScaleDetector = \
#                 cv2.htpa.TwoScaleDetector(self.SensorType,
#                                           cv_detector1,
#                                           cv_detector2,
#                                           cv_preproc_engine,
#                                           cv_proposal_generator, 
#                                           cv_postproc_engine)

    
#     def clear_data(self):
#         """
#         Reset the attributes storing estimation data, i.e. deleting any data
#         that has been passed to the detector previously

#         Returns
#         -------
#         None.

#         """
        
#         for detector in self.detectors:
#             detector.clear_data()
            
#         return None
    
#     def reset(self):
        
#         # The pre-processing engine has a built-in memory that is somehow
#         # hard to erase. Therefore build a new one here first
        
#         # Get parameters of old engine
#         attr_dict = self.preproc_engine.save()
        
#         # Build a new clean engine from it
#         clean_preproc_engine = build_preprocessing_engine(**attr_dict)
        
#         # Assign the new engine as an attribute
#         self.preproc_engine = clean_preproc_engine
        
#         # Just to be save: If the detector has already been fitted,
#         # also create a new instance of the Cpp class
#         if all(['cv_SingleScaleDetector' in detector.__dict__
#                 for detector in self.detectors]):
#             self._init_cv_multi_scale_detector()
    
#     def save(self):
#         """
#         Overwrites parent method which is only fit to save properties
#         of a single detector

#         Returns
#         -------
#         None.

#         """
        
#         # Get all properties of the instance by doing
#         properties = []
#         for d in dir(self):
#             if isinstance(getattr(type(self), d, None), property):
#                 properties.append(d)
                
        
#         # Save all properties to an attr_dict
#         # Some properties have their own save-method. Use that, where available
#         attr_dict = {}
        
#         # Loop over property keys
#         for p in properties:
            
#             # Get property value
#             prop = getattr(self,p) 
            
#             # Check if prop has a save method
#             save_method = getattr(prop, "save", None)
            
#             # Check if its a callable method
#             if callable(save_method):
#                 # If its a callable save method, call it and save dictionary
#                 attr_dict[p] = {}
#                 attr_dict[p] = save_method() 
#             elif p == 'detectors':
#                 # If the attribute is detectors, than call save method of every
#                 # single detector
#                 attr_dict[p] = ()
#                 for detector in prop:
#                     attr_dict[p] = attr_dict[p] + (detector.save(),)    
#             else:
#                 # Check if it's a property containing training data.
#                 # Training data is not saved because it takes to much storage
#                 if not (p == 'Xy') and not(p == 'annot_Xy'):
#                     attr_dict[p] = prop
                
#         # Get all class attributes as well
#         class_dict = {}
#         for attribute in self.__class__.__dict__.keys():
#             # Check if it's a built-in type
#             if (attribute[:2] != '__') and attribute not in attr_dict :
#                 # Check if its a method
#                 value = getattr(self.__class__, attribute)
#                 if not callable(value):
#                     # If not append to dict
#                     class_dict[attribute] = value
        
#         # Concatenate both
#         attr_dict.update(class_dict)
        
#         return attr_dict  
        
#     def _assign_annotations_to_subdetectors(self,annot_video:AnnotatedVideo):
#         """
#         Method iterates through all frames in annot_video and assigns the
#         annotations to the detector, which detection window has the largest
#         IoU with the bounding box of the given annotation. The result is 
#         written as the corresponding detector index into a new column
#         "detector_idx" of the frames annotation
#         Parameters
#         ----------
#         annot_video : AnnotatedVideo
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """    
        
#         # Loop over all frames
#         for frame in annot_video.frames:
            
#             # Get annotations of that frame
#             annot = frame.annotations
            
#             # Compare bounding box sizes of all annotations to the detection
#             # window sizes of the detectors using _calc_detector_IoU()
#             IoU_with_DetWindow = self._calc_detector_IoU(annot)
            
#             # Assign the detector with the maximal IoU to the frame's 
#             # annotations
#             annot['detector_idx'] = IoU_with_DetWindow['detector_idx']

#         return None
            
    
#     def _calc_detector_IoU(self,annot:pd.DataFrame):
#         """
#         Find the detector (index in self.detectors) with the highest IoU
#         between the detector's detection window and the given bounding box

#         Parameters
#         ----------
#         annot : pd.DataFrame
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """
        
#         # Calculate the IoU of each annotation with the detection window dw
#         # of every detector. The detector with the higher IoU will then
#         # 'take care' of that annotation
#         columns = [i for i in range(len(self.detectors))] + ['detector_idx']
#         dw_IoU = pd.DataFrame(data=[],
#                               columns=columns,
#                               index = annot.index)
#         if len(dw_IoU) == 0:
#             return dw_IoU
        
#         i = 0
#         for detector in self.detectors:
        
#             # annot = self.IoU_with_dw(annot)
#             dw_w = detector.dwsize[0]
#             dw_h = detector.dwsize[1]
            
#             dw_IoU[i] = \
#                 annot.apply(lambda x: IoU(((int(x.xtl), int(x.ytl)),
#                                            (int(x.xbr), int(x.ybr))),
#                                           ((int(x.xtl), int(x.ytl)),
#                                            (int(x.xtl+dw_w), int(x.ytl+dw_h)))),
#                             axis=1)
            
#             i = i + 1
        
#         # Get the row-wise maximum, that's the index of the detector that's
#         # responsible for that annotation
#         dw_IoU['detector_idx'] = dw_IoU.idxmax(axis=1)
        
#         return dw_IoU
        
#     def fit(self,data,**kwargs):
#         """
#         Calls fit function of all detectors for all scales

#         Returns
#         -------
#         None.

#         """
        
#         # Apply leave one out cross validation in order to figure out the 
#         # optimal regularization parameter C
#         cross_val_C = kwargs.pop('cross_val_C',False)  
        
#         # Dictionary for storing results in
#         scores = {}
        
        
#         # ------------- Determine optimal weight parameter C ------------------
        
#         # If cross validation is to be performed, C must be varied and for each
#         # variation of C a leave one out cross validation must be performed
#         if cross_val_C:
            
#             # Candidates for C
#             C = [0.01,0.1,1,2,5,10]
                        
#             # Create all possible splits using split_data()
#             splits = self.split_data(list(data.keys()),
#                                      len(data.keys()))
                        
#             # Vary C
#             for c in C:
                
#                 scores[c] = {}
                
#                 # Change C of all classifier engines by initializing them with
#                 # new parameters
#                 d = 0
#                 new_detectors = []
#                 for det in self.detectors:
#                     det_dict = det.save()
#                     det_dict['classifier_engine']['C'] = c
#                     del det_dict['classifier_engine']['svm_py']
#                     new_detectors.append(build_detector(**det_dict))
                    
#                 self.detectors = new_detectors
                
#                 # After changing C, train all classifier_engines on all splits
#                 for s in splits.keys():
                    
#                     # Clear saved data before each split
#                     self.clear_data()
                    
#                     # Assign all training examples of this split to detectors 
#                     for ss in splits[s]['train']:
#                         self.assign_Xy(data[ss]['pos']['Xy'])
#                         self.assign_Xy(data[ss]['neg']['Xy'])
                        
#                     # Fit classifiers on assigned training data
#                     for detector in self.detectors:
#                         detector.fit(**kwargs)
                        
#                     # Evaluate classifier on validation data as well
#                     Xy_val = []
#                     for ss in splits[s]['val']:
#                         Xy_val.append(data[ss]['pos']['Xy'])
#                         Xy_val.append(data[ss]['neg']['Xy'])
                    
#                     Xy_val_merged = defaultdict(lambda: ([], []))
                    
#                     # Collect arrays
#                     for d in Xy_val:
#                         for key, (X, y) in d.items():
#                             Xy_val_merged[key][0].append(X)
#                             Xy_val_merged[key][1].append(y)
                    
#                     # Vertically stack the arrays for each key
#                     Xy_val_merged = {k: (np.vstack(Xs), np.vstack(ys)) for k, (Xs, ys) in Xy_val_merged.items()}
                    
#                     scores[c][s] = self.predict_Xy(Xy_val_merged)
                    
#                 # After evaluating the current setting of C on all possible 
#                 # splits, calculate an aggregated statistic for that setting
#                 for c in scores.keys():
#                     scores[c]['score_cross_val'] = \
#                         np.array([np.array(scores[c][s]).mean() for s in scores[c].keys()]).mean()
                        
#             # In the end find the C that yielded the best results
#             C_opt = \
#                 max(scores, key=lambda c: scores[c]['score_cross_val'])
                    
#             # Assign the optimal C to the detectors
#             d = 0
#             new_detectors = []
#             for det in self.detectors:
#                 det_dict = det.save()
#                 det_dict['classifier_engine']['C'] = C_opt
#                 del det_dict['classifier_engine']['svm_py']
#                 new_detectors.append(build_detector(**det_dict))
                
#             self.detectors = new_detectors
            

#         # ---------------------- Fit classifier -----------------------------
#         # must still be fitted on the whole training dataset
        
#         # Clear all previous data
#         self.clear_data()
        
#         # Assign training examples to detectors 
#         for d in data.keys():
#             self.assign_Xy(data[d]['pos']['Xy'])
#             self.assign_Xy(data[d]['neg']['Xy'])
            
#         # List for storing training results
#         scores['train'] = []
        
#         for detector in self.detectors:
#             scores['train'].append(detector.fit(**kwargs))
                
#         # If fit was successful, initialize cv2.htpa.MultiScaleDetector
#         if all(['cv_SingleScaleDetector' in detector.__dict__
#                 for detector in self.detectors]):
#             self._init_cv_multi_scale_detector()
            
#         # --------------- Optimize decision boundary -------------------------
        
#         # In the previous section the svm parameters have been trained on feature
#         # vectors of positive and negative examples to minimize the 
#         # squared_hinge L2 loss. However, the target criterion is the F1 score.
        
#         # Hence the decision boundary of the classifier is now tuned in order
#         # to maximize the F1 score
        
#         annot_mgr = AnnotMgr(self.tparray)
        
#         opt_values = []
        
#         for cvat_task in data.keys():
            
#             # Get annotated video (original video, pre-preocessed annotations)
#             annotated_video = data[cvat_task]['annotated_gt_video']
            
#             # Apply detector
#             video_pred = self.predict(annotated_video,
#                                       pre_process = True,
#                                       post_process = True)
            
#             # get annotation dataframes
#             annot_true = annotated_video.get_annotations()
#             annot_pred = video_pred.get_annotations()
            
#             # comapare prediction and ground truth
#             if len(annot_true)!=0:
#                 annot_images = annot_true['image_id'].unique()
#                 annot_pred = annot_pred.loc[annot_pred['image_id'].isin(annot_images)]
            
#             # Calculate the performance metrics for each dataset
#             pmetrics = annot_mgr.calc_pmetrics(annot_true,
#                                                annot_pred,
#                                                **kwargs)
            
#             # Find the optimal confidence_limit (=decision boundary) and the
#             # corresponding F1-score
#             try:
#                 idx_opt = pmetrics['metrics']['F1'].idxmax()
#                 F1_opt = pmetrics['metrics'].loc[idx_opt,'F1']
#                 boundary_opt = idx_opt[1]
                
#                 opt_values.append((boundary_opt,F1_opt))
#             except:
#                 print('Fix this more generally!')
                
#         print(opt_values)    
#         # The overall optimal threshold is taken to be the weighted average of  
#         # optimal thresholds for each dataset
#         decision_boundary = sum(s * f for s, f in opt_values) /\
#             sum(f for _, f in opt_values)
        
#         # Assign the optimal boundary to each of the detectors on the python
#         # side
#         for detector in self.detectors:
#             detector.classifier_engine.decision_boundary = decision_boundary
        
#         # In order to change the underlying C++ instance, re-initialize them
#         # now with the new decision boundary
#         self._init_cv_multi_scale_detector()
        
#         return scores
        

#     def split_data(self,
#                     datasets:list,
#                     N_splits:int):
#         """
#         Method for splitting imported datasets into training and testing
#         datasets

#         Parameters
#         ----------
#         N_splits : int
#             Number of folds as described here:
#             https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
#         # train_datasets : list of int
#         #     keys of the datasets in self.data to use for training
#         # test_datasets : list of int
#         #     keys of the datasets in self.data to use for testing
        
#         **kwargs : dict
#             DESCRIPTION.

#         Returns
#         -------
#         splits : TYPE
#             DESCRIPTION.

#         """
#         from hsdetect.train.procedures import TrueProposals
        
#         true_proposal_proc = TrueProposals()
        
#         return true_proposal_proc.split_data(datasets, N_splits)        
        
    
#     def prep_Xy(self,annot_video:AnnotatedVideo,
#                      **kwargs:dict)->tuple:
#         """
#         Directs proposals depending on their size to appropriate detector
#         and calls it's prep_Xy method

#         Parameters
#         ----------
#         frame_dict : dict
#             DESCRIPTION.
#         annot : pd.DataFrame
#             DESCRIPTION.
#         **kwargs : dict
#             DESCRIPTION.

#         Returns
#         -------
#         tuple
#             DESCRIPTION.

#         """
        
#         # SCHREIBE DIESE GEMSAMTE FUNKTION SO UM, DASS prep_Xy EBENFALLS 
#         # EIN AnnotatedVideo FORMAT BEKOMMT. DIE UMWANDLUNG IN dict ERFOLGT VIEL 
#         # ZU FRÜH.
        
#         # Create a dictionary containing the regression matrix X and 
#         # target vector y compiled from the annotated video
#         detector_Xy = {}
        
#         # First assign each annotation to the proper detector based on the 
#         # match between the annotations bounding box size and the size of 
#         # the detectors detection window
#         self._assign_annotations_to_subdetectors(annot_video)
        
#         # Call each detectors prep_Xy() method and pass the annotated video.
#         # The information which annotation is processed by which detector
#         # is stored in the column "detector_idx" of each frame's annotation
#         for i in range(len(self.detectors)):
            
#             Xy = self.detectors[i].prep_Xy(annot_video,
#                                            **kwargs)
            
#             # Add Xy to the respective detector's dictionary of feature and 
#             # target vectors
#             detector_Xy[i] = Xy
            
        
#         # # dw_IoU = self._calc_detector_IoU(annot)
        
        
#         # # frame_dict,annot = annot_video.get_annotated_framdict()

#         # if len(annot)!=0:
            
#         #     # If the detector has multiple sub-detectors, one needs to first 
#         #     # find out which one is responsible for which annotation
#         #     if len(self.detectors) > 1:
                
#         #         # raise Exception("detectors[i].prep_Xy needs to be updated to receive AnnotatedVideo format!")
                
#         #         # dw_IoU = self._calc_detector_IoU(annot)
#         #         # annot['detector_idx'] = dw_IoU['detector_idx']
                
#         #         # Pass the appropriate annotations to the prep_Xy() method of 
#         #         # each detector
#         #         for i in range(len(self.detectors)):
                    
#         #             detector_annot = annot.loc[annot['detector_idx']==i].copy()
        
#         #             if len(detector_annot) != 0:
#         #                 Xy = self.detectors[i].prep_Xy(frame_dict,
#         #                                                detector_annot,
#         #                                                **kwargs)
#         #                 detector_Xy[i] = Xy
                        
#         #     else:
                
#         #         for frame in annot_video.frames:
#         #             annot = frame.annotations.copy()
#         #             annot['detector_idx'] = 0
#         #             frame.annotations = annot
                
#         #         Xy = self.detectors[0].prep_Xy(annot_video,
#         #                                        **kwargs)
                
#         #         detector_Xy[0] = Xy
                
                
#         # else:
#         #     pass
        
#         return detector_Xy 
    
#     def assign_Xy(self,Xy:dict):
#         """
#         Assigns tuples (X,y) arranged in a dictionary to to proper detector

#         Parameters
#         ----------
#         Xy : dict
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """
        
#         if not isinstance(Xy,dict):
#             raise Exception('Xy is expected to be a dictionary!')
        
#         for d in Xy.keys():
#             self.detectors[d].Xy = Xy[d]
    
#     def predict_Xy(self,Xy:dict)->list:
#         """
#         Directs tuples (X,y) arranged in a dictionary towards the proper detector
#         and evaluates its classifier_engine on it

#         Parameters
#         ----------
#         Xy : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         list
#             DESCRIPTION.

#         """

#         if not isinstance(Xy,dict):
#             raise Exception('Xy is expected to be a dictionary!')
        

#         scores = []
        
#         for d in Xy.keys():
            
#             X = Xy[d][0]
#             y = Xy[d][1]
            
#             scores.append(self.detectors[d].classifier_engine.svm_py.score(X,y))
       
#         return scores
    
#     def predict(self,AnnotFrameOrVideo,**kwargs)->pd.DataFrame:
#         """
#         Directs proposals depending on their size to appropriate detector
#         and calls it's predict method.

#         Parameters
#         ----------
#         img : np.ndarray
#             Frame to apply detector to.
#         img_idx : int
#             Index of that frame.
#         video: pd.DataFrame
#             Video or sequence of frames as pd.DataFrame to apply detector to
#         pre_process: bool
#             If true, preprocessing engine is applied to each frame. If false, 
#             no preprocessing is applied.

#         Returns
#         -------
#         bboxes.

#         """
        
#         img = kwargs.pop('img',None)
#         img_idx = kwargs.pop('img_idx',None)
#         video = kwargs.pop('video',None)
        
#         pre_process = kwargs.pop('pre_process',True)
#         post_process = kwargs.pop('post_process',True)
        
        
#         if 'bbox_prop' in kwargs.keys():
#             Exception('Implement the possibility in the MultiScaleDetector' + \
#                       'in Cpp to pass proposals instead of applying the'  + \
#                           'proposal generator. Overload predict for that purpose')
        
#         # Check if single frame or video is provided
#         if isinstance(AnnotFrameOrVideo,AnnotatedFrame):
#             raise Exception("This part needs to be adapted to handle AnnotatedFrame!")
            
#             frame = AnnotFrameOrVideo
            
            
#             # Cast image to proper type just to be sure
#             img = img.astype(np.uint16)
            
#             # Apply detector to the single given frame
#             bboxes, img_pre  = self.cv_MultiScaleDetector.predict(img,img_idx,
#                                                                   pre_process,
#                                                                   post_process)
            
#             # Convert to pd.DataFrame
#             bboxes = pd.DataFrame(data = bboxes.get_content(),
#                                      columns = bboxes.columns)
                        
#             # Cast datatypes
#             bboxes = bboxes.astype(self.cv_datatypes)
            
#             # Set original index back
#             bboxes = bboxes.set_index('idx',drop=True)
            
#             # Name index "id" for compatiblity with all other classes
#             bboxes.index.rename('id',inplace=True)
            
#             return bboxes, img_pre
            
#         elif isinstance(AnnotFrameOrVideo,AnnotatedVideo):
            
#             # If a video is provided, loop over the whole video and apply
#             # detector generator frame-wise
#             video = AnnotFrameOrVideo
            
#             # Initialize a list in which to store the detections for each frame
#             predicted_frames = []
            
#             for frame in video.frames:
                
#                 img = frame.image.astype(np.uint16)
#                 idx = frame.index 
                
#                 # Call the get_proposal() method of the Cpp class
#                 df_bboxes, img_processed = \
#                     self.cv_MultiScaleDetector.process(img,
#                                                        idx,
#                                                        pre_process,
#                                                        post_process)
                
#                 # Convert to pd.DataFrame
#                 df_bboxes = pd.DataFrame(data = df_bboxes.get_content(),
#                                          columns = df_bboxes.columns)
                
#                 # Depending on which filter is used, the prediction might
#                 # not correspong to the current input frame but to a previous
#                 # frame
#                 if len(df_bboxes) != 0:
#                     idx_pred = int(df_bboxes.loc[0,'image_id'])
#                 else:
#                     idx_pred = int(idx_pred+1)
                
#                 # Convert to an AnnotatedFrame
#                 predicted_frames.append(AnnotatedFrame(idx_pred,
#                                                        img_processed,
#                                                        df_bboxes))
                 
#             # Reset the detector in order to delete buffered data
#             self.reset()
                
#             # Convert the result into an AnnotatedVideo
#             video_pred = AnnotatedVideo(predicted_frames)
            
#             return video_pred
    
#     def predict_bboxes(self,img:np.ndarray,bboxes:cv2.htpa.DataFrame,
#                        img_idx:int):
#         """
#         Same as predict(), but does not generate proposals but is given a 
#         DataFrame with bounding boxes that are passed to the detectors
        
#         Parameters
#         ----------
#         img : np.ndarray
#             Frame to apply detector to.
#         img_idx : int
#             Index of that frame.

#         Returns
#         -------
#         None.

#         """
        
        
#         # Cast image to proper type just to be sure
#         img = img.astype(np.uint16)
        
#         # Apply proposal generator to the single given frame
#         bboxes = self.cv_MultiScaleDetector.predict_bboxes(img,
#                                                            bboxes)
#         # Convert to pd.DataFrame
#         bboxes = pd.DataFrame(data = bboxes.get_content(),
#                                  columns = bboxes.columns)
        
#         # Add a column for the image index
#         bboxes['image_id'] = img_idx
        
#         # Cast datatypes
#         bboxes = bboxes.astype(self.cv_datatypes)
        
#         # Set original index
#         bboxes = bboxes.set_index('idx',drop=True)
        
#         return bboxes
    

#     def detect(self,idx,img)->pd.DataFrame:
#         """
#         Simply calls the Cpp backend detector. Meant for live scenarios

#         Parameters
#         ----------
#         img : np.ndarray
#             Frame to apply detector to.
#         idx : int
#             Index of that frame.


#         Returns
#         -------
#         bboxes.

#         """
        
        
#         pre_process = True
#         post_process = True

#         # Cast image to proper type just to be sure
#         img = img.astype(np.uint16)
        
#         # Apply detector to the single given frame
#         bboxes, img_pre  = self.cv_MultiScaleDetector.process(img,idx,
#                                                               pre_process,
#                                                               post_process)
            
#         # Convert to pd.DataFrame
#         bboxes = pd.DataFrame(data = bboxes.get_content(),
#                                  columns = bboxes.columns)
                    
#         # Cast datatypes
#         bboxes = bboxes.astype(self.cv_datatypes)
        
#         # Set original index back
#         bboxes = bboxes.set_index('idx',drop=True)
        
#         # Name index "id" for compatiblity with all other classes
#         bboxes.index.rename('id',inplace=True)
        
#         return bboxes, img_pre
        