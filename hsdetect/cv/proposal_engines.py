# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:11:20 2024

@author: rehmer
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

import hsdetect
# from hsdetect.cv import cpp_wrapper
from hsdetect.annot.annot import AnnotatedFrame, AnnotatedVideo
from hspy.tparray import TPArray
from hsdetect.cv import misc
import cv2

from hsdetect import Engine

# Marker identifying this .py as a module containing engines
IS_ENGINE_MODULE = True

def build_proposal_engine(**kwargs):
    name = kwargs.pop('name')
    if name == 'ProposalEngineEnsemble':
        return ProposalEngineEnsemble(**kwargs)
    elif name == 'PropGenWatershed_v1_CV':
        return PropGenWatershed_v1_CV(**kwargs)
    elif name == 'PropGenWatershed_v2_CV':
        return PropGenWatershed_v2_CV(**kwargs)
    elif name == 'Watershed_32x32_L1k9':
        return Watershed_32x32_L1k9(**kwargs)    
    elif name == 'PropGenSlidingWindowPyramid_CV':
        return PropGenSlidingWindowPyramid_CV(**kwargs)
    elif name == 'PropGenDiffImg':
        return PropGenDiffImg(**kwargs)
    elif name == 'PropGenGaussCont':
        return PropGenGaussCont(**kwargs)
    elif name == 'PropGenDiffplusDOG':
        return PropGenDiffplusDOG(**kwargs)
    elif name == 'BackSub':
        return BackSub(**kwargs)
    else:
        raise Exception('Proposal Generator ' + name + ' not known!')

import ast       
def get_classes():
    with open(__file__, 'r') as f:
        tree = ast.parse(f.read(), filename=__file__)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


class ProposalEngine(Engine):
    
    cv_datatypes = {'idx':'int32','xtl':'int32','ytl':'int32',
                    'xbr':'int32','ybr':'int32','w':'int32',
                    'h':'int32','score':'float64','track':'int32',
                    'strikes':'int32','label':'int32',
                    'IoU_max':'float64','IoU_idxmax':'int32',
                    'detector_idx':'int32','image_id':'int32'}
    
    def __init__(self,SensorType:int,**attr_dict):
        
        self.SensorType = SensorType
        
        self.n_padding = attr_dict.pop('n_padding',0)
        self.filter_size = attr_dict.pop('filter_size',False)
        self.min_size = attr_dict.pop('min_size',(0,0))
        self.max_size = attr_dict.pop('max_size',(160,120))
        
        # Default mask
        htpa = TPArray(SensorType = SensorType)
        mask = 255*np.ones(htpa._npsize,
                       np.uint8)
        
        self.mask = attr_dict.pop('mask',mask)
        
                
    @property
    def SensorType(self):
        return self._SensorType
    
    @SensorType.setter
    def SensorType(self,SensorType:int):
        self._SensorType = SensorType
        
    @property
    def n_padding(self):
        return self._n_padding
    
    @n_padding.setter
    def n_padding(self,n_padding:int):
        self._n_padding = n_padding

    @property
    def filter_size(self):
        return self._filter_size
    
    @filter_size.setter
    def filter_size(self,filter_size:bool):
        self._filter_size = filter_size
        
    @property
    def min_size(self):
        return self._min_size
    
    @min_size.setter
    def min_size(self,min_size:tuple):
        self._min_size = min_size
        
    @property
    def max_size(self):
        return self._max_size
    
    @max_size.setter
    def max_size(self,max_size:tuple):
        self._max_size = max_size
        
    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self,mask:np.ndarray):
        self._mask = mask
        
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
        
        frame = frame.astype(np.uint16)
        
        prop_cv = self.cv_proposal_engine.process(frame,
                                                  idx)
        
        # Convert to pd.DataFrame
        prop = misc.cv2_to_pd_DataFrame(prop_cv)
        
        # Name index "id" for compatiblity with all other classes
        prop.index.rename('id',inplace=True)
        
        # Create an AnnotatedFrame from the detection result
        annot_frame =  AnnotatedFrame(index = idx,
                                      image = frame,
                                      annotations = prop)
        
        # Prepare dictionary for return
        return_dict = {'frame':frame,
                       'idx_pre':None,
                       'annot_frame':  annot_frame}
        
        if isinstance(self, hsdetect.cv.proposal_engines.BackSub):
            return_dict['bg'] = \
                self.cv_proposal_engine._bg
            return_dict['bg_mask'] = \
                self.cv_proposal_engine._bg_mask
            return_dict['bg_diff'] = \
                self.cv_proposal_engine._bg_diff
        
        return return_dict  
    
    
    def _process_avideo(self,
                        video:AnnotatedVideo,
                        **kwargs):
        
        # Check if inputs have proper type
        if not isinstance(video,AnnotatedVideo):
            raise TypeError('video is not type AnnotatedVideo')

        # Get keyword arguments
        update_bg = kwargs.pop('update_bg',True)
        use_gt = kwargs.pop('use_gt',True)
        
        # Initialize list to store video with detection results in
        annotated_frames = []

        for frame in video:
            
            # Get frame and index
            img = frame.image
            img = img.astype(np.uint16)

            idx = frame.index
            
            # Call process_frame() on frame
            return_dict = \
                self._process_frame(img,idx)
            
            
            # Append returned AnnotatedFrame 
            annotated_frames.append(return_dict['annot_frame'])
            
            # Update the background using the current frame and its ground-
            # truth annotations
            if isinstance(self, hsdetect.cv.proposal_engines.BackSub):
                if update_bg:
                    # Use the given annotations background updating
                    if use_gt:
                        detections = frame.annotations
                    # Use the proposals for background updating
                    else:
                        detections = return_dict['annot_frame'].annotations
                        
                    self.update_bg(img,
                                   detections)
        
        # Reset memory components (if any) of proposal engine before
        # processing next video
        self.reset()       
                
        # In the end return an annotated video
        return AnnotatedVideo(frames = annotated_frames)   
        
    def has_bbox_regression(self):
        if self.cv_proposal_engine.has_bbox_regression():
            return True
        else:
            return False
        
    def save(self):
        '''
        Returns all non-private an non-builtin attributes of this class
        as a dictionary with the purpose of reloading this instance from the
        attribute dictionary. 

        Returns
        -------
        None.

        '''
        
        attr_dict = {}
        
        for name in dir(self):
            # get the attribute *on the class*, not the instance
            attr = getattr(self.__class__, name, None)
        
            # check if it's a property
            if isinstance(attr, property):
                # read the property's value from the instance
                attr_dict[name] = getattr(self, name)
                
        # Append name
        attr_dict['name'] = self.name
        
        
        return attr_dict
    
    def reset(self):
        """
        Call reset() method of cv_proposal_engine

        Returns
        -------
        None.

        """
        
        self.cv_proposal_engine.reset()

class ProposalEngineEnsemble(ProposalEngine):
    
    name = 'ProposalEngineEnsemble'
    
    def __init__(self,**attr_dict):
        
        # Call constructor of parent class
        SensorType = attr_dict.pop('SensorType',None)
        super(ProposalEngineEnsemble, self).__init__(SensorType,
                                                     **attr_dict)
        
        self.engine1 = attr_dict.pop('engine1',None)
        self.engine2 = attr_dict.pop('engine2',None)
        self.engine3 = attr_dict.pop('engine3',None)
        self.engine4 = attr_dict.pop('engine4',None)
        self.engine5 = attr_dict.pop('engine5',None)

        
        # Initialize cpp instance
        self.initialize_cpp_instance()

    @property
    def engine1(self):
        return self._engine1
    @engine1.setter
    def engine1(self,engine):
        self._engine1 = self._init_engine(engine)

    @property
    def engine2(self):
        return self._engine2
    @engine2.setter
    def engine2(self,engine):
        self._engine2 = self._init_engine(engine)
        
    @property
    def engine3(self):
        return self._engine3
    @engine3.setter
    def engine3(self,engine):
        self._engine3 = self._init_engine(engine)
        
    @property
    def engine4(self):
        return self._engine4
    @engine4.setter
    def engine4(self,engine):
        self._engine4 = self._init_engine(engine)
        
    @property
    def engine5(self):
        return self._engine5
    @engine5.setter
    def engine5(self,engine):
        self._engine5 = self._init_engine(engine)
    
    
    def initialize_cpp_instance(self):
        
        # Create a list with all engines
        engines = [self.engine1,
                   self.engine2,
                   self.engine3,
                   self.engine4,
                   self.engine5] 
        
        # Loop through all python instances, extract their C++ instances if
        # available and pass to C++
        cv_instances = []
        
        for eng in engines:
            if eng is not None:
                cv_instances.append(eng.cv_proposal_engine)
            else:
                cv_instances.append(None)
        
        self.cv_proposal_engine = cv2.htpa.ProposalEngineEnsemble(cv_instances[0],
                                                                     cv_instances[1],
                                                                     cv_instances[2],
                                                                     cv_instances[3],
                                                                     cv_instances[4])
        
    def _init_engine(self,engine):
        
        if engine is None:
            return None
        elif isinstance(engine, dict):
            return build_proposal_engine(**engine)
        elif isinstance(engine,ProposalEngine):
            return engine
        else:
            raise Exception('Proposal Generator not recognized.')
        
    def get_engine_proposals(self,**kwargs)->dict:
        '''
        Applies every proposal engine in the ensemble individually to a given
        single frame passed as np.ndarray or a video passed as pd.DataFrame
        
        Parameters
        ----------
        img : np.ndarray
            Frame to apply proposal generator to.
        img_idx : int
            Index of that frame.
        video: pd.DataFrame
            Video or sequence of frames as pd.DataFrame to apply proposal
            generator to

        Returns
        -------
        dict : TYPE
            DESCRIPTION.

        '''
        
        # Create a list with all engines
        engines = [self.engine1,
                   self.engine2,
                   self.engine3,
                   self.engine4,
                   self.engine5]
        
        # Create a dictionary to store individual results
        engine_proposals = {}
               
        for eng,e in zip(engines,range(len(engines))):
            
            if eng is not None:
                
                engine_proposals[e] = eng.process(**kwargs)
                
            else:
                
                engine_proposals[e] = None               
                
        return engine_proposals

    def train_bbox_regression(self,proposals:dict):
        """
        Calls the train_bbox_regression() method of the engines within the
        ensembles

        Parameters
        ----------
        proposals : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Create a list with all engines
        engines = [self.engine1,
                   self.engine2,
                   self.engine3,
                   self.engine4,
                   self.engine5]
        
        # Create a dictionary to store individual results
        engine_scores = {}
               
        
        for eng,e in zip(engines,range(len(engines))):
            
            if eng is not None:
                
                score = eng.train_bbox_regression(proposals[e])
                
                engine_scores[e] = score
                
            else:
                
                engine_scores[e] = None
        
        return engine_scores
    
    
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

          
class ProposalEngineBBoxRegression(ProposalEngine):
    
    def __init__(self,**attr_dict):
        
        # Call constructor of parent class
        SensorType = attr_dict.pop('SensorType',None)
        super(ProposalEngineBBoxRegression, self).__init__(SensorType,
                                                           **attr_dict)
        
        # Initialize class member variables
        self.bbox_IoU_lim = attr_dict.pop('bbox_IoU_lim',0.6)
        self.bbox_coef = \
            attr_dict.pop('bbox_coef',np.zeros((16,),np.float32)).flatten()
        self.bbox_inter = \
            attr_dict.pop('bbox_inter',np.zeros((4,),np.float32)).flatten()

    @property
    def bbox_IoU_lim(self):
        return self._bbox_IoU_lim
    @bbox_IoU_lim.setter
    def bbox_IoU_lim(self,bbox_IoU_lim:int):
        self._bbox_IoU_lim = bbox_IoU_lim

    @property
    def bbox_coef(self):
        return self._bbox_coef
    @bbox_coef.setter
    def bbox_coef(self,bbox_coef:int):
        self._bbox_coef = bbox_coef

    @property
    def bbox_inter(self):
        return self._bbox_inter
    @bbox_inter.setter
    def bbox_inter(self,bbox_inter:int):
        self._bbox_inter = bbox_inter
        
    def initialize_cpp_instance(self):
        raise Exception('This method must be overwritten by the child class.')

    def reset(self):
        
        self.initialize_cpp_instance()

    def train_bbox_regression(self,bboxes:pd.DataFrame):
        
        # Copy original dataframe to avoid changing it
        bboxes = bboxes.copy()
        
        # First, throw out all proposals below a certain IoU with a ground
        # truth box, i.e. we are neither interested nor capable to correct
        # proposals wich have nothing to do with the ground truth
        bboxes = bboxes.loc[bboxes['IoU_max'] > self.bbox_IoU_lim]
        
        # Next calculate the normalized offsets        
        bboxes['t_xtl'] = bboxes['d_xtl'] / bboxes['w']
        bboxes['t_ytl'] = bboxes['d_ytl'] / bboxes['h']
        bboxes['t_xbr'] = bboxes['d_xbr'] / bboxes['w']
        bboxes['t_ybr'] = bboxes['d_ybr'] / bboxes['h']

        # Extract inputs to the bbox regression model in the correct order
        X = bboxes[list(self.cv_proposal_engine.bbox_reg_in)]
        Y = bboxes[list(self.cv_proposal_engine.bbox_reg_out)]
        
        # Estimate model coefficients using sklearn
        bbox_reg_model = LinearRegression()
        
        try:
            bbox_reg_model.fit(X,Y)
        
            # Calculate a score 
            score = bbox_reg_model.score(X,Y)
            
            # Assign estimated coefficients to attribute of python wrapper
            self.bbox_coef = bbox_reg_model.coef_.astype(np.float32).flatten()
            self.bbox_inter = bbox_reg_model.intercept_.astype(np.float32).flatten()
            
            # Re-initialize cpp instance of proposal generator
            self.initialize_cpp_instance()
        
        except:
            score = 0
        
        return score
        

class PropGenDiffImg(ProposalEngineBBoxRegression):
    
    name = 'PropGenDiffImg'
    
    def __init__(self,**attr_dict):
                
        # Call constructor of parent class
        super(PropGenDiffImg, self).__init__(**attr_dict)
        
        # Initialize class member variables
        self.movement_thresh = attr_dict.pop('movement_thresh',2)
        self.connectivity = attr_dict.pop('connectivity',4)
        self.area_thresh = attr_dict.pop('area_thresh',3)
        
        # Initialize cpp instance
        self.initialize_cpp_instance()

    def initialize_cpp_instance(self):
        
        self.cv_proposal_engine = cv2.htpa.PropGenDiffImg(self.SensorType,
                                                             self.n_padding,
                                                             self.filter_size,
                                                             self.min_size,
                                                             self.max_size,
                                                             self.mask,
                                                             self.movement_thresh,
                                                             self.connectivity,
                                                             self.area_thresh,
                                                             self.bbox_coef,
                                                             self.bbox_inter)
    
    @property
    def movement_thresh(self):
        return self._movement_thresh
    
    @movement_thresh.setter
    def movement_thresh(self,thresh:int):
        self._movement_thresh = thresh

    @property
    def connectivity(self):
        return self._connectivity
    
    @connectivity.setter
    def connectivity(self,connectivity:int):
        self._connectivity = connectivity
        
    @property
    def area_thresh(self):
        return self._area_thresh
    
    @area_thresh.setter
    def area_thresh(self,area_thresh:int):
        self._area_thresh = area_thresh


class PropGenGaussCont(ProposalEngineBBoxRegression):
    
    name = 'PropGenGaussCont'
    
    def __init__(self,**attr_dict):
               
        # Call constructor of parent class
        super(PropGenGaussCont, self).__init__(**attr_dict)
        
        # Initialize class member variables
        self.sigma1 = attr_dict.pop('sigma1',1.1)
        self.sigma2 = attr_dict.pop('sigma2',0)
        self.kernel_size = attr_dict.pop('kernel_size',3)
        self.dog_thresh = attr_dict.pop('dog_thresh',3)
        self.intensity_weight = attr_dict.pop('intensity_weight',3)

        
        
        
        # Initialize cpp instance
        self.initialize_cpp_instance()

    def initialize_cpp_instance(self):
        
        self.cv_proposal_engine = cv2.htpa.PropGenGaussCont(self.SensorType,
                                                               self.n_padding,
                                                               self.filter_size,
                                                               self.min_size,
                                                               self.max_size,
                                                               self.mask,
                                                               self.sigma1,
                                                               self.sigma2,
                                                               self.kernel_size,
                                                               self.dog_thresh,
                                                               self.intensity_weight,
                                                               self.bbox_coef,
                                                               self.bbox_inter)
    
    @property
    def sigma1(self):
        return self._sigma1
    @sigma1.setter
    def sigma1(self,sigma1:float):
        self._sigma1 = sigma1

    @property
    def sigma2(self):
        return self._sigma2
    @sigma2.setter
    def sigma2(self,sigma2:float):
        self._sigma2 = sigma2
        
    @property
    def kernel_size(self):
        return self._kernel_size
    @kernel_size.setter
    def kernel_size(self,kernel_size:int):
        self._kernel_size = kernel_size    
    
    @property
    def dog_thresh(self):
        return self._dog_thresh
    @dog_thresh.setter
    def dog_thresh(self,dog_thresh:int):
        self._dog_thresh = dog_thresh      
        
    @property
    def intensity_weight(self):
        return self._intensity_weight
    @intensity_weight.setter
    def intensity_weight(self,intensity_weight:float):
        self._intensity_weight = intensity_weight      


class PropGenDiffplusDOG(ProposalEngineBBoxRegression):
    
    name = 'PropGenDiffplusDOG'
    
    def __init__(self,**attr_dict):
        
        # init_dict =  kwargs.pop('init_dict',None)
        
        # if init_dict is not None:
        #     attr_dict = init_dict
        # else:
        #     attr_dict = kwargs
        
        # Call constructor of parent class
        super(PropGenDiffplusDOG, self).__init__(**attr_dict)
        
        # Initialize class member variables
        self.movement_thresh = attr_dict.pop('movement_thresh',2)
        self.connectivity = attr_dict.pop('connectivity',4)
        self.area_thresh = attr_dict.pop('area_thresh',3)
        self.sigma1 = attr_dict.pop('sigma1',1.1)
        self.sigma2 = attr_dict.pop('sigma2',0)
        self.kernel_size = attr_dict.pop('kernel_size',3)
        self.dog_thresh = attr_dict.pop('dog_thresh',3)
        self.dog_region = attr_dict.pop('dog_region',3)
        self.intensity_weight = attr_dict.pop('intensity_weight',3)

        # Initialize cpp instance
        self.initialize_cpp_instance()

    def initialize_cpp_instance(self):
        
        self.cv_proposal_engine = cv2.htpa.PropGenDiffplusDOG(self.SensorType,
                                                                self.n_padding,
                                                                self.filter_size,
                                                                self.min_size,
                                                                self.max_size,
                                                                self.mask,
                                                                self.movement_thresh,
                                                                self.connectivity,
                                                                self.area_thresh,
                                                                self.sigma1,
                                                                self.sigma2,
                                                                self.kernel_size,
                                                                self.dog_thresh,
                                                                self.dog_region,
                                                                self.bbox_coef,
                                                                self.bbox_inter)
    
    @property
    def movement_thresh(self):
        return self._movement_thresh
    
    @movement_thresh.setter
    def movement_thresh(self,thresh:int):
        self._movement_thresh = thresh

    @property
    def connectivity(self):
        return self._connectivity
    
    @connectivity.setter
    def connectivity(self,connectivity:int):
        self._connectivity = connectivity
        
    @property
    def area_thresh(self):
        return self._area_thresh
    
    @area_thresh.setter
    def area_thresh(self,area_thresh:int):
        self._area_thresh = area_thresh
    
    @property
    def sigma1(self):
        return self._sigma1
    @sigma1.setter
    def sigma1(self,sigma1:float):
        self._sigma1 = sigma1

    @property
    def sigma2(self):
        return self._sigma2
    @sigma2.setter
    def sigma2(self,sigma2:float):
        self._sigma2 = sigma2
        
    @property
    def kernel_size(self):
        return self._kernel_size
    @kernel_size.setter
    def kernel_size(self,kernel_size:int):
        self._kernel_size = kernel_size    
    
    @property
    def dog_thresh(self):
        return self._dog_thresh
    @dog_thresh.setter
    def dog_thresh(self,dog_thresh:int):
        self._dog_thresh = dog_thresh
        
    @property
    def dog_region(self):
        return self._dog_thresh
    @dog_region.setter
    def dog_region(self,dog_region:int):
        self._dog_region = dog_region
        
class BackSub(ProposalEngineBBoxRegression):
    
    name = 'BackSub'
    
    def __init__(self,**attr_dict):
        
        # init_dict =  kwargs.pop('init_dict',None)
        
        # if init_dict is not None:
        #     attr_dict = init_dict
        # else:
        #     attr_dict = kwargs
        
        # Call constructor of parent class
        super(BackSub, self).__init__(**attr_dict)
        
        # Initialize class member variables
        self.diff_thresh = attr_dict.pop('diff_thresh',30)
        self.bg_update_thresh = attr_dict.pop('bg_update_thresh',8)
        self.decay = attr_dict.pop('decay',0.99)
        self.N_init = attr_dict.pop('N_init',5)


        # Initialize cpp instance
        self.initialize_cpp_instance()

    def initialize_cpp_instance(self):
        
        self.cv_proposal_engine = cv2.htpa.BackSub(self.SensorType,
                                                   self.n_padding,
                                                   self.filter_size,
                                                   self.min_size,
                                                   self.max_size,
                                                   self.mask,
                                                   self.diff_thresh,
                                                   self.bg_update_thresh,
                                                   self.decay,
                                                   self.N_init,
                                                   self.bbox_coef,
                                                   self.bbox_inter)
        # print('Remove outcommented bg_update_thresh!')
    
    @property
    def diff_thresh(self):
        return self._diff_thresh
    
    @diff_thresh.setter
    def diff_thresh(self,diff_thresh:int):
        self._diff_thresh = diff_thresh

    @property
    def bg_update_thresh(self):
        return self._bg_update_thresh
    
    @bg_update_thresh.setter
    def bg_update_thresh(self,bg_update_thresh:int):
        self._bg_update_thresh = bg_update_thresh

    @property
    def decay(self):
        return self._decay
    
    @decay.setter
    def decay(self,decay:float):
        self._decay = decay
        
    @property
    def N_init(self):
        return self._N_init
    
    @N_init.setter
    def N_init(self,N_init:int):
        self._N_init = N_init
        
    def set_initial_bg(self, bg : np.ndarray , bg_mask : np.ndarray):
        """
        

        Parameters
        ----------
        bg : np.ndarray
            DESCRIPTION.
        bg_mask : np.ndarray
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Type checks 
        if not isinstance(bg,np.ndarray) or not (bg.dtype == np.uint16):
            raise TypeError(f'bg is type {bg.dtype} instead of {np.uint16}')

        if not isinstance(bg_mask,np.ndarray) or not (bg_mask.dtype == np.uint8):
            raise TypeError(f'bg_mask is type {bg_mask.dtype} instead of {np.uint8}')
            
        # Call method of cpp instance
        self.cv_proposal_engine.set_initial_bg(bg,
                                               bg_mask)
        
    def update_bg(self, frame : np.ndarray , detections : pd.DataFrame):
        """
        

        Parameters
        ----------
        bg : np.ndarray
            DESCRIPTION.
        bg_mask : np.ndarray
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Type checks 
        if not isinstance(frame,np.ndarray) or not (frame.dtype == np.uint16):
            raise TypeError(f'frame is type {frame.dtype} instead of {np.uint16}')

        if not isinstance(detections,pd.DataFrame):
            raise TypeError(f'detections is type {type(detections)} instead of {pd.DataFrame}')
        
        # Convert pd.DataFrame to cpp equivalent
        cv2_htpa_df = misc.pd_to_cv2_htpa_DataFrame(detections)
        
        # Call method of cpp instance
        self.cv_proposal_engine.update_bg(frame,
                                          cv2_htpa_df)
        

    def compile_initial_background_avg(self,
                                       avideo : AnnotatedVideo) -> (np.ndarray,np.ndarray):
        
        
        
        EXPAND_PX = 2
        first_frame = avideo.frames[0].image.astype(np.float64).copy()
        H, W = first_frame.shape[:2]
        first_annot = avideo.frames[0].annotations
        
        if not first_annot.empty:
            boxes = first_annot[["xtl", "ytl", "xbr", "ybr"]].copy()
            boxes["xtl"] = (boxes["xtl"] - EXPAND_PX).clip(lower=0)
            boxes["ytl"] = (boxes["ytl"] - EXPAND_PX).clip(lower=0)
            boxes["xbr"] = (boxes["xbr"] + EXPAND_PX).clip(upper=W - 1)
            boxes["ybr"] = (boxes["ybr"] + EXPAND_PX).clip(upper=H - 1)
            M_init = np.zeros((H, W), dtype=np.float64)
            
            for _, box in boxes.iterrows():
                x1, y1 = int(box["xtl"]), int(box["ytl"])
                x2, y2 = int(box["xbr"]), int(box["ybr"])
                M_init[y1:y2 + 1, x1:x2 + 1] = 1.0
                
            first_frame[M_init == 1] = np.nan
    
        nan_mask = np.isnan(first_frame)
        accumulator = np.where(nan_mask, np.nan, first_frame)   # (H, W, C)
        count = np.where(nan_mask, 0.0, 1.0)                    # (H, W)
    
        for aframe in avideo.frames:
            frame = aframe.image.astype(np.float64)
            annot = aframe.annotations
            if not annot.empty:
                boxes = annot[["xtl", "ytl", "xbr", "ybr"]].copy()
                boxes["xtl"] = (boxes["xtl"] - EXPAND_PX).clip(lower=0)
                boxes["ytl"] = (boxes["ytl"] - EXPAND_PX).clip(lower=0)
                boxes["xbr"] = (boxes["xbr"] + EXPAND_PX).clip(upper=W - 1)
                boxes["ybr"] = (boxes["ybr"] + EXPAND_PX).clip(upper=H - 1)
            else:
                continue
    
            M = np.zeros((H, W), dtype=np.float64)
            for _, box in boxes.iterrows():
                x1, y1 = int(box["xtl"]), int(box["ytl"])
                x2, y2 = int(box["xbr"]), int(box["ybr"])
                M[y1:y2 + 1, x1:x2 + 1] = 1.0
    
            bg_mask = (1.0 - M)                                         # (H, W)
    
            # Pixels that are still NaN in accumulator but visible in this frame:
            # initialise them directly rather than adding to NaN
            first_obs = np.isnan(accumulator) & (bg_mask == 1.0)  # (H, W)
            accumulator = np.where(
                first_obs,
                frame,
                accumulator + frame * bg_mask
            )
            count += bg_mask
        
        # Calculate background by averaging
        bg_float = np.where(count > 0, accumulator / count, 0)
        
        bg_uint16 = np.round(bg_float,decimals = 0).astype(np.uint16)
        
        # Compile a mask from the count, which is 255 everywhere where the count is
        # not equal to zero, and 0 otherwise
        bg_mask = np.where(count != 0, 1, 0).astype(np.uint8)*255
        
        return bg_uint16, bg_mask    


class PropGenWatershed_v1_CV(ProposalEngine):
    
    name = 'PropGenWatershed_v1_CV'
    
    def __init__(self,**kwargs):
        """
        

        Parameters
        ----------
        w : int
            DESCRIPTION.
        h : int
            DESCRIPTION.
        otsu_q : float
            DESCRIPTION.
        otsu_res : int
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        init_dict =  kwargs.pop('init_dict',None)
        
        if init_dict is not None:
            attr_dict = init_dict
        else:
            attr_dict = kwargs
        
        self.SensorType = attr_dict.pop('SensorType',None)
        
        self.bbsize_min = attr_dict.pop('bbsize_min',(0,0))
        self.bbsize_max = attr_dict.pop('bbsize_max',(200,200))
        
        self.otsu_q = attr_dict.pop('otsu_q',None)
        self.otsu_res = attr_dict.pop('otsu_res',None)
                
        cv_tparray = cv2.htpa.TPArray(self.SensorType)
                
        self.cv_proposal_engine = cv2.htpa.PropGenWatershed_v1(cv_tparray,
                                                                  cv_tparray.get_radiometric_mask(),
                                                                  self.otsu_q,
                                                                  self.otsu_res,
                                                                  self.bbsize_min,
                                                                  self.bbsize_max)
        
        self.cv_datatypes = {'idx':'int32','xtl':'int32','ytl':'int32',
                             'xbr':'int32','ybr':'int32','w':'int32',
                             'h':'int32','score':'float64','track':'int32',
                             'strikes':'int32','label':'int32',
                             'IoU_max':'float64','IoU_idxmax':'int32',
                             'detector_idx':'int32','image_id':'int32'}
        

    

    
    def reset(self):
        
        cv_tparray = cv2.htpa.TPArray(self.SensorType)
                
        self.cv_proposal_engine = cv2.htpa.PropGenWatershed_v1(cv_tparray,
                                                                  cv_tparray.get_radiometric_mask(),
                                                                  self.otsu_q,
                                                                  self.otsu_res,
                                                                  self.bbsize_min,
                                                                  self.bbsize_max)
    
class PropGenWatershed_v2_CV(PropGenWatershed_v1_CV):
    
    name = 'PropGenWatershed_v2_CV'
    
    def __init__(self,**kwargs):
        """
        

        Parameters
        ----------
        w : int
            DESCRIPTION.
        h : int
            DESCRIPTION.
        otsu_q : float
            DESCRIPTION.
        otsu_res : int
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        init_dict =  kwargs.pop('init_dict',None)
        
        if init_dict is not None:
            attr_dict = init_dict
        else:
            attr_dict = kwargs
        
        self.SensorType = attr_dict.pop('SensorType',None)
        
        self.bbsize_min = attr_dict.pop('bbsize_min',(0,0))
        self.bbsize_max = attr_dict.pop('bbsize_max',(200,200))
        
        self.otsu_q = attr_dict.pop('otsu_q',None)
        self.otsu_res = attr_dict.pop('otsu_res',None)
        
        self.laplace_kernel = attr_dict.pop('laplace_kernel',3)
        
        self.closing = attr_dict.pop('closing',None)
        self.closing_kernel_size = attr_dict.pop('closing_kernel_size',None)
        
        cv_tparray = cv2.htpa.TPArray(self.SensorType)
                
        self.cv_proposal_engine = cv2.htpa.PropGenWatershed_v2(cv_tparray,
                                                                  cv_tparray.get_radiometric_mask(),
                                                                  self.otsu_q,
                                                                  self.otsu_res,
                                                                  self.bbsize_min,
                                                                  self.bbsize_max,
                                                                  self.laplace_kernel,
                                                                  self.closing,
                                                                  self.closing_kernel_size)
        
        self.cv_datatypes = {'idx':'int32','xtl':'int32','ytl':'int32',
                             'xbr':'int32','ybr':'int32','w':'int32',
                             'h':'int32','score':'float64','track':'int32',
                             'strikes':'int32','label':'int32',
                             'IoU_max':'float64','IoU_idxmax':'int32',
                             'detector_idx':'int32','image_id':'int32'}
    
    def reset(self):
        
        cv_tparray = cv2.htpa.TPArray(self.SensorType)
        
        self.cv_proposal_engine = cv2.htpa.PropGenWatershed_v2(cv_tparray,
                                                                  cv_tparray.get_radiometric_mask(),
                                                                  self.otsu_q,
                                                                  self.otsu_res,
                                                                  self.bbsize_min,
                                                                  self.bbsize_max,
                                                                  self.laplace_kernel,
                                                                  self.closing,
                                                                  self.closing_kernel_size)

class Watershed_32x32_L1k9(PropGenWatershed_v1_CV):
    
    name = 'Watershed_32x32_L1k9'
    
    def __init__(self,**kwargs):
        """
        

        Parameters
        ----------
        SensorType : str
            DESCRIPTION.
        bbsize_min : tuple
            DESCRIPTION.
        bbsize_max : tuple
            DESCRIPTION.
        otsu_q : float
            DESCRIPTION.
        otsu_res : int
            DESCRIPTION.
        th_boost : int
            DESCRIPTION.
        extend_bboxes : bool
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        init_dict =  kwargs.pop('init_dict',None)
        
        if init_dict is not None:
            attr_dict = init_dict
        else:
            attr_dict = kwargs
        
        self.SensorType = attr_dict.pop('SensorType',None)
        
        self.bbsize_min = attr_dict.pop('bbsize_min',(0,0))
        self.bbsize_max = attr_dict.pop('bbsize_max',(200,200))
        
        self.otsu_q = attr_dict.pop('otsu_q',None)
        self.otsu_res = attr_dict.pop('otsu_res',None)
        
        self.th_boost = attr_dict.pop('th_boost',None)
        self.extend_bboxes = attr_dict.pop('extend_bboxes',None)
        
        self.bbox_IoU_lim = kwargs.pop('bbox_IoU_lim',0.6)
        self.bbox_coef = attr_dict.pop('bbox_coef',np.zeros((16,),np.float32)).flatten()
        self.bbox_inter = attr_dict.pop('bbox_inter',np.zeros((4,),np.float32)).flatten()
        
        self.initialize_cpp_instance()

        
        self.cv_datatypes = {'idx':'int32','xtl':'int32','ytl':'int32',
                             'xbr':'int32','ybr':'int32','w':'int32',
                             'h':'int32','score':'float64','track':'int32',
                             'strikes':'int32','label':'int32',
                             'IoU_max':'float64','IoU_idxmax':'int32',
                             'detector_idx':'int32','image_id':'int32'}

    def initialize_cpp_instance(self):
        
        cv_tparray = cv2.htpa.TPArray(self.SensorType)
                
        self.cv_proposal_engine = cv2.htpa.Watershed_32x32_L1k9(cv_tparray,
                                                                   cv_tparray.get_radiometric_mask(),
                                                                   self.otsu_q,
                                                                   self.otsu_res,
                                                                   self.th_boost,
                                                                   self.extend_bboxes,
                                                                   self.bbsize_min,
                                                                   self.bbsize_max,
                                                                   self.bbox_coef.flatten(),
                                                                   self.bbox_inter.flatten())

    def reset(self):
        
        self.initialize_cpp_instance()

    def train_bbox_regression(self,bboxes:pd.DataFrame):
        
        # Copy original dataframe to avoid changing it
        bboxes = bboxes.copy()
        
        # First, throw out all proposals below a certain IoU with a ground
        # truth box, i.e. we are neither interested nor capable to correct
        # proposals wich have nothing to do with the ground truth
        bboxes = bboxes.loc[bboxes['IoU_max'] > self.bbox_IoU_lim]
        
        # Next calculate the normalized offsets        
        bboxes['t_xtl'] = bboxes['d_xtl'] / bboxes['w']
        bboxes['t_ytl'] = bboxes['d_ytl'] / bboxes['h']
        bboxes['t_xbr'] = bboxes['d_xbr'] / bboxes['w']
        bboxes['t_ybr'] = bboxes['d_ybr'] / bboxes['h']

        # Extract inputs to the bbox regression model in the correct order
        X = bboxes[list(self.cv_proposal_engine.bbox_reg_in)]
        Y = bboxes[list(self.cv_proposal_engine.bbox_reg_out)]
        
        # Estimate model coefficients using sklearn
        bbox_reg_model = LinearRegression()
        bbox_reg_model.fit(X,Y)
        bbox_reg_model.score(X,Y)
        
        # Assign estimated coefficients to attribute of python wrapper
        self.bbox_coef = bbox_reg_model.coef_.astype(np.float32)
        self.bbox_inter = bbox_reg_model.intercept_.astype(np.float32)
        
        # Re-initialize cpp instance of proposal generator
        self.initialize_cpp_instance()
    
    


        

class PropGenSlidingWindowPyramid_CV(ProposalEngine):
    
    name = 'PropGenSlidingWindowPyramid_CV'
    
    def __init__(self,**kwargs):
        """
        

        Parameters
        ----------
        w : int
            DESCRIPTION.
        h : int
            DESCRIPTION.
        otsu_q : float
            DESCRIPTION.
        otsu_res : int
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        init_dict =  kwargs.pop('init_dict',None)
        
        if init_dict is not None:
            attr_dict = init_dict
        else:
            attr_dict = kwargs
        
        self.SensorType = attr_dict.pop('SensorType',None)
        self.otsu_q = attr_dict.pop('otsu_q',None)
        self.otsu_res = attr_dict.pop('otsu_res',None)
        self.window_size = attr_dict.pop('window_size',None)
        self.step_size = attr_dict.pop('step_size',None)
        self.scale_factor = attr_dict.pop('scale_factor',None)
        self.levels = attr_dict.pop('levels',None)
        
        cv_tparray = cv2.htpa.TPArray(self.SensorType)
                
        self.cv_proposal_engine = cv2.htpa.PropGenSlidingWindowPyramid(cv_tparray,
                                                                          cv_tparray.get_radiometric_mask(),
                                                                          self.otsu_q,
                                                                          self.otsu_res,
                                                                          self.window_size,
                                                                          self.step_size,
                                                                          self.scale_factor,
                                                                          self.levels)
        
        self.cv_datatypes = {'idx':'int32','xtl':'int32','ytl':'int32',
                             'xbr':'int32','ybr':'int32','w':'int32',
                             'h':'int32','score':'float64','track':'int32',
                             'strikes':'int32','label':'int32',
                             'IoU_max':'float64','IoU_idxmax':'int32',
                             'detector_idx':'int32','image_id':'int32'}
        

        
    def reset(self):
        
        cv_tparray = cv2.htpa.TPArray(self.SensorType)
                
        self.cv_proposal_engine = cv2.htpa.PropGenSlidingWindowPyramid(cv_tparray,
                                                                          cv_tparray.get_radiometric_mask(),
                                                                          self.otsu_q,
                                                                          self.otsu_res,
                                                                          self.window_size,
                                                                          self.step_size,
                                                                          self.scale_factor,
                                                                          self.levels)