# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:17:19 2023

@author: Rehmer
"""

import cv2
import numpy as np

from hspytools.tparray import TPArray
import pandas as pd

from .misc import pd_to_cv2_htpa_DataFrame, pd_to_cv2_htpa_row_detect

# from matplotlib import pyplot as plt
# import pickle as pkl
# from pathlib import Path
# from .gradient import Gradient
# from hspytools.cv.filters import Gradient
# from .preproc_engines import SlidingWindow

# import time
# import copy


def build_feature_engine(**kwargs):
    name = kwargs.pop('name')
    if name == 'FeatureEngineEnsemble':
        return FeatureEngineEnsemble(**kwargs)
    elif name == 'FeatEngHOG':
        return FeatEngHOG(**kwargs)
    elif name == 'FeatEngAnnot':
        return FeatEngAnnot(**kwargs)
    elif name == 'FeatEngDetWinIoU':
        return FeatEngDetWinIoU(**kwargs)
    elif name == 'FeatEngHist':
        return FeatEngHist(**kwargs)
    elif name == 'FeatEngStat':
        return FeatEngStat(**kwargs)
    else:
        raise TypeError('Feature Engine ' + name + ' not known!')

import ast       
def get_classes():
    with open(__file__, 'r') as f:
        tree = ast.parse(f.read(), filename=__file__)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

class FeatureEngine():
    
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
        
    def get_features(self,
                     img:np.ndarray,
                     mask:np.ndarray,
                     bboxes:pd.DataFrame | pd.Series | cv2.htpa.DataFrame)->np.ndarray:
        
        # Cast img to proper type
        img = img.astype(np.uint16)
        
        # Check type of annotation and cast if necessary
        if not isinstance(bboxes, cv2.htpa.DataFrame):
            bboxes_cpp = pd_to_cv2_htpa_DataFrame(bboxes)
        else:
            bboxes_cpp = bboxes

        # Apply process() method of Cpp class FeatEngHOG
        X_feat = self.cv_feature_engine.process(img,
                                                mask,
                                                bboxes_cpp)
        
                
        return X_feat
    
    def get_features_roi(self,
                         roi:np.ndarray,
                         roi_mask:np.ndarray,
                         roi_annot:pd.DataFrame | pd.Series | cv2.htpa.DataFrame)->np.ndarray:
        
        if not roi.dtype == np.uint16:
            raise TypeError('roi is not type uint16')
            
            if not roi_mask.dtype == np.uint8:
                raise TypeError('roi_mask is not type uint8')
        
        if not (isinstance(roi_annot,pd.DataFrame) or
                isinstance(roi_annot,pd.Series) or
                isinstance(roi_annot,cv2.htpa.row_detect)):
            raise TypeError('roi has wrong type')
        
        # Check type of annotation and cast if necessary
        if not isinstance(roi_annot, cv2.htpa.DataFrame):
            bboxes_cv = pd_to_cv2_htpa_row_detect(roi_annot)
        else:
            bboxes_cv = roi_annot
            
        # Apply process() method of Cpp class FeatEngHOG
        x_feat = self.cv_feature_engine.process_roi(roi,
                                                    roi_mask,
                                                    bboxes_cv)
        
        return x_feat
        
    def save(self):
        '''
        Returns all non-private an non-builtin attributes of this class
        as a dictionary with the purpose of reloading this instance from the
        attribute dictionary. 

        Returns
        -------
        None.

        '''
        
        
        
        # Get all names of properties of the instance by doing
        properties = []
        for name in dir(self):
            
            # Get the attribute:
            attr = getattr(self.__class__, name, None)
            
            # Check if it is a property of the class (including base classes)
            if isinstance(attr, property):
            
                try:
                    # Check if it is a property of the isntance
                    getattr(self, name)
                    is_property = True
                    
                except:
                    is_property = False
            else:
                is_property = False
                
            if is_property:
                properties.append(name)

                
        # Loop through all properties and add them to the attribute dictionary
        # according to their type
        attr_dict = {}
        for name in properties:
            
            # get the attribute, i.e. its value
            prop = getattr(self, name, None)
    
            # Check if the property has a save() method itself
            save_method = getattr(prop, "save", None)
                
            # If the save method is callable, execute it
            if callable(save_method):
                # If its a callable save method, call it
                attr_dict[name] = {}
                attr_dict[name] = save_method()
            else:
                attr_dict[name] = prop

                
        # Append name, which is not a property
        attr_dict['name'] = self.name
        
        return attr_dict
    
class FeatureEngineEnsemble(FeatureEngine):
    
    name = 'FeatureEngineEnsemble'
    
    def __init__(self,**attr_dict):
        
        # Call constructor of parent class
        SensorType = attr_dict.pop('SensorType',None)
        super(FeatureEngineEnsemble, self).__init__(SensorType,
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
                cv_instances.append(eng.cv_feature_engine)
            else:
                cv_instances.append(None)
        
        self.cv_feature_engine = cv2.htpa.FeatureEngineEnsemble(cv_instances[0],
                                                                cv_instances[1],
                                                                cv_instances[2],
                                                                cv_instances[3],
                                                                cv_instances[4])
        
    def _init_engine(self,engine):
        
        if engine is None:
            return None
        elif isinstance(engine, dict):
            return build_feature_engine(**engine)
        elif isinstance(engine,FeatureEngine):
            return engine
        else:
            raise Exception('Feature Engine not recognized.')
        
    def get_engine_features(self,**kwargs)->dict:
        '''
        Applies every feature engine in the ensemble individually to a given
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
        engine_features = {}
               
        for eng,e in zip(engines,range(len(engines))):
            
            if eng is not None:
                
                engine_features[e] = eng.get_features(**kwargs)
                
            else:
                
                engine_features[e] = None               
                
        return engine_features

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

class FeatEngHOG(FeatureEngine):
    
    name = 'FeatEngHOG'
    
    def __init__(self,**attr_dict):
        
        # Call constructor of parent class
        super(FeatEngHOG, self).__init__(attr_dict['SensorType']) 
                
        self.dw_size = attr_dict.pop('dw_size',(6,12))   
        self.cell_size = attr_dict.pop('cell_size',2)                             # size of quadratic cell for which histogram is calculated
        self.block_size = attr_dict.pop('block_size',4)                           # size of quadratic block over which cell-histograms are normalized. Must be multiple of cell_size
        self.block_stride = attr_dict.pop('block_stride',2)                       # increment by which block is shifted during block normalization
        self.n_bins  = attr_dict.pop('n_bins',18)
        self.signedGradient  = attr_dict.pop('signedGradient',True)
        self.n_padding  = attr_dict.pop('n_padding',0)
        
        # Initialize FeatEngHOG from OpenCV
        self.initialize_cpp_instance()
        
    @property
    def dw_size(self):
        return self._dw_size
    @dw_size.setter
    def dw_size(self,dw_size:int):
        self._dw_size = dw_size

    @property
    def cell_size(self):
        return self._cell_size
    @cell_size.setter
    def cell_size(self,cell_size:int):
        self._cell_size = cell_size
        
    @property
    def block_size(self):
        return self._block_size
    @block_size.setter
    def block_size(self,block_size:int):
        self._block_size = block_size

    @property
    def block_stride(self):
        return self._block_stride
    @block_stride.setter
    def block_stride(self,block_stride:int):
        self._block_stride = block_stride

    @property
    def n_bins(self):
        return self._n_bins
    @n_bins.setter
    def n_bins(self,n_bins:int):
        self._n_bins = n_bins
        
    @property
    def signedGradient(self):
        return self._signedGradient
    @signedGradient.setter
    def signedGradient(self,signedGradient:bool):
        self._signedGradient = signedGradient

    @property
    def n_padding(self):
        return self._n_padding
    @n_padding.setter
    def n_padding(self,n_padding:int):
        self._n_padding = n_padding
        
    def initialize_cpp_instance(self):
        
        self.cv_feature_engine = cv2.htpa.FeatEngHOG(self.SensorType,
                                                     self.dw_size,
                                                     self.cell_size,
                                                     self.block_size,
                                                     self.block_stride,
                                                     self.n_bins,
                                                     self.signedGradient,
                                                     self.n_padding)

class FeatEngAnnot(FeatureEngine):
    
    name = 'FeatEngAnnot'
    
    def __init__(self,**attr_dict):
        
        # Call constructor of parent class
        super(FeatEngAnnot, self).__init__(attr_dict['SensorType']) 
        
        self.xtl = attr_dict.pop('xtl',False)   
        self.ytl = attr_dict.pop('ytl',False)   
        self.xbr = attr_dict.pop('xbr',False)   
        self.ybr = attr_dict.pop('ybr',False)   
        self.w = attr_dict.pop('w',False)   
        self.h = attr_dict.pop('h',False)   
        
        # Initialize FeatEngAnnot from OpenCV
        self.initialize_cpp_instance()
        
    @property
    def xtl(self):
        return self._xtl
    @xtl.setter
    def xtl(self,xtl:bool):
        self._xtl = xtl

    @property
    def ytl(self):
        return self._ytl
    @ytl.setter
    def ytl(self,ytl:bool):
        self._ytl = ytl
        
    @property
    def xbr(self):
        return self._xbr
    @xbr.setter
    def xbr(self,xbr:bool):
        self._xbr = xbr
        
    @property
    def ybr(self):
        return self._ybr
    @ybr.setter
    def ybr(self,ybr:bool):
        self._ybr = ybr
        
    @property
    def w(self):
        return self._w
    @w.setter
    def w(self,w:bool):
        self._w = w
       
    @property
    def h(self):
        return self._w
    @h.setter
    def h(self,h:bool):
        self._h = h
        
    def initialize_cpp_instance(self):
        
        self.cv_feature_engine = cv2.htpa.FeatEngAnnot(self.xtl,
                                                       self.ytl,
                                                       self.xbr,
                                                       self.ybr,
                                                       self.w,
                                                       self.h)
class FeatEngDetWinIoU(FeatureEngine):
    
    name = 'FeatEngDetWinIoU'
    
    def __init__(self,**attr_dict):
        
        # Call constructor of parent class
        super(FeatEngDetWinIoU, self).__init__(attr_dict['SensorType']) 
        
        self.DetWinSize = attr_dict.pop('DetWinSize',(0,0))
        
        # Initialize FeatEngDetWinIoU from OpenCV
        self.initialize_cpp_instance()
        
    @property
    def DetWinSize(self):
        return self._DetWinSize
    @DetWinSize.setter
    def DetWinSize(self,DetWinSize:tuple):
        self._DetWinSize = DetWinSize
        
    def initialize_cpp_instance(self):
        
        self.cv_feature_engine = cv2.htpa.FeatEngDetWinIoU(self.DetWinSize)        
        
class FeatEngHist(FeatureEngine):
    
    name = 'FeatEngHist'
    
    def __init__(self,**attr_dict):
        
        # Call constructor of parent class
        super(FeatEngHist, self).__init__(attr_dict['SensorType']) 
    
        self.N = attr_dict.pop('N',0)
        self.dKStart = attr_dict.pop('dKStart',0)
        self.dKEnd = attr_dict.pop('dKEnd',0)
        self.RoINorm = attr_dict.pop('RoINorm',0)
        self.HistNorm = attr_dict.pop('HistNorm',0)
        
        # Initialize FeatEngDetWinIoU from OpenCV
        self.initialize_cpp_instance()

    @property
    def N(self):
        return self._N
    @N.setter
    def N(self,N:int):
        self._N = N
               
    @property
    def dKStart(self):
        return self._dKStart
    @dKStart.setter
    def dKStart(self,dKStart:int):
        self._dKStart = dKStart
        
    @property
    def dKEnd(self):
        return self._dKEnd
    @dKEnd.setter
    def dKEnd(self,dKEnd:int):
        self._dKEnd = dKEnd
        
    @property
    def RoINorm(self):
        return self._RoINorm
    @RoINorm.setter
    def RoINorm(self,RoINorm:bool):
        self._RoINorm = RoINorm
        
    @property
    def HistNorm(self):
        return self._HistNorm
    @HistNorm.setter
    def HistNorm(self,HistNorm:bool):
        self._HistNorm = HistNorm
        
    def initialize_cpp_instance(self):
        
        self.cv_feature_engine = cv2.htpa.FeatEngHist(N = self.N,
                                                      dKStart = self.dKStart,
                                                      dKEnd = self.dKEnd,
                                                      RoINorm = self.RoINorm,
                                                      HistNorm = self.HistNorm)         

class FeatEngStat(FeatureEngine):
    
    name = 'FeatEngStat'
    
    def __init__(self,**attr_dict):
        
        # Call constructor of parent class
        super(FeatEngStat, self).__init__(attr_dict['SensorType']) 
    
        self.pmin = attr_dict.pop('pmin',0)
        self.pmax = attr_dict.pop('pmax',0)
        self.pcenter = attr_dict.pop('pcenter',0)
        self.peccentr = attr_dict.pop('peccentr',0)
        self.pelong = attr_dict.pop('pelong',0)
        self.piso = attr_dict.pop('piso',0)
        
        # Initialize FeatEngDetWinIoU from OpenCV
        self.initialize_cpp_instance()
        

    @property
    def pmin(self):
        return self._pmin
    @pmin.setter
    def pmin(self, pmin: bool):
        self._pmin = pmin
    
    
    @property
    def pmax(self):
        return self._pmax  
    @pmax.setter
    def pmax(self, pmax: bool):
        self._pmax = pmax
    
    
    @property
    def pcenter(self):
        return self._pcenter
    @pcenter.setter
    def pcenter(self, pcenter: bool):
        self._pcenter = pcenter
    
    
    @property
    def peccentr(self):
        return self._peccentr
    @peccentr.setter
    def peccentr(self, peccentr: bool):
        self._peccentr = peccentr
    
    
    @property
    def pelong(self):
        return self._pelong
    @pelong.setter
    def pelong(self, pelong: bool):
        self._pelong = pelong
    
    
    @property
    def piso(self):
        return self._piso
    @piso.setter
    def piso(self, piso: bool):
        self._piso = piso
        
    def initialize_cpp_instance(self):
        
        self.cv_feature_engine = cv2.htpa.FeatEngStat(pmin = self.pmin,
                                                      pmax = self.pmax,
                                                      pcenter = self.pcenter,
                                                      peccentr = self.peccentr,
                                                      pelong = self.pelong,
                                                      piso = self.piso)  