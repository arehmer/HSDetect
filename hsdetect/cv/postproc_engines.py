# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 09:54:07 2023

@author: Rehmer
"""

# import pandas as pd
# import pickle as pkl
# from pathlib import Path
import cv2

# from .gradient import  ConvolutionFilter
# from hsdetect.cv.misc import pd_IoU 
# from eval_measures import IoU


def build_postprocessing_engine(**kwargs):
    name = kwargs.pop('name')
    if name == 'PostprocEngineNMS':
        return PostprocEngineNMS(**kwargs)

import ast       
def get_classes():
    with open(__file__, 'r') as f:
        tree = ast.parse(f.read(), filename=__file__)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

class PostprocEngine():
    
    pass


class PostprocEngineNMS(PostprocEngine):
    
    name = 'PostprocEngineNMS'
    
    def __init__(self,**attr_dict):
        
        self.IoU_thresh = attr_dict.pop('IoU_thresh',None)
        self.filter_nested = attr_dict.pop('filter_nested',None)
        
        # Initialize PostprocEng_NMS from OpenCV
        self.cv_postproc_engine = \
            cv2.htpa.PostprocEng_NMS(self.IoU_thresh,
                                     self.filter_nested)
        
    def process(self,bbox_pred:cv2.htpa.DataFrame,label:int)->cv2.htpa.DataFrame:
        '''
        Applies process() method of Cpp class PostprocEng_NMS

        Parameters
        ----------
        bbox_pred : cv.htpa.DataFrame
            DESCRIPTION.

        Returns
        -------
        bbox_pred : cv.htpa.DataFrame
            DESCRIPTION.

        '''
        
        return self.cv_postproc_engine.process(bbox_pred,label)
    
    def save(self):
        '''
        Returns all non-private an non-builtin attributes of this class
        as a dictionary with the purpose of reloading this instance from the
        attribute dictionary. 

        Returns
        -------
        None.

        '''
        
        # Get all instance attributes 
        attr_dict = { k:v for k,v in vars(self).items() \
                     if not k.startswith('_') }
        
        # Throw out the svm_cpp as it cannot be pickled but will
        # be initialized when calling self.__init__()
        if 'cv_postproc_engine' in self.__dict__:
            del attr_dict['cv_postproc_engine']
        
        # Get all class attributes
        class_dict = {}
        for attribute in PostprocEngineNMS.__dict__.keys():
            # Check if it's a built-in type
            if attribute[:2] != '__':
                # Check if its a method
                value = getattr(PostprocEngineNMS, attribute)
                if not callable(value):
                    # If not append to dict
                    class_dict[attribute] = value

        # Concatenate both
        attr_dict.update(class_dict)
        
        return attr_dict