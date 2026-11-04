# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 09:54:07 2023

@author: Rehmer
"""

# import pandas as pd
# import pickle as pkl
# from pathlib import Path
import cv2
import pandas as pd

from hspy.tparray import TPArray

# from .gradient import  ConvolutionFilter
# from hsdetect.cv.misc import pd_IoU 
# from eval_measures import IoU

from hsdetect import Engine

# Marker identifying this .py as a module containing engines
IS_ENGINE_MODULE = True

def build_postprocessing_engine(**kwargs):
    name = kwargs.pop('name')
    if name == 'PostprocEngineNMS':
        return PostprocEngineNMS(**kwargs)
    elif name == 'BoxRescale':
        return BoxRescale(**kwargs)
    else:
        raise Exception('Preprocessing Engine ' + name + ' not known!')

import ast    
   
def get_classes():
    with open(__file__, 'r') as f:
        tree = ast.parse(f.read(), filename=__file__)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

class PostprocEngine(Engine):
    
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


def shrink_geometry(w: int, h: int, factor: float):
    """Geometry realized by shrink_and_center(): the content size is rounded and
    the offset floor-divided, so the per-axis scales differ from `factor` and
    from each other. Both ShrinkCenter and BoxRescale must use THESE values."""
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    left  = (w - new_w) // 2
    top   = (h - new_h) // 2
    return left, top, new_w / w, new_h / h
    
class BoxRescale(PostprocEngine):
    """Maps annotations from ShrinkCenter's canvas coordinates back to sensor
    coordinates. Must be configured with the SAME factor as ShrinkCenter."""

    name = 'BoxRescale'

    def __init__(self, **attr_dict):
        self.SensorType = attr_dict['SensorType']          # property, for save()
        self.factor = attr_dict.pop('factor', 1.0)         # property, for save()
        self.drop_outside = attr_dict.pop('drop_outside', True)
        self._tparray = TPArray(SensorType=self.SensorType)

    def __call__(self, annotations: pd.DataFrame) -> pd.DataFrame:
        w, h = self._tparray.width, self._tparray.height
        left, top, sx, sy = shrink_geometry(w, h, self.factor)

        annot = annotations.copy()
        if len(annot) == 0 or self.factor == 1.0:
            return annot                      # preserves columns and index name

        if self.drop_outside:
            # A detection whose centre lies in the zero border cannot be a real
            # object -- it is a padding artifact. Filter BEFORE rescaling.
            cx = (annot['xtl'] + annot['xbr']) / 2.0
            cy = (annot['ytl'] + annot['ybr']) / 2.0
            keep = (cx.between(left, left + round(w*self.factor)) &
                    cy.between(top,  top  + round(h*self.factor)))
            annot = annot.loc[keep]
            if len(annot) == 0:
                return annot

        annot[['xtl','xbr']] = (annot[['xtl','xbr']] - left) / sx
        annot[['ytl','ybr']] = (annot[['ytl','ybr']] - top)  / sy
        annot['w'] = annot['w'] / sx
        annot['h'] = annot['h'] / sy
        return annot