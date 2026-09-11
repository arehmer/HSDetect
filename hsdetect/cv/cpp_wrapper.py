import cv2
import numpy as np
import pandas as pd

class PyramidLevel:
    
    def __init__(self, cpp_obj):
        
        if cpp_obj is None:
            raise ValueError("cpp_obj must be provided")

        self._cpp_obj = cpp_obj

        # Deep copy C++ members to Python-side variables
        self._image = cpp_obj.image.copy()  # NumPy copy of cv::Mat
        self._bboxes = self._deepcopy_bboxes(cpp_obj.bboxes)
        self._scale_factor = float(cpp_obj.scale_factor)


    # ----- image -----
    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        if not isinstance(value, (np.ndarray, cv2.Mat)):
            raise TypeError("image must be a NumPy array or cv2.Mat")
        self._image = value.copy()

    # ----- bboxes -----
    @property
    def bboxes(self):
        return self._bboxes

    @bboxes.setter
    def bboxes(self, bboxes):
        if isinstance(bboxes, cv2.htpa.DataFrame):
            self._bboxes = self._deepcopy_bboxes(bboxes)
        elif isinstance(bboxes, pd.DataFrame):  
            self._bboxes = bboxes
        else:
            raise TypeError("bboxes must be cv2.htpa.DataFrame or pd.DataFrame")
            

    # ----- scale_factor -----
    @property
    def scale_factor(self):
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("scale_factor must be numeric")
        self._scale_factor = float(value)

    # ---- Utility ----
    def to_cpp(self):
        """Convert this Python object back to a C++ PyramidLevel."""
        return cv2.htpa_PyramidLevel(self._image, self._bboxes, self._scale_factor)
    
    def _deepcopy_bboxes(self, bboxes):
        """
        Custom deep copy logic for bboxes.
        """
        if not isinstance(bboxes, cv2.htpa.DataFrame):
            raise TypeError("bboxes must an instance of cv2.htpa.DataFrame")
    
        # Convert to pd.DataFrame
        bboxes = pd.DataFrame(data = bboxes.get_content(),
                              columns = bboxes.columns)
        
        return bboxes

    def numLevels(self):
        return self._cpp_obj.numLevels()
    
    def get_original_annotations(self)->pd.DataFrame():
        """
        Converts the bounding box coordinates from that level back to the
        coordinate system of the original, i.e. unscaled, frame
        """
        bbox_cols = ['xtl','ytl','xbr','ybr']
        
        level_annot = self.bboxes.copy()
        
        
        level_annot[bbox_cols] = \
            (level_annot[bbox_cols] * self.scale_factor)
        
        # REVIEW: Is rounding here detrimental because it shifts the bounding 
        # box somewhere it should be?
        level_annot[bbox_cols] = \
            level_annot[bbox_cols].round().astype(np.int32)
        
        level_annot['w'] = level_annot['xbr'] - level_annot['xtl']
        level_annot['h'] = level_annot['ybr'] - level_annot['ytl']

        return level_annot
        
class SlidingWindowPyramid:
    def __init__(self, cpp_obj):
        
        # Save original C++ object
        self._cpp_obj = cpp_obj
        
        # Make a deep copy of levels from the original C++ object
        self._levels = []

        for i in range(cpp_obj.size()):
            cpp_level = cpp_obj.getLevel(i)
            
            # Wrap and store a copy of each level
            level = PyramidLevel(cpp_level)
            self._levels.append(level)
    
    def __getitem__(self, idx):
     return self._levels[idx]
    
    # ----- image -----
    @property
    def levels(self):
        return self._levels
            
    def get_level(self, idx):
        cpp_level = self._levels[idx]
        return cpp_level
    
    def numLevels(self):
        return len(self._levels)

    # def add_level(self, level: PyramidLevel):
    #     self._cpp_obj.addLevel(level.to_cpp())

    # def __getitem__(self, idx):
    #     return self.get_level(idx)

    # def __len__(self):
    #     return self._cpp_obj.size()

    # def clear(self):
    #     self._cpp_obj.clear()
    
