# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:08:31 2024

@author: rehmer
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any
import pickle as pkl

import shutil
import subprocess

from typing import Any, Callable, Dict, List, Optional, Tuple 


import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont

from pathlib import Path
import cv2
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

from hsdetect.cv.cpp_wrapper import SlidingWindowPyramid
# from hsdetect.gui.players import AnnotatedVideoPlayer
from hspytools.tparray import TPArray, ArrayTypes
from hspy.readers import HTPAdGUI_FileReader 


class RoI:
    def __init__(self,
                 roi_pix : np.ndarray,
                 roi_annot : pd.DataFrame):
        
        self.array = roi_pix
        self.annot = roi_annot
        
    @property
    def array(self):
        return self._array
    @array.setter
    def array(self,array):
        self._array = array

    @property
    def annot(self):
        return self._annot
    @annot.setter
    def annot(self,annot):
        self._annot = annot        
        
        

class PickledDataFrame:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data = None   # internal cache

    @property
    def data(self) -> pd.DataFrame:
        """Return the loaded pandas DataFrame, loading from disk if necessary."""
        if self._data is None:
            with open(self.path, "rb") as f:
                self._data = pkl.load(f)
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame):
        """Allow manually replacing the cached data."""
        self._data = value

    def __repr__(self):
        loaded = self._data is not None
        return f"PickledDataFrame(path='{self.path}', loaded={loaded})"


class PickledArray:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data = None   # internal cache

    @property
    def data(self) -> np.ndarray:
        """Return the loaded numpy array, loading from disk if necessary."""
        if self._data is None:
            with open(self.path, "rb") as f:
                self._data = pkl.load(f)
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        """Allow manually replacing the cached data."""
        self._data = value

    def __repr__(self):
        loaded = self._data is not None
        return f"PickledArray(path='{self.path}', loaded={loaded})"

class PickledROI:
    def __init__(self,
                 path_array: str | Path,
                 path_annot: str | Path):
        
        self.path_array = Path(path_array)
        self.path_annot = Path(path_annot)
        
        # Check if both files exist
        if not path_array.is_file():
            raise Exception(f"{path_array} does not exist.")

        if not path_annot.is_file():
            raise Exception(f"{path_annot} does not exist.")
        
        self.array = PickledArray(path_array)        # internally cached array           
        self.annot = PickledDataFrame(path_annot)    # internally cached annotation
        
        self._data = None

    @property
    def array(self) -> np.ndarray:
        """Return the loaded numpy array, loading from disk if necessary."""
        if self._array is None:
            with open(self.path_array, "rb") as f:
                self._array = pkl.load(f)
        return self._array.data

    @array.setter
    def array(self, pa: PickledArray):
        """Allow manually replacing the cached annotation."""
        self._array = pa
        
    @property
    def annot(self) -> pd.DataFrame:
        """Return the data attribute of the PickledDataFrame."""
        return self._annot.data

    @annot.setter
    def annot(self, pdf: PickledDataFrame):
        """Allow manually replacing the cached array."""
        self._annot = pdf
   
    @property
    def data(self) -> Dict[str, np.ndarray | pd.Series]:
        """Return the loaded numpy array, loading from disk if necessary."""
        if self._data is None:
                array = self.array
                annot = self.annot
                
                self.data = {'array':array,
                             'annot':annot}
                
        return self._data
    
    @data.setter
    def data(self, value: Dict[str, np.ndarray | pd.Series]):
        """Allow manually replacing the cached data."""
        self._data = value
    
    def __repr__(self):
        loaded = self._data is not None
        return f"PickledROI(loaded={loaded})"

class AnnotatedFrame:
    
    cv_datatypes = {'idx':'int32','xtl':'int32','ytl':'int32',
                    'xbr':'int32','ybr':'int32','w':'int32',
                    'h':'int32','score':'float64','track':'int32',
                    'strikes':'int32','label':'int32',
                    'IoU_max':'float64','IoU_idxmax':'int32',
                    'detector_idx':'int32','image_id':'int32'}
    
    def __init__(self,
                 index:int,
                 image:np.ndarray=None, 
                 annotations:cv2.htpa.DataFrame=None,
                 mask:np.ndarray=None, 
                 pyramid:SlidingWindowPyramid=None,
                 metadata=None):
        
        """
        metadata: optional dict for frame-level info (e.g., 'no_cv')
        """
        
        if image is not None and pyramid is None:
            # Single-frame mode
            self.mode = "single"
            self.index = index
            self.pyramid = None
            self.mask = mask
            
            self.metadata = metadata if metadata is not None else {}
            
            # In case no annotations are provided, create an empty dataframe
            # with proper columns
            if annotations is None:
                annotations = pd.DataFrame(data=[],
                                           columns = self.cv_datatypes)
            
            self._import_image_annotations(image,annotations)

        elif pyramid is not None and image is None and annotations is None:
            # Pyramid mode
            self.mode = "pyramid"
            self.index = index
            self.image = None
            self._annotations = pd.DataFrame(data=[],
                                             columns = self.cv_datatypes)
            self.mask = mask
            self._import_image_pyramid(image,pyramid)

        else:
            raise ValueError("Provide either (image + optional annotations) or a pyramid")

    def is_single(self):
        return self.mode == "single"

    def is_pyramid(self):
        return self.mode == "pyramid"

    def get_image(self):
        if not self.is_single():
            raise RuntimeError("Not in single-frame mode")
        return self.image
    
    def get_mask(self):
        if not self.is_single():
            raise RuntimeError("Not in single-frame mode")
        return self.mask

    @property
    def annotations(self):
        """
        Returns annotations for that frame. If the annotations are provided as
        an image pyramid, bounding box coordinates from scaled levels are con-
        verted into coordinates of the original frame.

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        if self.is_single():
            return self._annotations
        
        elif self.is_pyramid():
            pyramid_annotations = []
            
            for level in self.pyramid.levels:
                level_annotations = level.get_original_annotations()
                
                if len(level_annotations) != 0:
                    pyramid_annotations.append(level_annotations)
                    
            if len(pyramid_annotations) != 0:
                pyramid_annotations = pd.concat(pyramid_annotations)
              
            return pyramid_annotations
    
    @annotations.setter
    def annotations(self,annot:pd.DataFrame):
        """
        Sets the annotations of a single frame or a pyramid. Important: Indices
        have to be identical.

        Returns
        -------
        None.

        """
        
        if self.is_single():
            if not self.annotations.index.isin(annot.index).all():
                raise ValueError("Indices are not identical.")
            else:
                # Cast columns to proper datatype
                annot = annot.astype({col: dtype for col, dtype \
                                      in self.cv_datatypes.items() if col in annot.columns})
                self._annotations[annot.columns] = annot 
                
        elif self.is_pyramid():
            for level in self.pyramid.levels:
                level_idx = level.bboxes.index
                if not set(level_idx).issubset(set(annot.index)):
                    raise ValueError("Indices are not identical.")
                else:
                    level.bboxes.loc[level_idx,annot.columns] = \
                        annot.loc[level_idx]
                    
    def get_RoIs(self):
        """
        Extract regions of interest (ROIs) from an image based on bounding boxes
        defined in a DataFrame.
    
        Parameters
        ----------
        image : np.ndarray
            Image array with shape (H, W) or (H, W, C).
        df : pd.DataFrame
            DataFrame containing at least the columns [x1_col, y1_col, x2_col, y2_col].
            The DataFrame index will be used as the dictionary key.
        x1_col, y1_col, x2_col, y2_col : str
            Column names in `df` for top-left (x1, y1) and bottom-right (x2, y2) coordinates.
    
        Returns
        -------
        Dict[Any, dict]
            A dictionary mapping:
            index -> {
                "roi": np.ndarray,  # cropped image region
                "row": pd.Series    # the corresponding row from df
            }
        """
        
        # Get frame and annotations
        frame = self.image
        annot = self.annotations
        
        # If no annotations are given, return an empty dictionary
        if len(annot) == 0:
            RoIs =  {}
            return RoIs
        
        # Optional: basic column check
        for col in ('xtl','ytl','xbr','ybr'):
            if col not in annot.columns:
                raise ValueError(f"Required column '{col}' not found in annotations.")
    
        RoIs: Dict[Any, dict] = {}
    
        for idx, row in annot.iterrows():
            # Extract coordinates and make sure they are ints
            xtl = int(row['xtl'])
            ytl = int(row['ytl'])
            xbr = int(row['xbr'])
            ybr = int(row['ybr'])
    
            # Remember: NumPy uses image[row (y), col (x)]
            roi = frame[ytl:ybr, xtl:xbr]
    
            RoIs[idx] = {"array": roi,
                        "annot": row,  # this is already a Series
                        }
    
        return RoIs
        
        

    def get_pyramid(self):
        if not self.is_pyramid():
            raise RuntimeError("Not in pyramid mode")
        return self.pyramid

    def __repr__(self):
        if self.is_single():
            return f"<AnnotatedFrame SINGLE image shape={self.image.shape}, annotations={len(self.annotations)}>"
        else:
            return f"<AnnotatedFrame PYRAMID levels={(self.pyramid.numLevels())}>"

    def _import_image_annotations(self,
                                  image:np.ndarray,
                                  annotations:pd.DataFrame):
        """
        Imports an image and its annotations. 

        Parameters
        ----------
        image : TYPE
            DESCRIPTION.
        annotations : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        ############################ Import image ############################
        self.image = image
        
        ############################ Import annotations####################### 

        annot_import = annotations.copy()  
        
        # If passed and expected columns are already identical, only do 
        # conversion to proper type        
        if sorted(annotations.columns) == sorted(self.cv_datatypes.keys()):
            
            # Cast datatypes, check if columns are identical first
            annot_import = annotations.astype(self.cv_datatypes)

            
        else:
            
            # First find all columns that are missing in the passed annotations
            col_missing = set(self.cv_datatypes.keys()) - set(annotations.columns)
            col_missing = list(col_missing)
            
            # Add all these columns to annotations as -99, the default value
            annot_import[col_missing] = -99
            
            # Excption: w and h can be calculate from the given data
            annot_import['w'] = annot_import['xbr'] - annot_import['xtl']
            annot_import['h'] = annot_import['ybr'] - annot_import['ytl']
            
        
        # If the annotations do not have a unique index, reset it
        if not len(annot_import) == len(annot_import.index.unique()):
            annot_import = annot_import.reset_index(drop=True)
        
        # if the name of the index is not already "id" rename it for 
        # compatiblity with all other classes
        if not annot_import.index.name == "id":
            annot_import.index.rename('id',inplace=True)
                        
        self._annotations = annot_import
        
        
    def _import_image_pyramid(self,
                              image:np.ndarray,
                              pyramid:SlidingWindowPyramid):
        """
        Imports an image and pyramid. 

        Parameters
        ----------
        image : TYPE
            DESCRIPTION.
        annotations : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        ############################ Import image ############################
        self.image = image
        
        ############################ Import pyramid ##########################            
        
        # Reindex annotations and cast to proper type on all levels 
        idx0 = 0
        for level in pyramid.levels:
            # Create a new unique index for this level
            level_idx = pd.Index(range(idx0,idx0+len(level.bboxes)),
                                 name = 'id')
            
            level.bboxes.index = level_idx
            
            # Increment idx0 for next level
            idx0 = idx0 + len(level.bboxes)
        
            # Cast datatypes, check if columns are identical first
            if sorted(level.bboxes.columns) == sorted(self.cv_datatypes.keys()):
                level.bboxes = level.bboxes.astype(self.cv_datatypes)    
        
        self.pyramid = pyramid
        
    def to_png(self,dir_path : Path,**kwargs) -> None:
        """
        Write frame to png

        Parameters
        ----------
        dir_path : Path
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
        
        vmin = kwargs.pop('vmin',None)
        vmax = kwargs.pop('vmax',None)
        
        # Limit percentiles for linear scaling
        if vmin is None:
            vmin = np.percentile(self.image, 2) # 2nd percentile
        if vmax is None:
            vmax = np.percentile(self.image, 98) # 98nd percentile
        
        # File name equals index
        file_name = str(self.index) + '.png'
        
        
        matplotlib.image.imsave(dir_path / file_name,
                                self.image,
                                vmin = vmin,
                                vmax = vmax,
                                cmap = 'plasma')
        
        return file_name

class AnnotatedVideo:
    def __init__(self, frames=None, metadata=None):
        """
        frames: list of AnnotatedFrame objects
        metadata: optional dict for video-level info (e.g., 'fps', 'source', etc.)
        """
        if frames is not None and isinstance(frames,list):
            self.from_AnnotatedFrames(frames)
        else:
            self.frames = []
        self.frames = frames if frames is not None else []
        self.metadata = metadata if metadata is not None else {}

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

    def __setitem__(self, idx, frame):
        if not isinstance(frame, AnnotatedFrame):
            raise TypeError("Only AnnotatedFrame objects are allowed")
        self.frames[idx] = frame

    def append(self, frame):
        if not isinstance(frame, AnnotatedFrame):
            raise TypeError("Only AnnotatedFrame objects can be appended")
        self.frames.append(frame)

    def is_pyramid_video(self):
        return all(f.is_pyramid() for f in self.frames)

    def is_single_frame_video(self):
        return all(f.is_single() for f in self.frames)

    def get_metadata(self):
        return self.metadata

    def set_metadata(self, key, value):
        self.metadata[key] = value

    def __repr__(self):
        mode = "MIXED"
        if self.is_single_frame_video():
            mode = "SINGLE"
        elif self.is_pyramid_video():
            mode = "PYRAMID"
        return f"<AnnotatedVideo {len(self.frames)} frames, mode={mode}>"
    
    def to_png(self,
               dir_path : Path,
               M : int = 1,
               **kwargs : dict) -> None:
        """
        Write frames to .png-files

        Parameters
        ----------
        dir_path : Path
            Target directory.
        M : int, optional
            Downsampling factor. E.g. M=2 means that every second frame is 
            uploaded to CVAT The default is 1.

        Returns
        -------
        None
            DESCRIPTION.

        """
        
        vmin = kwargs.pop('vmin',None)
        vmax = kwargs.pop('vmax',None)
        
        # Create dir_path if it does not exist
        dir_path.mkdir(parents = True,
                       exist_ok = True)
                
        
        files = []
        
        for frame in self.frames[::M]:
            files.append(frame.to_png(dir_path,
                                      vmin = vmin,
                                      vmax = vmax))
    
    def to_txt(self,
               file:Path,ArrayType:int):
        """
        Converts the AnnotatedVideo to a pd.DataFrame and then uses 
        hspy.readers.HTPAdGUI_FileReader to store it as a .txt file.

        Parameters
        ----------
        file : Path
            Path to .txt file to write AnnotatedVideo to.
        ArrayType: int
            One of the ArrayTypes specified in hspy.tparray.ArrayTypes

        Returns
        -------
        None.

        """
        
        # Check input arguments first
        if not isinstance(file,Path):
            raise TypeError(f'{file} is type {type(file)} instead of Path.')
        if not isinstance(ArrayType, int):
            raise TypeError(f'{ArrayType} is type {type(ArrayType)} instead of int.')
        
        # Try to initialize tparray object from ArrayType. Will throw error if
        # ArrayType is not in hspy.tparray.ArrayTypes
        htpa = TPArray(ArrayType = ArrayType)
        
        # Obtain frames of the video as a pd.DataFrame
        df = self.frames_as_pdDataFrame()
        
        # Make sure the number of pixels in df matches the number of pixels of
        # htpa
        if not len(df.columns) == len(htpa.DataCols.pix):
            raise ValueError('Number of pixels in AnnotatedVideo does not '+\
                             f'match pixels expected by ArrayType {ArrayType}')
        
        # The AnnotatedVideo Type only stores pixels, therefore all the other
        # columns (ptat, vdd, etc) need to be replaced by dummy values to
        # maintain compatibility to SoftArray GUI
        df_list = [df]
        for key in htpa.DataCols._groups.keys():
            
            if key == 'pix':
                continue
            df_temp = pd.DataFrame(index = df.index,
                                   columns = htpa.DataCols._groups[key],
                                   data = -99)
            
            df_list.append(df_temp)
        
        df_concat = pd.concat(df_list,axis=1)
        
        # Create a dummy file-header, because SoftArray expects one
        dummy_header = f'ARRAYTYPE={ArrayType}ExportedAnnotatedVideo\r\n'
        
        # Initialize a reader and write both to file
        reader = HTPAdGUI_FileReader(htpa)
        
        reader.export_txt(df_concat,
                          dummy_header,
                          file)
        
    def to_yolo_dataset(self,
                         anchors: List[tuple[float, float]],
                         grid_shape: Tuple[int, int],
                         num_classes: int = 1,
                         class_column: str = "label"):
        """
        Build a tf.data.Dataset of (image, y_true) pairs from this video's
        frames, ready to feed to a YOLOv2-style detector.

        Assumes this AnnotatedVideo's frames already contain
        background-subtracted images (background subtraction is handled
        separately, upstream of this method). Annotations are expected in
        (xtl, ytl, xbr, ybr) pixel format, as stored in
        AnnotatedFrame.annotations; they are converted to (x, y, w, h)
        center format, assigned to a grid cell + anchor, and encoded as
        (tx, ty, tw, th) targets.

        Note: `anchors` should typically be computed once, globally,
        across ALL videos in your training set (e.g. via
        targets.kmeans_anchors() over the pooled annotations of every
        sequence) -- not recomputed per video -- so that every video's
        y_true tensor shares the same anchor definitions.

        Parameters
        ----------
        anchors : list of (w, h) tuples
            Anchor boxes in pixel units, matching the units of this
            video's annotation coordinates.
        grid_shape : (grid_h, grid_w)
            Output grid size of the detector for this video's frame
            resolution (e.g. model.grid_shape for the YOLOv2Tiny instance
            this data will train).
        num_classes : int
            Number of explicit class channels (0 to disable, folding the
            single implicit class into objectness only).
        class_column : str
            Column in each frame's annotations holding the integer class
            label. Defaults to 'label', per AnnotatedFrame.cv_datatypes.

        Returns
        -------
        tf.data.Dataset
            Yields (image, y_true) pairs, one per frame, in frame order.
        """
        from hsdetect.dl.video_dataset import build_video_dataset
        
        # Try to obtain 'id' from metadata and return an error if it does not
        # exist or has the wrong type
        metadata = self.get_metadata()
        
        if 'sequence_id' not in metadata.keys():
            raise ValueError('No sequence_id specified in self.metadata!')
        else:
            sequence_id = metadata['sequence_id']
        
        if not isinstance(sequence_id,int):
            raise TypeError(f'sequence_id is {type(sequence_id)} instead of {int}')
            
        return build_video_dataset(
            frames=self.frames,
            anchors=anchors,
            grid_shape=grid_shape,
            num_classes=num_classes,
            class_column=class_column,
            sequence_id = sequence_id
        )
    
    def frames_as_pdDataFrame(self):
        """
        Returns all frames as rows of a pandas DataFrame.

        Returns
        -------
        None.

        """
        
        # List for storing flattened frames
        frames_list = []
        
        # Loop over all frames, flatten and append to list
        for frame in self.frames:
            img = frame.image.flatten()
            frames_list.append(img)
        
        frame_array = np.vstack(frames_list)
        pd_index = pd.Index(name = 'image_id',
                            data = range(frame_array.shape[0]))
        
        return pd.DataFrame(data = frame_array,
                            columns = ['pix'+str(p) for p in range(frame_array.shape[1])],
                            index = pd_index)
    
        
    
    def get_RoIs(self):
        
        RoIs = {}
        for frame in self.frames:
            RoIs.update(frame.get_RoIs())
        
        return RoIs
    
    def store_RoIs(self,folder_dir:Path):
        """
        Stores all RoIs and their corresponding annotations in a folder with 
        structure:
            folder_dir / #frame / frame
                                / RoIs
                                / annot

        Parameters
        ----------
        folder_dir : Path
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Define directories to store content in
        frame_dir = folder_dir / 'frames'
        annot_dir = folder_dir / 'annots'
        roi_dir = folder_dir / 'rois'
        
        # Make those directories
        frame_dir.mkdir(parents=True, exist_ok=True)
        roi_dir.mkdir(parents=True, exist_ok=True)
        annot_dir.mkdir(parents=True, exist_ok=True)
        
        for frame in self.frames:
            
            
            # Store the full frame itself
            pkl.dump(frame.image,
                     open(frame_dir / (str(frame.index) + '.np'),'wb'))
                                      
            # Store the DataFrame with all annotations
            pkl.dump(frame.annotations,
                     open(annot_dir / (str(frame.index) + '.df'),'wb'))
            
            # Get RoIs of that frame
            rois = frame.get_RoIs()
            
            # If no RoIs exist (i.e. no annotations) for that frame, skip
            if len(rois) == 0:
                continue
            
            # Create a directory for all RoIs of that frame
            roi_frame_dir = roi_dir / str(frame.index)
            roi_frame_dir.mkdir(parents=True, exist_ok=True)
            
            
            # Store every RoI (box with pixels and annotation) as separate files
            for ID, roi in rois.items():
                pkl.dump(roi['array'],open(roi_frame_dir / (str(ID) + '.np') ,'wb'))
                pkl.dump(roi['annot'],open(roi_frame_dir / (str(ID) + '.df') ,'wb'))
            
            
            
            # pkl.dump()
            
    
    def from_AnnotatedFrames(self,frames:list):
        """
        Takes a list of annotated frames, sorts them according to their index
        in ascending order and removes frames with duplicate indices keeping
        only the most recent one.

        Parameters
        ----------
        frames : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Write frames to a dictionary. This serves the purpose of 
        # automatically overwriting frames with the same index, keeping
        # the most current one
        frame_dict = {}
        for frame in frames:
            if isinstance(frame, AnnotatedFrame):
                    frame_dict[frame.index] = frame
                    
        # Sort the dictionary just in case frames were provided in an unsorted
        # order
        frame_dict = \
            {k: frame_dict[k] for k in sorted(frame_dict.keys())}

        # Next write the sorted AnnotatedFrames to a list
        self.frames = [frame_dict[k] for k in frame_dict.keys()]
        
        return None
                        
                    
                
    def get_masks_dict(self):
        """
        Returns a dictionary with structure {frame_index<int>:mask<np.ndarray>}

        Returns
        -------
        None.

        """
        mask_dict = {}
        
        for frame in self.frames:
            idx = frame.index
            mask = frame.get_mask()

            mask_dict[idx] = mask
                
        return mask_dict
    
    def get_annotated_framdict(self)->dict:
        
        annotations = []
        framedict = {}
                
        for frame in self.frames:

            # Get annotations of current frame
            frame_annot = frame.annotations
            
            # Skip if no annotations exist for that frame
            if len(frame_annot) == 0:
                continue
            else:
                annotations.append(frame.annotations)
                framedict[frame.index] = frame.image
        
        # Concatenate all annotations from all frames
        annotations = pd.concat(annotations)

        # At this point the indices are not unique and need to be reset
        index_name = annotations.index.name
        annotations = annotations.reset_index(drop=True)
        annotations.index.name = index_name
                
        return framedict,annotations
    
    def get_annotations(self)->pd.DataFrame:
        """
        

        Returns
        -------
        frame_dict : TYPE
            DESCRIPTION.
        annot_df : TYPE
            DESCRIPTION.

        """
        """
        

        Returns
        -------
        annot_df : pd.DataFrame
            Dataframe containing the annotations of each frame.

        """        
        
        annotations = []
        
        for frame in self.frames:
            # Get annotations of current frame
            frame_annot = frame.annotations
            
            # Skip if no annotations exist for that frame
            if len(frame_annot) == 0:
                continue
            else:
                annotations.append(frame.annotations)
        
        # Concatenate all annotations from all frames
        if len(annotations) != 0:
            annotations = pd.concat(annotations)
        
            # At this point the indices are not unique and need to be reset
            index_name = annotations.index.name
            annotations = annotations.reset_index(drop=True)
            annotations.index.name = index_name
            
        else:
            # If all annotations were empty, pass at least one empty frame so 
            # that the column headers are compatible
            annotations = frame_annot
        
        return annotations
    
    def set_annotations(self, 
                        annot:pd.DataFrame,
                        mode = 'w')->None:
        """
        

        Parameters
        ----------
        annot : pd.DataFrame
                Pandas DataFrame containing the annotations that are to be 
                assigned to this video. Must contain a column "image_id" which
                signals to which frame the annotations belong.
        mode, optional: string
            DESCRIPTION. The default is 'w'.

        Returns
        -------
        None
            DESCRIPTION.

        """
        
        
        # Loop through all frames, assign annotations from annot to that frame        
        for frame in self.frames:
            
            frame_idx = frame.index
            
            # Get existing annotations to that frame+
            annot_old = frame.annotations.copy()
            
            # Filter for new annotations corresponding to that frame
            annot_new = annot.loc[annot['image_id']==frame_idx].copy()
            
            # Depending on the writing mode, replace or append the existing
            # annotations
            if mode == 'w':
                frame._annotations = annot_new
                
            elif mode == 'a':
                annot_concat = pd.concat([annot_old,annot_new])
                
                # Reindex while keeping original index name
                idx_name = annot_concat.index.name
                annot_concat = annot_concat.reset_index(drop=True)
                annot_concat.index.name = idx_name
                
                frame._annotations = annot_concat
                
        return None
    
    def import_single_frame_video(self,
                                  video:pd.DataFrame,
                                  SensorType:int,
                                  annotations:pd.DataFrame=None,
                                  masks:dict = None):
        """
        Import a video and corresponding annotations. 

        Parameters
        ----------
        video : pd.DataFrame
            DESCRIPTION.
        annotations : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        tparray = TPArray(SensorType=SensorType)
        
        # Create a dummy dictionar for masks if none is provided
        if masks is None:
            masks = {idx:None for idx in video.index }
        
        # Loop over the video frame by frame
        for i in video.index:
            
            # Get the frame as numpy array
            frame = video.loc[i,tparray._pix].values.reshape(tparray._npsize) 
            
            # Get all annotations for frame i
            if annotations is not None:
                frame_annotations = annotations.loc[annotations['image_id'] == i]
            else:
                # cols = AnnotatedFrame().cv_datatypes
                # frame_annotations = pd.DataFrame(data = [],
                #                                  )
                frame_annotations = None
            
            # get a mask if available
            if i in masks.keys():
                mask = masks[i]
            else:
                mask = None
            
            # Create an AnnotatedFrame
            annotated_frame = AnnotatedFrame(index = i,
                                             image = frame,
                                             annotations = frame_annotations,
                                             mask = mask)
            
            # Append AnnotatedFrame to video
            self.append(annotated_frame)
                        
        return None

    def to_mp4(self,mp4_path:Path,**kwargs):
        """
        A function for writing a video sequence given as a DataFrame to .mp4
        in a sepcified folder (path)
        

        Parameters
        ----------
        df_video : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        None.

        """        
        
        # Framerate
        fps = kwargs.pop('fps',20)
        crf = kwargs.pop('crf',5)
        global_scaling = kwargs.pop('global_scaling',True)
        
        if global_scaling:
            vmin = min(float(f.image.min()) for f in self.frames)
            vmax = max(float(f.image.max()) for f in self.frames)
        
        # First, write the whole sequence as .png in a temporary folder
        png_folder = mp4_path.parents[0] / 'png_temp'
        
        self.to_png(png_folder,
                    vmin = vmin,
                    vmax = vmax)
        
        
        # Make a video from the .png files using subprocess with ffmpeg
        png_files = sorted(png_folder.glob('*.png'), key=lambda p: int(p.stem))
        start_number = int(png_files[0].stem)
        
        # os.chdir(png_folder)
        result = subprocess.run(['ffmpeg','-y',
                                  '-framerate', str(fps),
                                  '-start_number', str(start_number),
                                  '-i', '%d.png',
                                  '-r', str(fps),
                                  # '-c:v', 'libx264',
                                  '-crf',str(crf),
                                  # '-pix_fmt', 'yuv420p',
                                  # '-movflags', '+faststart',
                                  mp4_path.as_posix()],
                                  cwd = png_folder,
                                  capture_output=True,
                                  text=True)
        
        # result = subprocess.run(['ffmpeg', '-y',
        #                  '-framerate', str(fps),
        #                  '-start_number', str(start_number),
        #                  '-i', '%d.png',
        #                  '-r', str(fps),
        #                  '-c:v', 'libx264',
        #                  '-crf', str(crf),
        #                  '-pix_fmt', 'yuv420p',
        #                  # '-profile:v', 'baseline ',
        #                  # '-level', '3.0',
        #                  '-colorspace', 'bt709',
        #                  '-color_primaries', 'bt709',
        #                  '-color_trc', 'bt709',
        #                  '-movflags', '+faststart',
        #                  mp4_path.as_posix()],
        #                  cwd=png_folder,
        #                  capture_output=True,
        #                  text=True)
        
        # Remove temporary folder
        # if result.ret
        if result.returncode == 0:
            shutil.rmtree(png_folder)

        return None  
    
    def write_frames_with_bboxes(self, output_dir: str, color='lime', linewidth=2):
        """
        Writes each frame of the video with annotated bounding boxes to individual PNG files
        using matplotlib for visualization.
    
        Parameters
        ----------
        output_dir : str
            Directory where PNG files will be saved.
        color : str
            Color of the bounding boxes. Default is 'lime'.
        linewidth : int
            Line width of the bounding boxes. Default is 2.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
        for frame in self.frames:
            if not frame.is_single():
                raise RuntimeError("Only single-frame videos are supported for export.")
    
            image = frame.get_image()
            annotations = frame.get_annotations()
    
            fig, ax = plt.subplots()
            ax.imshow(image)
    
            for _, row in annotations.iterrows():
                xtl, ytl = row['xtl'], row['ytl']
                width = row['xbr'] - row['xtl']
                height = row['ybr'] - row['ytl']
                rect = patches.Rectangle((xtl-0.5, ytl-0.5), width, height,
                                         linewidth=linewidth, edgecolor=color,
                                         facecolor='none')
                ax.add_patch(rect)
    
            ax.axis('off')
            fig.tight_layout(pad=0)
            filename = output_path / f"frame_{frame.index:04d}.png"
            fig.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
    
    def play(
        self,
        *,
        global_scaling = True,
        fps: int = 30,
        start_index: int = 0,
        loop: bool = True,
        scale: float = 1.0,
        title: Optional[str] = None,
        draw_fn: Optional[
            Callable[[Image.Image, int, Any], Image.Image]
        ] = None,
    ) -> None:
        
        player = AnnotatedVideoPlayer(
            self,
            fps=fps,
            start_index=start_index,
            loop=loop,
            scale=scale,
            title=title,
            draw_fn=draw_fn,
        )
        player.run()
        
class AnnotatedVideoPlayer:
    """
    Standalone Tkinter player for an AnnotatedVideo.

    Parameters
    ----------
    video : AnnotatedVideo
        Your AnnotatedVideo instance (must have .frames and .cv_datatypes).
    fps : int, default=30
        Target frames per second.
    start_index : int, default=0
        Initial frame index (0-based).
    loop : bool, default=True
        Whether to loop when reaching the last frame.
    scale : float, default=1.0
        Overall display scale (e.g., 0.5 shrink, 2.0 enlarge).
    title : str, optional
        Window title.
    draw_fn : callable, optional
        Callable(img, idx, key) -> PIL.Image, applied to the PIL image BEFORE
        overlaying canvas rectangles.
    """

    dtypes = {'xtl':'float32','ytl':'float32',
              'xbr':'float32','ybr':'float32','score':'float32',
              'image_id':'int32'}

    def __init__(
        self,
        video: AnnotatedVideo,
        *,
        fps: int = 30,
        start_index: int = 0,
        loop: bool = True,
        scale: float = 1.0,
        title: Optional[str] = None,
        draw_fn: Optional[Callable[[Image.Image, int, Any], Image.Image]] = None,
        global_scaling = True
    ) -> None:
        self.video = video
        self.fps = fps
        self.loop = loop
        self.scale = float(scale)
        self.title = title or "AnnotatedVideo Player"
        self.draw_fn = draw_fn
        
        
        # Determine global scaling parameters
        if global_scaling:
            self.vmin = min(float(f.image.min()) for f in video.frames)
            self.vmax = max(float(f.image.max()) for f in video.frames)
        else:
            self.vmin = None
            self.max = None
        # Expect the same attribute that your original play() used:
        # used to pick bbox columns.
        self.cv_datatypes = getattr(video, "cv_datatypes", {})

        # Playback state
        n_frames = len(self.video.frames)
        self.idx = int(np.clip(start_index, 0, n_frames - 1)) if n_frames > 0 else 0
        self.playing = False
        self.delay_ms = max(1, int(1000 / max(1, fps)))

        # Tk-related attributes, initialised later
        self.root: tk.Tk | None = None
        self.canvas: tk.Canvas | None = None
        self.image_item: int | None = None
        self.rect_items: list[int] = []
        self.text_items: list[int] = []
        self.scrub: tk.Scale | None = None
        self.btn_play: tk.Button | None = None

    # ------------------------------------------------------------------ UI BUILDING

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry("+100+80")

        # Canvas for image + overlays
        self.canvas = tk.Canvas(self.root, width=640, height=480,
                                highlightthickness=0, bd=0)
        self.canvas.pack()
        # keep strong ref to current PhotoImage
        self.canvas.photo = None  # type: ignore[attr-defined]

        # Controls
        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=8, pady=8)

        btn_prev = tk.Button(controls, text="⏮ Prev", width=8)
        self.btn_play = tk.Button(controls, text="▶ Play", width=8)
        btn_next = tk.Button(controls, text="Next ⏭", width=8)

        btn_prev.pack(side="left", padx=4)
        self.btn_play.pack(side="left", padx=4)
        btn_next.pack(side="left", padx=4)

        n_frames = len(self.video.frames)
        self.scrub = tk.Scale(
            self.root,
            from_=1,
            to=max(1, n_frames),
            orient="horizontal",
            showvalue=True,
            length=480,
            label="Frame",
            command=self._on_scrub,
        )
        self.scrub.pack(fill="x", padx=8, pady=(0, 10))
        self.scrub.set(self.idx + 1)

        # wire up controls
        btn_next.config(command=self.next_frame)
        btn_prev.config(command=self.prev_frame)
        self.btn_play.config(command=self.toggle_play)

        # keyboard shortcuts
        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("q", lambda e: self.root.destroy())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    # ------------------------------------------------------------------ PUBLIC API

    def run(self) -> None:
        """Create the Tk window, show the initial frame, and enter mainloop."""
        if len(self.video.frames) == 0:
            raise ValueError("AnnotatedVideo has no frames.")

        self._build_ui()
        self.show(self.idx)
        self.root.mainloop()  # type: ignore[union-attr]

    # ------------------------------------------------------------------ FRAME / IMAGE HELPERS

    def _get_boxes_for_frame(self, i: int):
        """Return iterable of boxes for frame i (xyxy or xywh)."""
        frame = self.video.frames[i]
        bxs = getattr(frame, "annotations", None)
        return bxs

    def _frame_to_pil(self, i: int) -> tuple[Image.Image, float]:
        
        arr = self.video.frames[i].image  # numpy array

        # Target display width: base 640 px, scaled by 'scale'
        target_w = max(1, int(round(640 * max(0.01, self.scale))))
        sf = target_w / arr.shape[1]
        w_scaled = target_w
        h_scaled = int(round(arr.shape[0] * sf))

        # Resize (float for quality)
        arr = arr.astype(np.float64)
        arr = cv2.resize(arr, dsize=(w_scaled, h_scaled),
                         interpolation=cv2.INTER_LINEAR)

        # Normalize to [0,1]
        amin, amax = float(arr.min()), float(arr.max())
        
        if self.vmin is not None:
            amin = self.vmin
            
        if self.vmax is not None:
            amax = self.vmax
            
        
        if amax > amin:
            arr = (arr - amin) / (amax - amin)
        else:
            arr = np.zeros_like(arr, dtype=np.float64)

        # Colormap -> RGB uint8
        rgb = (matplotlib.cm.plasma(arr)[..., :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb)

        # Optional pixel-space drawing
        if self.draw_fn is not None:
            out = self.draw_fn(img, i, None)
            if isinstance(out, Image.Image):
                img = out

        return img, sf

    # ------------------------------------------------------------------ BBOX / OVERLAYS

    def _shift_bboxes(self, df: pd.DataFrame) -> pd.DataFrame:
        shift_interval = 0.05

        # shift duplicated edges (left, top, right, bottom)
        for c in ["xtl", "ytl", "xbr", "ybr"]:
            c_dupl = df[df.duplicated(subset=[c], keep=False)]
            for i, idx in enumerate(c_dupl.index):
                shift = i * shift_interval
                df.loc[idx, c] -= np.float32(shift)

        # Check if some right/left edges are identical and shift
        lr_dupl_idx = df.index[df["xbr"].isin(df["xtl"])]
        lr_dupl = df.loc[lr_dupl_idx]

        for i, idx in enumerate(lr_dupl.index):
            shift = i * shift_interval
            df.loc[idx, "xbr"] -= np.float32(shift)

        # Check if some lower/upper edges are identical and shift
        ul_dupl_idx = df.index[df["ybr"].isin(df["ytl"])]
        ul_dupl = df.loc[ul_dupl_idx]

        for i, idx in enumerate(ul_dupl.index):
            shift = i * shift_interval
            df.loc[idx, "ybr"] += np.float32(shift)

        return df

    def _rect_patches_from_annot_df(self, df: pd.DataFrame, sf: float) -> tuple[list[int], list[int]]:
        if self.canvas is None:
            return [], []
    
        # Keep only those columns from the annotations that are necessary 
        needed_cols = ["xtl", "ytl", "xbr", "ybr"]
        optional_cols = ['occluded', 'outside', 'in_frame', 'IoU_max','score']
        for opt_col in optional_cols:
            if opt_col in df.columns:
                needed_cols.append(opt_col)
        
        # Cast DataFrame to float type because shifted boxes will not
        # coincide with full pixels
        df = df[needed_cols].copy()
    
        # Ensure numeric types
        for c in ["xtl", "ytl", "xbr", "ybr"]:
            df[c] = df[c].astype(np.float32)
        for c in optional_cols:
            if c in needed_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    
        # Then shift boxes (shift function expects xtl/ytl/xbr/ybr columns)
        df = self._shift_bboxes(df.copy())
    
        # Resample a colormap so all annotations have a different color
        N_annot = len(df.index)
        colormap = matplotlib.colormaps["nipy_spectral"].resampled(max(1, N_annot))
    
        rects: list[int] = []
        texts: list[int] = []
    
        # Text styling / placement
        pad = 2  # pixels (in *canvas* coordinates)
        font = ("TkDefaultFont", 9, "bold")
    
        for p in range(N_annot):
            # Pick color from colormap
            color = matplotlib.colors.to_hex(colormap(p))
            
            # If it is a confirmed track, always make box green
            if "confirmed" in df.columns:
                if df.iloc[p]["confirmed"] == 1.0:
                    color = \
                        matplotlib.colors.to_hex(matplotlib.colormaps["tab10"].colors[2])
                if df.iloc[p]["confirmed"] == 0.0:
                    color = \
                        matplotlib.colors.to_hex(matplotlib.colormaps["tab10"].colors[1])
                        
            # depending on whether the box is labelled as outside, occluded, or 
            # in_frame, change the boxes appearance
            opts = {}
            opts['width'] = 0.2
            if "occluded" in df.columns:
                if df.iloc[p]["occluded"] == 1.0:
                    opts['dash'] = (10,5)                    # draw dashed line if box is marked as occluded
            if "outside" in df.columns:
                if df.iloc[p]["outside"] == 1.0:
                    opts['width'] = 0                        # no line if box is marked as outside 
                    opts['fill'] = color
            if "in_frame" in df.columns:
                if df.iloc[p]["in_frame"] == 0.0:
                    opts['stipple'] = 'gray12'                # stipple if box is marked as not in_frame
                    opts['fill'] = color
                
            x1 = float(df.iloc[p]["xtl"]) * sf
            y1 = float(df.iloc[p]["ytl"]) * sf
            x2 = float(df.iloc[p]["xbr"]) * sf
            y2 = float(df.iloc[p]["ybr"]) * sf
            
            rect = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color,
                **opts
            )
            rects.append(rect)
    
            # Draw score text (top-left corner by default)
            if "score" in df.columns:
                score_val = df.iloc[p]["score"]
                if pd.notna(score_val):
                    label = f"{float(score_val):.2f}"
                    txt = self.canvas.create_text(
                        x1 + pad,
                        y1 + pad,
                        anchor="nw",
                        text=label,
                        fill=color,          # match box color
                        font=font,
                    )
                    texts.append(txt)
            
            # Draw score text (top-left corner by default)
            if "IoU_max" in df.columns:
                IoU_val = df.iloc[p]["IoU_max"]
                if pd.notna(IoU_val):
                    label = f"IoU={float(IoU_val):.2f}"
                    txt = self.canvas.create_text(
                        x1 + pad,
                        y2 + pad,
                        anchor="nw",
                        text=label,
                        fill=color,          # match box color
                        font=font,
                    )
                    texts.append(txt)
    
        return rects, texts

    def _draw_overlays(self, i: int, sf: float) -> None:
        if self.canvas is None:
            return
    
        # Delete rectangle patches from previous frame
        for rid in self.rect_items:
            self.canvas.delete(rid)
        self.rect_items = []
    
        # Delete score texts from previous frame
        for tid in getattr(self, "text_items", []):
            self.canvas.delete(tid)
        self.text_items = []
    
        # Get DataFrame with bounding boxes for current frame
        bxs = self._get_boxes_for_frame(i)
        if bxs is None or len(bxs) == 0:
            return
    
        # Create rectangle patches + score labels from current frame
        self.rect_items, self.text_items = self._rect_patches_from_annot_df(bxs, sf)

    # ------------------------------------------------------------------ CORE DISPLAY

    def show(self, i: int) -> None:
        if self.canvas is None:
            return

        n_frames = len(self.video.frames)
        i = int(np.clip(i, 0, n_frames - 1))

        img, sf = self._frame_to_pil(i)

        photo = ImageTk.PhotoImage(img, master=self.canvas)
        self.canvas.config(width=img.width, height=img.height)

        if self.image_item is None:
            self.image_item = self.canvas.create_image(
                0, 0, anchor="nw", image=photo
            )
        else:
            self.canvas.itemconfig(self.image_item, image=photo)

        # strong ref
        self.canvas.photo = photo  # type: ignore[attr-defined]

        self.idx = i

        # sync scrubber
        if self.scrub is not None:
            self.scrub.set(i + 1)

        # Draw overlays AFTER updating the image
        self._draw_overlays(i, sf)

    # ------------------------------------------------------------------ CONTROL CALLBACKS

    def next_frame(self) -> None:
        n_frames = len(self.video.frames)
        if self.idx + 1 < n_frames:
            self.show(self.idx + 1)
        elif self.loop and n_frames > 0:
            self.show(0)

    def prev_frame(self) -> None:
        n_frames = len(self.video.frames)
        if self.idx > 0:
            self.show(self.idx - 1)
        elif self.loop and n_frames > 0:
            self.show(n_frames - 1)

    def toggle_play(self) -> None:
        self.playing = not self.playing
        if self.btn_play is not None:
            self.btn_play.config(text="⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            self._play_loop()

    def _play_loop(self) -> None:
        if not self.playing or self.root is None:
            return
        self.next_frame()
        self.root.after(self.delay_ms, self._play_loop)

    def _on_scrub(self, val: str) -> None:
        try:
            idx = int(val) - 1
        except ValueError:
            return
        self.show(idx)

class AnnotationEngine:
    def __init__(self):
        pass

    def compare_videos_by_iou(self,
                              ground_truth_video:AnnotatedVideo,
                              proposal_video:AnnotatedVideo)->AnnotatedVideo:
        """
        For each frame, computes the max IoU between each proposed box and all
        ground-truth boxes. Modifies proposal_video in-place and returns a 
        DataFrame of the proposed boxes and their IoU with the ground truth as
        as well.

        Returns:
            prop_boxes: pd.DataFrame
        """
        # assert len(ground_truth_video) == len(proposal_video), "Video lengths must match"

        prop_bboxes_video = []
        
        # Loop over all frames in the proposal video
        for frame_idx in range(len(proposal_video)):
                        
            # Get the annotations for that frame from each video
            gt_boxes = ground_truth_video[frame_idx].annotations
            
            # If no annotations exist at all, this frame was not labelled and
            # therefore proposals can not be compared against any ground truth
            if len(gt_boxes) == 0:
                proposal_video[frame_idx]._annotations = \
                    proposal_video[frame_idx]._annotations.loc[[]]
                continue
            
            # Get all not-outside annotations
            gt_boxes = gt_boxes.loc[gt_boxes['outside']!=True]
            
            # If all annotations are outside, this might be a frame that 
            # contains instances of the target class, but was skipped as a
            # measure to reduce labelling work
            if len(gt_boxes) == 0:
                proposal_video[frame_idx]._annotations = \
                    proposal_video[frame_idx]._annotations.loc[[]]
                continue
            
            # Get proposed RoIs
            prop_bboxes_frame = proposal_video[frame_idx].annotations
            
            # If no proposals exist, skip
            if len(prop_bboxes_frame) == 0:
                continue
                
            # Compute the IoU between all bounding boxes in prop_boxes and
            # gt_boxes
            iou_matrix = self._compute_iou_matrix(prop_bboxes_frame,
                                                  gt_boxes)
            
            # Convert the matrix into a DataFrame with the row index corresponing
            # to the index of the proposed bounding boxes and the column index
            # corresponding to the index of the ground truth box
            iou_matrix = pd.DataFrame(data = iou_matrix,
                                      index = prop_bboxes_frame.index,
                                      columns = gt_boxes.index)
            
            # If no ground truth exists, set IoU of every proposal to zero and
            # set id of most similar ground truth box to 0, to indicate no
            # true bounding box existed
            if len(gt_boxes) == 0:
                iou_matrix['IoU_max'] = 0.0
                iou_matrix['IoU_idxmax'] = 0
            else:
                # If a ground truth exists, find most similar true box to every
                # proposed RoI
                iou_matrix['IoU_max'] = iou_matrix[gt_boxes.index].max(axis=1)
                iou_matrix['IoU_idxmax'] = iou_matrix[gt_boxes.index].idxmax(axis=1)
            
            # Assign the maximum IoU value and the corresponding label to 
            # the proposed bounding boxes
            prop_bboxes_frame['IoU_max'] = \
                iou_matrix['IoU_max'].astype(np.float64)
            prop_bboxes_frame['IoU_idxmax'] = \
                iou_matrix['IoU_idxmax'].astype(np.int32)
            
            # Because bounding box regression methods might need it: calculate
            # the offsets between the true bounding box coordinates and the
            # predicted bounding box
            bbox_coords = ['xtl','ytl','xbr','ybr']
            for gt_box in gt_boxes.index:
                pred_idx = \
                    prop_bboxes_frame.loc[prop_bboxes_frame['IoU_idxmax'] == gt_box].index
                
                # pred_idx can be empty, if a ground-truth box out of many 
                # has not been proposed
                if len(pred_idx) != 0:
                    prop_bboxes_frame.loc[pred_idx,['d_xtl','d_ytl','d_xbr','d_ybr']] = \
                        (gt_boxes.loc[gt_box,bbox_coords] - \
                            prop_bboxes_frame.loc[pred_idx,bbox_coords]).values
                            
            # Reassign the results to the annotated video
            proposal_video[frame_idx]._annotations = prop_bboxes_frame
                        
        return proposal_video
    
    def PerfStats(self,
                  ground_truth_video:AnnotatedVideo,
                  proposal_video:AnnotatedVideo,
                  IoU_pos:float,
                  stage:str,
                  **kwargs)->dict:
        """
        Determine performance statistics by comparing proposed RoIs with 
        the ground truth. If specified, the ground truth annotations are
        filtered. Filtered out ground truth annotations will therefore not
        influence the calculated statistics

        Parameters
        ----------
        ground_truth_video : AnnotatedVideo
            DESCRIPTION.
        proposal_video : AnnotatedVideo                  filters:dict
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        # Filters to apply to the annotation 
        # Structure of the dict is {'column_name':desired_value}
        standard_filter = {'occluded':False,
                           'outside':False,
                           'in_frame':True}

        filters = kwargs.pop('filters', standard_filter)
                
        score_thresh = kwargs.pop('score_thresh',None)
        
        valid_stages = ['proposed','classified','tracked']
        if stage not in valid_stages:
            raise ValueError(f"stage can take values {valid_stages} but is {stage}.")
        
        if stage == 'classified' and score_thresh is None:
            raise ValueError('If stage is set to classified, score_thresh must be provided.')
        
        # Initialize lists for storing TP, FP, FN and others
        TP = []
        FP = []
        FN = []
        OT = []
        
        # Loop over all frames in proposal_video
        for k in range(len(proposal_video)):
                        
            # Get the annotations for that frame from each video
            gt = ground_truth_video[k].annotations
            
            # Get proposed annotations for this frame
            roi = proposal_video[k].annotations
            
            # Apply filter to gt
            gt_filt = self.filter_annot(gt, filters)
            
            # Determine IDs of the bounding boxes that have been filtered out
            ID_filt = set(gt.index) - set(gt_filt.index) 
            
            if len(ID_filt)!=0:
                pass
            
            # All proposed RoIs that refer to the filtered out ground truth 
            # boxes are dropped, as they should not influence the performance
            # statistics
            idx_drop = roi.loc[(roi['IoU_idxmax'].isin(ID_filt)) &\
                               (roi['IoU_max'] != 0)].index
            
            # Add dropped RoIs to "others"
            ot = roi.loc[idx_drop].copy()
            
            roi.drop(index=idx_drop, inplace=True)
            
            # The remaining proposed RoIs can be classified as TP, FP and FN
            if stage == 'classified':
                tp = roi.loc[(roi['IoU_max']>=IoU_pos) &\
                             (roi['score']>=score_thresh)].copy()
                    
                tp = tp.loc[tp.groupby("IoU_idxmax")["IoU_max"].idxmax()]
                                
                fp = roi.loc[(roi['IoU_max']<IoU_pos) &\
                             (roi['score']>=score_thresh)].copy()
                
                fn = gt_filt.loc[~gt_filt.index.isin(tp['IoU_idxmax'])].copy()
            
            elif stage == 'proposed':
                tp = roi.loc[roi['IoU_max']>=IoU_pos].copy()   
                tp = tp.loc[tp.groupby("IoU_idxmax")["IoU_max"].idxmax()]
                
                fp = roi.copy().drop(tp.index)
                
                fn = gt_filt.loc[~gt_filt.index.isin(tp['IoU_idxmax'])].copy()
                
            elif stage == 'tracked':
                tp = roi.loc[(roi['IoU_max']>=IoU_pos) &\
                             (roi['confirmed']==True)].copy()
                tp = tp.loc[tp.groupby("IoU_idxmax")["IoU_max"].idxmax()]
                
                fp = roi.loc[(roi['IoU_max']<IoU_pos) &\
                             (roi['confirmed']==True)].copy()
                
                fn = gt_filt.loc[~gt_filt.index.isin(tp['IoU_idxmax'])].copy()
            
            # Add a column for storing the category label
            tp['classification'] = 'TP'
            fp['classification'] = 'FP'
            fn['classification'] = 'FN'
            ot['classification'] = 'OT'
            
            # Append to lists
            TP.append(tp); FP.append(fp); FN.append(fn); OT.append(ot)
        
        # Concatenate all lists to pd.DataFrames
        TP = pd.concat(TP); FP = pd.concat(FP)
        FN = pd.concat(FN); OT = pd.concat(OT)
        
        # Calculate precision, recall and F1 score
        if (len(TP)+len(FP)) == 0:
            prec = 0
        else:
            prec = len(TP) / (len(TP)+len(FP))
        
        
        if (len(TP)+len(FN)) == 0:
            rec = 0
        else:
            rec = len(TP) / (len(TP)+len(FN))
            
        if (prec + rec) == 0:
            F1 = 0
        else:
            # Calculate F1 score
            F1 = 2*(prec * rec) / (prec + rec)
            
        # Store results
        stat = {}
        stat['results'] = {}
        stat['results']['TP'] = TP
        stat['results']['FP'] = FP
        stat['results']['FN'] = FN
        stat['results']['OT'] = OT
        
        stat['precision'] = prec
        stat['recall'] = rec
        stat['F1'] = F1
        
        return stat
            
            
    
    def _compute_iou_matrix(self, df1, df2):
        """
        Computes IoU between all boxes in df1 (proposed) and df2 (ground truth).
        Assumes columns: xtl, ytl, xbr, ybr
        Returns a numpy array of shape (len(df1), len(df2))
        """
        boxes1 = df1[["xtl", "ytl", "xbr", "ybr"]].values.astype(np.float32)
        boxes2 = df2[["xtl", "ytl", "xbr", "ybr"]].values.astype(np.float32)

        iou_matrix = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

        for i, (x1a, y1a, x2a, y2a) in enumerate(boxes1):
            for j, (x1b, y1b, x2b, y2b) in enumerate(boxes2):
                xa = max(x1a, x1b)
                ya = max(y1a, y1b)
                xb = min(x2a, x2b)
                yb = min(y2a, y2b)


                inter_area = max(0, xb - xa) * max(0, yb - ya)
                area_a = (x2a - x1a) * (y2a - y1a)
                area_b = (x2b - x1b) * (y2b - y1b)
                union_area = area_a + area_b - inter_area

                iou = inter_area / union_area if union_area > 0 else 0.0
                iou_matrix[i, j] = iou
                
        return iou_matrix
    
    def filter_annot(self,annot:pd.DataFrame,filters)->pd.DataFrame:
        
        annot_filt = annot.copy()
        
        # Apply provided filter criteria consecutively
        for col,val in filters.items():
            annot_filt = annot_filt.loc[annot_filt[col]==val]
            
        return annot_filt
