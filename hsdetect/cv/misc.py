# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:07:59 2023

@author: Rehmer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import cv2

from hspy.tparray import TPArray, ArrayTypes, SensorTypes

# Python mirror of enum cv2::htpa::Kernel in Misc.hpp. Needs to be updated 
# manually!
Kernel = {'Epanechnikov':0,
          'Gaussian':1,
          'Box':2,
          'Hann':3}

def Engine_to_FileStorage(engine, file:Path):
    """
    Calls an engine's save() method to obtain its parameter dictionary and then
    writes it to file via OpenCV's cv::FileStorage method.

    Parameters
    ----------
    engine : TYPE
        DESCRIPTION.
    filename : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Call engine's save() method to obtain parameter dictionary
    attr_dict = engine.save()
    
    # Open a file fs
    fs = cv2.FileStorage(file,cv2.FILE_STORAGE_WRITE)
    
    # Make sure file is opened, raise error otherwise
    if not fs.isOpened():
        raise IOError(f"Could not open '{file}' for writing with cv2.FileStorage")
    
    # Write the engine's name in the header
    fs.write("engine", engine.name)
    
    # Store model as a node (nice structure in YAML)
    #fs.startWriteStruct("svm_model", cv2.FileNode_MAP)
    
    # Loop through every single attribute stored in the dictionary and write it
    # to file
    for attr,value in attr_dict.items():
        
        # Exception for booleans: store as int
        if(isinstance(value, bool)):
            value = int(value)
        fs.write(attr, value)
        
    
    
    
    
    
def cv2_row_detect_to_pd_Series(row_detect : cv2.htpa.row_detect)->pd.Series:
    

    # Create an empty pd.Series to write values to
    pdSeries = pd.Series(index = cv2.htpa.DataFrame(0).get_headers())
    
    
    # Iterate over columns (pd-Series index)
    for c in pdSeries.index:
        
        dtype = type(getattr(row_detect,c))
        dvalue = getattr(row_detect,c)
        
        pdSeries[c] = dvalue
        pdSeries[c].astype(dtype)
        
    return pdSeries
    
    

def cv2_dfTrack_to_pd_DataFrame(df_cpp : cv2.htpa.DataFrameTrack_v0):
    
    # Create a dictionary with {column_name:datatype} for every column
    row_detect = cv2.htpa.row_track_v0()
    
    dtypes = {}
    for c in df_cpp.get_headers():
        if hasattr(row_detect, c):
            dtypes[c] = type(getattr(row_detect,c))
    
    df = pd.DataFrame(data = df_cpp.get_content(),
                      columns = df_cpp.get_headers()).astype(dtypes)
    
    return df

def cv2_dfDetect_to_pd_DataFrame(df_cpp : cv2.htpa.DataFrameTrack_v0):
    
    # Create a dictionary with {column_name:datatype} for every column
    row_detect = cv2.htpa.row_detect()
    
    dtypes = {}
    for c in df_cpp.get_headers():
        if hasattr(row_detect, c):
            dtypes[c] = type(getattr(row_detect,c))
    
    df = pd.DataFrame(data = df_cpp.get_content(),
                      columns = df_cpp.get_headers()).astype(dtypes)
    
    return df


def cv2_to_pd_DataFrame(df_cpp : cv2.htpa.DataFrame):
    
    return cv2_dfDetect_to_pd_DataFrame(df_cpp)

def pd_to_cv2_htpa_DataFrame(df: pd.Series | pd.DataFrame):
    """
    Converts pandas DataFrame to cv2.htpa.DataFrame instance which is a 
    std::vector of structs
    with definition
    
        struct CV_EXPORTS_W_SIMPLE row_detect
        {
            CV_PROP_RW int idx = -99;
            CV_PROP_RW int xtl = -99;
            CV_PROP_RW int ytl = -99;
            CV_PROP_RW int xbr = -99;
            CV_PROP_RW int ybr = -99;
            CV_PROP_RW int w = -99;
            CV_PROP_RW int h = -99;
            CV_PROP_RW int image_id = -99;
            CV_PROP_RW float score = -99.0f;
            CV_PROP_RW int track = -99;
            CV_PROP_RW int strikes = -99;
            CV_PROP_RW int label = -99;
            CV_PROP_RW float IoU_max = -99.0f;
            CV_PROP_RW int IoU_idxmax = -99;
            CV_PROP_RW int detector_idx = -99;
        };

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    df_cpp : TYPE
        DESCRIPTION.

    """
    
    
    
    if isinstance(df,pd.Series):
        df= df.to_frame().T
           
    # Init cv2.DataFrame 
    df_cpp = cv2.htpa.DataFrame(len(df))

    # Iterate over all rows
    for k,row in df.iterrows():
        
        # Init new cv2 row
        row_cv = cv2.htpa.row_detect()
        
        # Iterate over all columns
        for col in row.keys():
            
            # Skip if column in bbox does not exist in bbox_cv type
            if not hasattr(row_cv, col):
                continue
            
            # Get type cpp instance expects
            dtype = type(getattr(row_cv, col))  
            
            # Convert to that type and assign 
            setattr(row_cv, col, dtype(row[col]))
        
        # Add new row to cpp DataFrame
        df_cpp.add_row(row_cv)
            
    return df_cpp

def pd_to_cv2_htpa_row_detect(df: pd.Series | pd.DataFrame):
    """
    Converts pandas DataFrame or Series cv2.htpa.row_detect instance 
    with definition
    
        struct CV_EXPORTS_W_SIMPLE row_detect
        {
            CV_PROP_RW int idx = -99;
            CV_PROP_RW int xtl = -99;
            CV_PROP_RW int ytl = -99;
            CV_PROP_RW int xbr = -99;
            CV_PROP_RW int ybr = -99;
            CV_PROP_RW int w = -99;
            CV_PROP_RW int h = -99;
            CV_PROP_RW int image_id = -99;
            CV_PROP_RW float score = -99.0f;
            CV_PROP_RW int track = -99;
            CV_PROP_RW int strikes = -99;
            CV_PROP_RW int label = -99;
            CV_PROP_RW float IoU_max = -99.0f;
            CV_PROP_RW int IoU_idxmax = -99;
            CV_PROP_RW int detector_idx = -99;
        };

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    df_cpp : TYPE
        DESCRIPTION.

    """
    
    
    
    if isinstance(df,pd.Series):
        df= df.to_frame().T
        
    if len(df)!=1:
        raise ValueError('Length of df must be equal to 1!')

    # Iterate over all rows
    for k,row in df.iterrows():
        
        # Init new cv2 row
        row_cv = cv2.htpa.row_detect()
        
        # Iterate over all columns
        for col in row.keys():
            
            # Skip if column in bbox does not exist in bbox_cv type
            if not hasattr(row_cv, col):
                continue
            
            # Get type cpp instance expects
            dtype = type(getattr(row_cv, col))  
            
            # Convert to that type and assign
            if isinstance(row[col],np.ndarray):
                setattr(row_cv, col, dtype(row[col].item()))
            else:
                setattr(row_cv, col, dtype(row[col]))
            
    return row_cv
    

def IoU(b1,b2):
    """
    
    
    Parameters
    ----------
    b1 : tuple of tuples
        first tuple is upper left corner, second one lower right corner 
        of bounding box 1.
    b2 : tuple of tuples
        first tuple is upper left corner, second one lower right corner 
        of bounding box 2.
    size: size of target picture as (w,h)
    
    Returns
    -------
    IoU.
    
    """ 

    max_dim = max(max(max(b1)),max(max(b2)))
    
    img1 = np.zeros((max_dim,max_dim))
    img2 = np.zeros((max_dim,max_dim))
    
    img1[b1[0][1]:b1[1][1],b1[0][0]:b1[1][0]] = 1
    
    img1 = img1.flatten()
    b1_idx = np.nonzero(img1)[0]   
    
    img2[b2[0][1]:b2[1][1],b2[0][0]:b2[1][0]] = 1
    
    img2 = img2.flatten()
    b2_idx = np.nonzero(img2)[0]  
    
    # Calculate intersecting number of pixels
    intersection = len(np.intersect1d(b1_idx,b2_idx))
    
    # Calculate total number of pixels
    union = len(np.union1d(b1_idx,b2_idx))
    
    # Calculate IoU
    IoU = intersection / union

    return IoU

def pd_IoU(annot1:pd.Series,annot2:pd.Series):
    """
    A IoU version acting on pandas DataFrames rather than tuples
    

    Parameters
    ----------
    annot1 : TYPE
        DESCRIPTION.
    annot2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Check tyoe
    if (not isinstance(annot1,pd.Series)) or (not isinstance(annot2,pd.Series)):
        raise TypeError('annotations mus be pd.Series.')
    
    # First get corner points of both boxes
    tl_1 = (annot1['xtl'].astype(np.int32),
            annot1['ytl'].astype(np.int32))
    
    br_1 = (annot1['xbr'].astype(np.int32),
            annot1['ybr'].astype(np.int32))

    tl_2 = (annot2['xtl'].astype(np.int32),
            annot2['ytl'].astype(np.int32))
    
    br_2 = (annot2['xbr'].astype(np.int32),
            annot2['ybr'].astype(np.int32))    
    
    # pass to original IoU function and return
    
    return IoU((tl_1,br_1),(tl_2,br_2))


def mask_circular(r:int,
                  ArrayType: ArrayTypes = None,
                  SensorType : SensorTypes = None):
    """
    Returns a centered circular mask of type np.uint8
    
    Parameters
    ----------
    r : int
        Radius in pixel.
    ArrayType : ArrayTypes
        ArrayType from hspy.tparray.ArrayTypes.

    Returns
    -------
    mask : TYPE
        DESCRIPTION.

    """
    
    # Check inputs
    if ArrayType is None and SensorType is None:
        raise ValueError('Either ArrayType or SensorType must be specified.')

    if ArrayType is not None and SensorType is not None:
        raise ValueError('Only ArrayType or SensorType must be specified.')    
    
    
    # Create instance of TPArray
    if ArrayType is not None:
        htpa = TPArray(ArrayType = ArrayType)
    if SensorType is not None:
        htpa = TPArray(SensorType = SensorType)
    
    # create circular mask
    h = htpa.height
    w = htpa.width

    center = (h/2, w/2)                         # Center of array
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[1])**2 + (Y - center[0])**2)

    mask = dist <= r
    
    mask = mask.astype(np.uint8)*255

    return mask    