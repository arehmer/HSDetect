# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:31:12 2026

@author: rehmer
"""
import pickle as pkl
from pathlib import Path
# import numpy as np
from hspy.tparray import SensorTypes

from hsdetect import build_engine


SCRIPT_DIR = Path(__file__).resolve().parent

file_in = SCRIPT_DIR / 'models' / 'yolo_v1' / 'inf_engine_params.dict'
file_out = SCRIPT_DIR / 'models' / 'yolo_v1_scaled' / 'inf_engine_params_scaled.dict'


# %% Load inf_engine_params.dict
with open(file_in,'rb') as io:
    engine_params_dict = pkl.load(io)


# %% Create an engine for downscaling the data
scaling_factor = 0.6

shrink_engine = build_engine(name = 'ShrinkCenter',
                             SensorType = SensorTypes['HTPA160x120DR1_L3K95_0K8'],
                             factor = scaling_factor)

box_rescale_engine = build_engine(name = 'BoxRescale',
                              SensorType = SensorTypes['HTPA160x120DR1_L3K95_0K8'],
                              factor = scaling_factor)

# %% Insert shrink_engine parameters before the yolo model and the  
# box_rescale_engine parameters after  the decoding engine
engine_pipeline = engine_params_dict['engines']

engine_pipeline.insert(3,shrink_engine.save())
engine_pipeline.append(box_rescale_engine.save())

# %% Also change dl_model path
engine_pipeline[4]['dl_model'] = \
    'C:/Users/Rehmer/Documents/GitHub/HSDetect/demos/live/multiprocessed/models/yolo_v1_scaled/YOLOv2Tiny8X.keras'

# %% Write pipeline back to parameter dict
engine_params_dict['engines'] = engine_pipeline

build_engine(**engine_params_dict)

# %% Store the updated parameter dictionary
pkl.dump(engine_params_dict,
         open(file_out,'wb'))
