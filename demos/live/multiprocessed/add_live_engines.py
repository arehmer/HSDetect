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


ADD_POSTURE_ENGINE = True

MODEL_NAME = 'v13_adam'


SCRIPT_DIR = Path(__file__).resolve().parent

file_in = SCRIPT_DIR / 'models' / MODEL_NAME / 'inf_engine_params.dict'
file_out = SCRIPT_DIR / 'models' / (MODEL_NAME + '_live') / 'inf_engine_params_scaled.dict'


# %% Make output dir
dir_out = file_out.parent

dir_out.mkdir(exist_ok=True, parents=True)

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
engine_pipeline.insert(4,shrink_engine.save())

# %% Also change dl_model path
engine_pipeline[5]['dl_model'] = (dir_out / 'YOLOv2Tiny8X.keras').as_posix()

# %% Add the pose estimator if required
if ADD_POSTURE_ENGINE:
    
    param_dict = SCRIPT_DIR / 'models' / 'PoseClassifier' / 'PoseEstimator_HistGradientBoosting_with_ralign.pkl'
    
    with open(param_dict,'rb') as f:
        pose_estimator_params = pkl.load(f)
    
    pose_engine = build_engine(**pose_estimator_params)
    
    engine_pipeline.append(pose_engine.save())

# %% Append box_rescale engine
engine_pipeline.append(box_rescale_engine.save())
    
# %% Write pipeline back to parameter dict
engine_params_dict['engines'] = engine_pipeline

# %% Build engine to make sure it will build live as well
try:
    build_engine(**engine_params_dict)
except:
    raise RuntimeError('Unable to build engine!')
    
# %% Store the updated parameter dictionary
pkl.dump(engine_params_dict,
          open(file_out,'wb'))
