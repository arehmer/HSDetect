# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:31:12 2026

@author: rehmer
"""
from pathlib import Path
# import numpy as np
from hspy.tparray import SensorTypes

from hsdetect import build_engine
from hsdetect.cv.inf_engines import InferenceEngine
from hsdetect.engine_io import load_engine_params, save_engine_params


ADD_POSTURE_ENGINE = True

MODEL_NAME = 'v12_adam'


SCRIPT_DIR = Path(__file__).resolve().parent

file_in = SCRIPT_DIR / 'models' / MODEL_NAME / 'inf_engine_params.json'
file_out = SCRIPT_DIR / 'models' / (MODEL_NAME + '_live') / 'inf_engine_params_scaled.json'

#%% Rebuild the inference engine in the correct order
engine_order = ['PreprocEngAWA', 'ZeroCorners', 'BackgroundSub', 'MedianSub', 'DLEngine', 'YOLO_EnDecode_Engine', 'BoxPoseClassifier']

# %% Make output dir
dir_out = file_out.parent

dir_out.mkdir(exist_ok=True, parents=True)

# %% Load inf_engine_params.json
engine_params_dict = load_engine_params(file_in)

engine_pipeline = engine_params_dict['engines']
# %% Change dl_model path
# The Keras model is expected in the output directory. save_engine_params()
# stores the path relative to the parameter file, i.e. as file name only.
InferenceEngine.get_engine_params(engine_params_dict, 'DLEngine')['dl_model'] =     (dir_out / 'YOLOv2Tiny8X.keras').as_posix()

# %% Add the pose estimator if required
if ADD_POSTURE_ENGINE:
    
    # Only a reference to the classifier JSON is stored in the parameters
    model_file = SCRIPT_DIR / 'models' / 'PoseClassifier' / 'PoseEstimator_HistGradientBoosting_with_ralign.json'
    
    pose_engine = build_engine(name = 'BoxPoseClassifier',
                               model_file = model_file)
    
    engine_pipeline.append(pose_engine.save())
    
# %% Make sure all engines have the right order
engine_pipeline_ordered = []

for engine in engine_order:
    
    for e in engine_pipeline:
        if e['name'] == engine:
            engine_pipeline_ordered.append(e)
            break

# %% Write pipeline back to parameter dict
engine_params_dict['engines'] = engine_pipeline_ordered

# %% Build engine to make sure it will build live as well
try:
    inf_engine =  build_engine(**engine_params_dict)
except:
    raise RuntimeError('Unable to build engine!')
    
# %% Store the updated parameters
save_engine_params(engine_params_dict, file_out)
