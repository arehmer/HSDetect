# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:31:12 2026

@author: rehmer
"""
import pickle as pkl
import numpy as np
from hspy.tparray import SensorTypes

from hsdetect import build_engine


file = 'C:/Users/Rehmer/Documents/GitHub/HSDetect/demos/live/multiprocessed/models/inf_engine_params.dict'



with open(file,'rb') as io:
    inf_engine_params = pkl.load(io)

    
    # Add a decoding engine to the inference pipeline
    decoding_engine = build_engine(name = 'YOLO_EnDecode_Engine',
                                   SensorType = SensorTypes['HTPA160x120DR1_L3K95_0K8'],
                                   yolo_anchors = np.array([[15., 13.],
                                                            [23., 17.],
                                                            [18., 23.]]),
                                   yolo_grid_size = (15,20),
                                   yolo_n_classes = 0)
    
    decoding_engine_params = decoding_engine.save()
    inf_engine_params['engines'].append(decoding_engine_params)
    
    # Build inference engine
    inference_engine = build_engine(**inf_engine_params)
    print('Inference engine built.')
    
    # inf_engine_params = inference_engine.save()
    # pkl.dump(inf_engine_params,open('./models/inf_engine_params.dict','wb'))

for i in range(10):
    fake_frame = np.random.random(size = (120,160)).astype(np.uint16)
    result = inference_engine(fake_frame)


# result = inference_engine.engines[3](fake_frame)

# result = inference_engine.engines[3](fake_frame.
