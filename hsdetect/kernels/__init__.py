# -*- coding: utf-8 -*-
"""
hsdetect.kernels -- the numerical core of the preprocessing chain.

NumPy only: no cv2, no hspy, no tensorflow, no configuration objects and no
engine classes. Every function takes arrays and plain numbers and returns
arrays, so that the same code runs

  * at inference time, wrapped by the engines in hsdetect.cv.preproc_engines,
  * while preparing training data, in HSDetectTrain's yolotiny.preprocess,

instead of each end carrying its own copy. The engines keep the chain's
configuration and state in properties, because Engine.save() serializes a
chain by walking them; this package keeps the arithmetic.

The device's C++ filter cv::htpa::AWA_Filter implements the same algorithm as
awa.AWAFilter. That duplication is deliberate -- it crosses a language
boundary -- and tests/test_awa_conformance.py holds the two to the same
output.
"""

from hsdetect.kernels.awa import AWAFilter
from hsdetect.kernels.background import (BackgroundSubtractor,
                                         box_mask,
                                         subtract_background,
                                         update_background)
from hsdetect.kernels.mask import detection_mask, zero_corners
from hsdetect.kernels.scene import (estimate_scene_level,
                                    make_net_input,
                                    net_input,
                                    stack_channels,
                                    subtract_scene_level)

__all__ = ['AWAFilter',
           'BackgroundSubtractor',
           'box_mask',
           'detection_mask',
           'estimate_scene_level',
           'make_net_input',
           'net_input',
           'stack_channels',
           'subtract_background',
           'subtract_scene_level',
           'update_background',
           'zero_corners']
