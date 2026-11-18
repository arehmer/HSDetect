"""
DLEngine: runs the detection network, exported as ONNX, with ONNX Runtime.

The ONNX model (exported by AlexYoloPureSec, yolotiny.training.save_detector)
contains the box decoding. For one frame it returns

    (1, grid_h * grid_w * num_anchors, 5 + C)

rows ordered by (cell_y, cell_x, anchor): xtl, ytl, xbr, ybr, conf,
class_0 .. class_{C-1}, in pixels and with sigmoid applied. Thresholding and
NMS follow in YOLO_EnDecode_Engine.
"""
from pathlib import Path

import numpy as np

from hspy.tparray import TPArray
from hsdetect import Engine, Payload
from hsdetect.kernels import make_net_input

IS_ENGINE_MODULE = True

def build_dl_engine(**kwargs):
    name = kwargs.pop('name')
    if name == 'DLEngine':
        return DLEngine(**kwargs)
    else:
        raise TypeError(f'Unknown Inference Engine {name}!')


import ast
def get_classes():
    with open(__file__, 'r') as f:
        tree = ast.parse(f.read(), filename=__file__)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


class DLEngine(Engine):

    name = 'DLEngine'
    # accepts is set from the model's channel count, see __init__.
    accepts = (Payload.FRAME_MASKED, Payload.FRAME_STACKED)
    emits = Payload.TENSOR

    def __init__(self,
             SensorType:int,
             **attr_dict:dict):

        self.SensorType = SensorType
        self.htpa = TPArray(SensorType = SensorType)
        self.dl_model = attr_dict.pop('dl_model',None)

    @property
    def SensorType(self):
        return self._SensorType

    @SensorType.setter
    def SensorType(self,SensorType:int):
        self._SensorType = SensorType

    @property
    def dl_model(self):
        """Path of the ONNX model (str), None if no model is loaded."""
        return self._dl_model

    @dl_model.setter
    def dl_model(self,dl_model):
        self._session = None
        self._dl_model = None

        # Fall back to the class contract while no model is loaded, so that a
        # previous model's channel count does not linger on this instance.
        self.__dict__.pop('accepts', None)

        if dl_model is None:
            return
        if not isinstance(dl_model, (str, Path)) or Path(dl_model).suffix != '.onnx':
            raise ValueError(f'dl_model must be the path of an .onnx model, got {dl_model!r}.')

        import onnxruntime as ort
        self._session = ort.InferenceSession(str(dl_model),
                                             providers=['CPUExecutionProvider'])
        self._input_name = self._session.get_inputs()[0].name
        self._dl_model = Path(dl_model).as_posix()

        # The contract follows the model: a one-channel model takes the masked
        # frame, a two-channel one the stacked frame. Declaring it here is what
        # retires InferenceEngine._check_input_channels(), which had to name
        # DLEngine and BackgroundSub literally to compare the two.
        channels = self.input_shape[-1]
        self.accepts = (Payload.FRAME_STACKED if channels == 2
                        else Payload.FRAME_MASKED)

    @property
    def input_shape(self) -> tuple:
        """Input shape of the model without the batch axis, e.g. (120, 160, 2)."""
        if self._session is None:
            return None
        return tuple(self._session.get_inputs()[0].shape[1:])

    def __call__(self,data):

        frame = data['frame']

        if frame is None:
            return None
        else:
            # make_net_input() casts to float32 and promotes a 2-D frame to
            # (H,W,1), as in training. A second channel, if any, arrives
            # already stacked and normalized from BackgroundSub. The batch
            # axis is added by indexing.
            frame = make_net_input(frame)[np.newaxis, ...]
            return self._session.run(None, {self._input_name: frame})[0]

    def save(self):
        return {'name': self.name,
                'SensorType': self.SensorType,
                'dl_model': self.dl_model}
