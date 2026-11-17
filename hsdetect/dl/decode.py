"""
decode.py

YOLO_EnDecode_Engine: the post-processing after the network. The ONNX model
(see DLEngine) already decodes its output into boxes in pixels; this engine
applies the confidence threshold and non-max suppression, clips the boxes to
the image and returns them as a DataFrame.

Training, including the encoding of annotations into YOLO targets, lives in
the AlexYoloPureSec project (yolotiny).
"""

import numpy as np
import pandas as pd

from hspy.tparray import TPArray

from hsdetect import Engine

IS_ENGINE_MODULE = True

def build_decoding_engine(**kwargs):
    name = kwargs.pop('name')
    if name == 'YOLO_EnDecode_Engine':
        return YOLO_EnDecode_Engine(**kwargs)
    else:
        raise TypeError(f'Unknown Inference Engine {name}!')

import ast
def get_classes():
    with open(__file__, 'r') as f:
        tree = ast.parse(f.read(), filename=__file__)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


class YOLO_EnDecode_Engine(Engine):
    """
    Decoded network output of one frame -> DataFrame of boxes.

    Input: (1, grid_h * grid_w * n_anchors, 5 + yolo_n_classes), rows ordered
    by (cell_y, cell_x, anchor): xtl, ytl, xbr, ybr, conf, class_0 .. , as
    returned by DLEngine, or None while the background is not established.

    Output: DataFrame with the columns
        xtl, ytl, xbr, ybr, w, h, conf, class_0 .., cell_y, cell_x, anchor_idx
    and the index 'id', boxes with conf >= conf_threshold after NMS.

    Parameters
    ----------
    SensorType : int
        Sensor, gives the image size the boxes are clipped to.
    yolo_grid_size : (grid_h, grid_w)
        Output grid of the network, for cell_y / cell_x.
    yolo_n_classes : int
        Number of class channels of the network (0: person only).
    conf_threshold : float
        Boxes with an objectness below this value are dropped. Every grid
        cell / anchor carries a prediction, so without a threshold the full
        grid would come back as detections (and freeze the background
        estimate in BackgroundSub.update_bg()).
    nms_iou_threshold : float or None
        Of two boxes overlapping by more than this IoU only the more
        confident one is kept. Slightly below the usual 0.45-0.5 because
        neighbouring cells tend to predict the same object. None: off.
    clip_boxes : bool
        Clip the boxes to the image after NMS (w, h then describe the visible
        part only).
    """

    name = 'YOLO_EnDecode_Engine'

    def __init__(self,
                 SensorType : int,
                 yolo_grid_size : tuple | np.ndarray | list,
                 yolo_n_classes : int,
                 conf_threshold : float = 0.5,
                 nms_iou_threshold : float = 0.4,
                 clip_boxes : bool = True):

        self.SensorType = SensorType
        self.yolo_grid_size = yolo_grid_size      # (grid_h, grid_w)
        self.yolo_n_classes = yolo_n_classes
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.clip_boxes = clip_boxes

        htpa = TPArray(SensorType = self.SensorType)
        self.img_size = (htpa.height, htpa.width)

    @property
    def SensorType(self):
        return self._SensorType

    @SensorType.setter
    def SensorType(self, SensorType : int):
        self._SensorType = SensorType

    @property
    def yolo_grid_size(self):
        return self._yolo_grid_size

    @yolo_grid_size.setter
    def yolo_grid_size(self, yolo_grid_size : tuple | np.ndarray | list):
        self._yolo_grid_size = tuple(int(v) for v in yolo_grid_size)

    @property
    def yolo_n_classes(self):
        return self._yolo_n_classes

    @yolo_n_classes.setter
    def yolo_n_classes(self, yolo_n_classes : int):
        self._yolo_n_classes = int(yolo_n_classes)

    @property
    def conf_threshold(self):
        return self._conf_threshold

    @conf_threshold.setter
    def conf_threshold(self, conf_threshold : float):
        self._conf_threshold = float(conf_threshold)

    @property
    def nms_iou_threshold(self):
        return self._nms_iou_threshold

    @nms_iou_threshold.setter
    def nms_iou_threshold(self, nms_iou_threshold : float):
        self._nms_iou_threshold = None if nms_iou_threshold is None \
            else float(nms_iou_threshold)

    @property
    def clip_boxes(self):
        return self._clip_boxes

    @clip_boxes.setter
    def clip_boxes(self, clip_boxes : bool):
        self._clip_boxes = bool(clip_boxes)

    def _columns(self) -> list:
        return ['xtl', 'ytl', 'xbr', 'ybr', 'w', 'h', 'conf'] + \
            [f'class_{i}' for i in range(self.yolo_n_classes)] + \
            ['cell_y', 'cell_x', 'anchor_idx']

    def _empty_annotations(self) -> pd.DataFrame:
        # Same columns and index as a non-empty result, so that consumers
        # (BackgroundSub.update_bg(), the plotting processes, ...) can index
        # it instead of having to special-case zero rows
        return pd.DataFrame(columns = self._columns(),
                            index = pd.Index([], name = 'id'))

    def __call__(self, output : np.ndarray) -> pd.DataFrame:

        if output is None:
            return self._empty_annotations()

        output = np.asarray(output, dtype = np.float32)
        if output.ndim == 3:
            if output.shape[0] != 1:
                raise ValueError(f'Expected batch size 1 but received {output.shape[0]}.')
            output = output[0]

        grid_h, grid_w = self.yolo_grid_size
        depth = 5 + self.yolo_n_classes
        n_rows = output.shape[0]
        if output.ndim != 2 or output.shape[1] != depth or n_rows % (grid_h * grid_w):
            raise ValueError(f'Expected decoded output of shape '
                             f'(grid_h * grid_w * n_anchors, {depth}) with grid '
                             f'{self.yolo_grid_size}, got {output.shape}. Is the '
                             f'model an ONNX export that includes the decoding?')
        n_anchors = n_rows // (grid_h * grid_w)

        # Threshold
        rows = np.flatnonzero(output[:, 4] >= self.conf_threshold)
        if len(rows) == 0:
            return self._empty_annotations()

        # Non-max suppression (greedy, class-agnostic). The kept boxes come
        # out ordered by descending confidence.
        if self.nms_iou_threshold is not None and len(rows) > 1:
            boxes = output[rows, :4].astype(np.float64)
            scores = output[rows, 4].astype(np.float64)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            order = scores.argsort()[::-1]
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                rest = order[1:]
                xtl = np.maximum(boxes[i, 0], boxes[rest, 0])
                ytl = np.maximum(boxes[i, 1], boxes[rest, 1])
                xbr = np.minimum(boxes[i, 2], boxes[rest, 2])
                ybr = np.minimum(boxes[i, 3], boxes[rest, 3])
                inter = np.clip(xbr - xtl, 0, None) * np.clip(ybr - ytl, 0, None)
                union = areas[i] + areas[rest] - inter
                iou = inter / np.clip(union, 1e-9, None)
                order = rest[iou <= self.nms_iou_threshold]
            rows = rows[keep]

        det = output[rows].astype(np.float64)
        boxes = det[:, :4]
        if self.clip_boxes:
            h, w = self.img_size
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(w))
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(h))

        cell, anchor_idx = np.divmod(rows, n_anchors)
        cell_y, cell_x = np.divmod(cell, grid_w)

        df = pd.DataFrame(boxes, columns = ['xtl', 'ytl', 'xbr', 'ybr'],
                          index = pd.Index(range(len(rows)), name = 'id'))
        df['w'] = df['xbr'] - df['xtl']
        df['h'] = df['ybr'] - df['ytl']
        df['conf'] = det[:, 4]
        for i in range(self.yolo_n_classes):
            df[f'class_{i}'] = det[:, 5 + i]
        df['cell_y'], df['cell_x'], df['anchor_idx'] = cell_y, cell_x, anchor_idx

        return df[self._columns()]

    def save(self):
        return {'name': self.name,
                'SensorType': self.SensorType,
                'yolo_grid_size': self.yolo_grid_size,
                'yolo_n_classes': self.yolo_n_classes,
                'conf_threshold': self.conf_threshold,
                'nms_iou_threshold': self.nms_iou_threshold,
                'clip_boxes': self.clip_boxes}
