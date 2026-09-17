"""
decode.py

Inference-time counterpart to targets.py's encode_box()/build_y_true().
Converts the model's raw output tensor into actual bounding boxes in
pixel coordinates, applying the same sigmoid/exp decoding loss.py uses
internally during training (bx=sigma(tx)+cx, bw=pw*e^tw, ...), then a
confidence threshold and non-max suppression.

This is the "YOLOv2 post process" step drawn as a separate box after the
CNN in Figure 4 of Vandersteegen et al. -- deliberately kept out of the
model itself (see model.py/layers.py's DetectionHead, which stays linear)
for the same reasons discussed earlier (encoding-agnostic model, simple
quantizable export graph).
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import tensorflow as tf


@dataclass
class Detection:
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    conf: float
    class_id: int


def decode_predictions(
    y_pred: tf.Tensor,
    anchors: np.ndarray,
    img_h: int,
    img_w: int,
    num_classes: int = 1,
    conf_threshold: float = 0.5,
    nms_iou_threshold: float = 0.3,
    max_detections: int = 100,
) -> List[Detection]:
    """Decode ONE image's raw model output into pixel-space detections.

    Parameters
    ----------
    y_pred : tf.Tensor, shape (grid_h, grid_w, num_anchors * (5 + num_classes))
        Raw output for a SINGLE image (no batch dimension) -- e.g.
        `model(x, training=False)[0]`.
    anchors : np.ndarray, shape (num_anchors, 2)
        (width, height) anchors in pixel units, matching what targets.py
        used to encode the training targets.
    img_h, img_w : int
        Input image size the model was built for (model.input_shape[:2]).
    num_classes : int
        Number of explicit class channels (0 -> no class softmax/sigmoid;
        every detection is reported as class_id=0 with confidence =
        objectness alone).
    conf_threshold : float
        Minimum score (objectness, or objectness*class-confidence if
        num_classes > 0) for a box to be kept before NMS.
    nms_iou_threshold : float
        IoU threshold for non-max suppression.
    max_detections : int
        Cap on the number of boxes returned per image.

    Returns
    -------
    List[Detection], in pixel coordinates (xtl, ytl, xbr, ybr), sorted by
    descending confidence (tf.image.non_max_suppression's own ordering).
    """
    anchors = np.asarray(anchors, dtype=np.float32)
    num_anchors = anchors.shape[0]
    depth = 5 + num_classes

    grid_h = y_pred.shape[0]
    grid_w = y_pred.shape[1]
    y_pred = tf.reshape(y_pred, (grid_h, grid_w, num_anchors, depth))

    # ---- Same decoding loss.py applies internally (Terven Fig. 7 / paper page 3) ----
    tx = tf.sigmoid(y_pred[..., 0])
    ty = tf.sigmoid(y_pred[..., 1])
    tw = y_pred[..., 2]
    th = y_pred[..., 3]
    obj_conf = tf.sigmoid(y_pred[..., 4])

    stride_y = img_h / grid_h
    stride_x = img_w / grid_w

    cy, cx = tf.meshgrid(tf.range(grid_h), tf.range(grid_w), indexing="ij")
    cx = tf.cast(cx, tf.float32)[:, :, tf.newaxis]  # (grid_h, grid_w, 1), broadcasts over anchors
    cy = tf.cast(cy, tf.float32)[:, :, tf.newaxis]

    bx = (tx + cx) * stride_x  # pixel-space center x
    by = (ty + cy) * stride_y  # pixel-space center y

    anchor_w = tf.constant(anchors[:, 0], dtype=tf.float32)  # (num_anchors,)
    anchor_h = tf.constant(anchors[:, 1], dtype=tf.float32)
    bw = anchor_w * tf.exp(tw)
    bh = anchor_h * tf.exp(th)

    x1 = bx - bw / 2.0
    y1 = by - bh / 2.0
    x2 = bx + bw / 2.0
    y2 = by + bh / 2.0

    if num_classes > 0:
        class_probs = tf.sigmoid(y_pred[..., 5:])  # matches loss.py's independent-logistic classifiers
        class_id = tf.argmax(class_probs, axis=-1, output_type=tf.int32)
        class_conf = tf.reduce_max(class_probs, axis=-1)
        score = obj_conf * class_conf
    else:
        class_id = tf.zeros_like(obj_conf, dtype=tf.int32)
        score = obj_conf

    # ---- Flatten (grid_h, grid_w, num_anchors) -> a flat list of candidate boxes ----
    boxes_yx = tf.stack([y1, x1, y2, x2], axis=-1)  # tf.image.non_max_suppression wants [y1,x1,y2,x2]
    boxes_flat = tf.reshape(boxes_yx, (-1, 4))
    scores_flat = tf.reshape(score, (-1,))
    class_flat = tf.reshape(class_id, (-1,))

    # ---- Confidence threshold BEFORE NMS (cheaper, and NMS needs a finite candidate set) ----
    keep_mask = scores_flat >= conf_threshold
    boxes_flat = tf.boolean_mask(boxes_flat, keep_mask)
    scores_flat = tf.boolean_mask(scores_flat, keep_mask)
    class_flat = tf.boolean_mask(class_flat, keep_mask)

    if tf.shape(boxes_flat)[0] == 0:
        return []

    selected = tf.image.non_max_suppression(
        boxes_flat, scores_flat, max_output_size=max_detections, iou_threshold=nms_iou_threshold
    )

    final_boxes = tf.gather(boxes_flat, selected).numpy()
    final_scores = tf.gather(scores_flat, selected).numpy()
    final_classes = tf.gather(class_flat, selected).numpy()

    detections = []
    for (y1_, x1_, y2_, x2_), conf, cls in zip(final_boxes, final_scores, final_classes):
        detections.append(Detection(
            xtl=float(x1_), ytl=float(y1_), xbr=float(x2_), ybr=float(y2_),
            conf=float(conf), class_id=int(cls),
        ))
    return detections


if __name__ == "__main__":
    # Self-test: construct a raw y_pred by hand such that, after decoding,
    # we should recover a box at a known location.
    grid_h, grid_w, num_anchors, num_classes = 6, 8, 5, 1
    img_h, img_w = 24, 32
    anchors = np.array([(5.0, 7.0), (6.0, 10.0), (8.0, 9.0), (7.0, 12.0), (9.0, 11.0)], dtype=np.float32)

    depth = 5 + num_classes
    y_pred = np.full((grid_h, grid_w, num_anchors, depth), -10.0, dtype=np.float32)  # ~0 confidence everywhere

    # Plant one confident detection at cell (row=2, col=3), anchor=1, class=0.
    cell_y, cell_x, anchor_idx = 2, 3, 1
    y_pred[cell_y, cell_x, anchor_idx, 0] = 0.0    # sigmoid(0) = 0.5 -> tx offset = 0.5
    y_pred[cell_y, cell_x, anchor_idx, 1] = 0.0    # ty offset = 0.5
    y_pred[cell_y, cell_x, anchor_idx, 2] = 0.0    # tw = 0 -> bw = anchor_w
    y_pred[cell_y, cell_x, anchor_idx, 3] = 0.0    # th = 0 -> bh = anchor_h
    y_pred[cell_y, cell_x, anchor_idx, 4] = 10.0   # sigmoid(10) ~ 1.0 objectness
    y_pred[cell_y, cell_x, anchor_idx, 5] = 10.0   # confident class 0

    y_pred_tf = tf.constant(y_pred.reshape(grid_h, grid_w, num_anchors * depth))

    detections = decode_predictions(
        y_pred_tf, anchors, img_h, img_w, num_classes=num_classes,
        conf_threshold=0.5, nms_iou_threshold=0.3,
    )

    print(f"Decoded {len(detections)} detection(s):")
    for d in detections:
        print(f"  {d}")

    stride_y, stride_x = img_h / grid_h, img_w / grid_w
    expected_cx = (cell_x + 0.5) * stride_x
    expected_cy = (cell_y + 0.5) * stride_y
    expected_w, expected_h = anchors[anchor_idx]

    assert len(detections) == 1
    d = detections[0]
    got_cx = (d.xtl + d.xbr) / 2
    got_cy = (d.ytl + d.ytb if hasattr(d, 'ytb') else d.ytl + d.ybr) / 2
    got_w = d.xbr - d.xtl
    got_h = d.ybr - d.ytl
    print(f"\nexpected center=({expected_cx:.2f},{expected_cy:.2f}) size=({expected_w:.2f},{expected_h:.2f})")
    print(f"got      center=({got_cx:.2f},{got_cy:.2f}) size=({got_w:.2f},{got_h:.2f})")
    assert abs(got_cx - expected_cx) < 1e-3
    assert abs(got_cy - expected_cy) < 1e-3
    assert abs(got_w - expected_w) < 1e-3
    assert abs(got_h - expected_h) < 1e-3
    print("\ndecode_predictions self-test PASSED")
