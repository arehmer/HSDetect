"""
video_dataset.py

Standalone logic for converting a sequence of frames (each exposing an
`.image` array and an `.annotations` DataFrame with xtl/ytl/xbr/ybr
columns -- i.e. matching `AnnotatedFrame`'s interface in annot.py) into a
`tf.data.Dataset` of (image, y_true) pairs, ready for training.

This module is intentionally decoupled from `annot.py` itself (which pulls
in project-specific dependencies like `hsod`/`hspy` that may not be
importable in every environment) -- it only relies on duck typing:
anything with `.image` (a 2D/3D array) and `.annotations` (a DataFrame
with xtl/ytl/xbr/ybr, and optionally a class label column) works.
`AnnotatedVideo.to_yolo_dataset()` in annot.py is a thin wrapper around
`build_video_dataset()` below.

Assumes images are ALREADY background-subtracted (per request) -- no
background handling happens here.
"""

from typing import List, Tuple, Sequence, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from hsod.dl.targets import xyxy_to_center, assign_boxes, encode_box


def build_frame_targets(
    image: np.ndarray,
    annotations: Optional[pd.DataFrame],
    anchors: np.ndarray,
    grid_h: int,
    grid_w: int,
    num_classes: int = 1,
    class_column: str = "label",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the (image, y_true) pair for a single frame.

    Parameters
    ----------
    image : np.ndarray
        Background-subtracted frame, shape (H, W) or (H, W, 1).
    annotations : pd.DataFrame or None
        Bounding boxes for this frame in (xtl, ytl, xbr, ybr) format.
        May be empty or None (frame with no objects) -- y_true will then
        be all zeros. If a `class_column` (default 'label', matching
        AnnotatedFrame.cv_datatypes) is present, it's used as the class
        id; otherwise every box defaults to class 0.
    anchors : np.ndarray, shape (num_anchors, 2)
        (width, height) anchor boxes in pixel units, same units as the
        annotation coordinates.
    grid_h, grid_w : int
        Output grid size of the detector (matches model.grid_shape for
        this frame's resolution).
    num_classes : int
        Number of explicit class channels (0 disables class slots,
        matching loss.py's convention).
    class_column : str
        Column in `annotations` holding the integer class label, if any.
    

    Returns
    -------
    image_arr : np.ndarray, shape (H, W, 1), float32
    y_true : np.ndarray, shape (grid_h, grid_w, num_anchors, 5 + num_classes), float32
    """
    num_anchors = anchors.shape[0]
    depth = 5 + num_classes
    y_true = np.zeros((grid_h, grid_w, num_anchors, depth), dtype=np.float32)

    image_arr = np.asarray(image, dtype=np.float32)
    if image_arr.ndim == 2:
        image_arr = image_arr[..., np.newaxis]
    img_h, img_w = image_arr.shape[0], image_arr.shape[1]

    if annotations is not None and len(annotations) > 0:
        df = annotations.copy()
        if class_column in df.columns and "class_id" not in df.columns:
            df["class_id"] = df[class_column]
        
        # Add x,y,w,h bounding box coordinates
        df_center = xyxy_to_center(df)
        
        # Calculate the stride by dividing input size by grid cell size
        stride_y = img_h / grid_h
        stride_x = img_w / grid_w
        
        # Assign each of the bounding boxes to a grid cell and an anchor
        assignments = assign_boxes(df_center, anchors, grid_h, grid_w, stride_y, stride_x)
        
        # Compile the target vector.
        for a in assignments:
            
            # Determine target quantities tx, ty, tw, th 
            tx, ty, tw, th = encode_box(a, anchors, stride_y, stride_x)
            
            # Assign target quantities to the proper entries of the target 
            # vector (all remaining entries are left to be zero)
            y_true[a.cell_y, a.cell_x, a.anchor_idx, 0] = tx
            y_true[a.cell_y, a.cell_x, a.anchor_idx, 1] = ty
            y_true[a.cell_y, a.cell_x, a.anchor_idx, 2] = tw
            y_true[a.cell_y, a.cell_x, a.anchor_idx, 3] = th
            
            # The likelihood of an object being in that box is always one
            y_true[a.cell_y, a.cell_x, a.anchor_idx, 4] = 1.0
            
            # Assign the conditional class probability
            if num_classes > 0:
                class_idx = min(int(a.class_id), num_classes - 1)
                y_true[a.cell_y, a.cell_x, a.anchor_idx, 5 + class_idx] = 1.0

    return image_arr, y_true




def build_video_dataset(
    frames: Sequence,
    anchors: List[Tuple[float, float]],
    grid_shape: Tuple[int, int],
    num_classes: int = 1,
    class_column: str = "label",
    sequence_id: Optional[int] = None
) -> tf.data.Dataset:
    """Loop over a sequence of frame-like objects (each with `.image` and
    `.annotations`) and build a tf.data.Dataset of (image, y_true) pairs.

    Parameters
    ----------
    frames : sequence of objects exposing `.image` and `.annotations`
        Typically `AnnotatedVideo.frames` (a list of `AnnotatedFrame`).
    anchors : list of (w, h) tuples
        Anchor boxes, pixel units, shared across the whole video (and
        typically across the whole training set -- see note in the
        accompanying explanation about computing these globally).
    grid_shape : (grid_h, grid_w)
        Output grid size of the detector.
    num_classes : int
        Number of explicit class channels (0 to disable).
    class_column : str
        Column name in each frame's annotations holding the class label.
   sequence_id : int or None
        If given, every element additionally carries this integer id (as
        an int64 scalar), so multiple videos' datasets can later be
        combined and re-split by sequence -- see combine_sequence_caches()
        and split_by_sequence_id() below. If None (default), elements are
        plain (image, y_true) pairs, matching the original behavior.
        NOTE: don't mix sequence_id=None and sequence_id=<int> datasets
        that you intend to concatenate/combine later -- their element
        structures differ and concatenation will fail.

    Returns
    -------
    tf.data.Dataset
        Yields (image, y_true) pairs (or (image, y_true, sequence_id)
        triples), one per frame, in the same order as `frames`. Images
        have shape (H, W, 1); y_true has shape
        (grid_h, grid_w, num_anchors, 5 + num_classes).
    """
    if len(frames) == 0:
        raise ValueError("build_video_dataset() received an empty frame sequence.")

    anchors_arr = np.asarray(anchors, dtype=np.float32)
    grid_h, grid_w = grid_shape

    images_list = []
    y_true_list = []

    for frame in frames:
        # If the frame is flagged as no_cv, it is not compiled as part of the
        # dataset
        if 'no_cv' in frame.metadata.keys():
            if frame.metadata['no_cv'] == True:
                continue
        image_arr, y_true = build_frame_targets(
            image=frame.image,
            annotations=frame.annotations,
            anchors=anchors_arr,
            grid_h=grid_h,
            grid_w=grid_w,
            num_classes=num_classes,
            class_column=class_column,
        )
        images_list.append(image_arr)
        y_true_list.append(y_true)
    
    # Convert images and y_true lists to numpy arrays
    images_arr = np.stack(images_list, axis=0)
    y_true_arr = np.stack(y_true_list, axis=0)
    
    # If no sequence_id is given, compile the pairs (image, y_true) into a 
    # tf.data.Dataset and return
    if sequence_id is None:
        return tf.data.Dataset.from_tensor_slices((images_arr, y_true_arr))

    # Otherwise create an additional array which contains the sequence_id at
    # every element, thereby assigning the sequence_id to every (image, y_true)
    # pair -> (image, y_true, sequence_id)
    seq_id_arr = np.full((len(images_arr),), sequence_id, dtype=np.int64)
    return tf.data.Dataset.from_tensor_slices((images_arr,
                                               y_true_arr,
                                               seq_id_arr))

if __name__ == "__main__":
    # Self-test using lightweight mock objects that duck-type the
    # AnnotatedFrame interface (`.image`, `.annotations`), so this module
    # is fully testable without importing annot.py's heavier dependencies.

    class _MockFrame:
        def __init__(self, image, annotations):
            self.image = image
            self.annotations = annotations

    H, W = 24, 32
    grid_h, grid_w = 6, 8
    rng = np.random.default_rng(0)

    frames = []
    for i in range(4):
        img = rng.normal(0, 1, size=(H, W)).astype(np.float32)  # already "background-subtracted"
        n_boxes = rng.integers(0, 3)  # some frames have zero objects
        rows = []
        for _ in range(n_boxes):
            w = rng.uniform(4, 8)
            h = rng.uniform(6, 12)
            cx = rng.uniform(w / 2, W - w / 2)
            cy = rng.uniform(h / 2, H - h / 2)
            rows.append({
                "xtl": cx - w / 2, "ytl": cy - h / 2,
                "xbr": cx + w / 2, "ybr": cy + h / 2,
                "label": 0,
            })
        annot = pd.DataFrame(rows, columns=["xtl", "ytl", "xbr", "ybr", "label"])
        frames.append(_MockFrame(img, annot))

    anchors = [(5.0, 7.0), (6.0, 10.0), (8.0, 9.0), (7.0, 12.0), (9.0, 11.0)]

    dataset = build_video_dataset(frames, anchors=anchors, grid_shape=(grid_h, grid_w), num_classes=1)

    print("Dataset element spec:", dataset.element_spec)
    total_objects_in = sum(len(f.annotations) for f in frames)
    total_objects_out = 0
    for image, y_true in dataset:
        assert image.shape == (H, W, 1)
        assert y_true.shape == (grid_h, grid_w, 5, 6)
        total_objects_out += int(tf.reduce_sum(y_true[..., 4]).numpy())

    print("Frames in dataset:", sum(1 for _ in dataset))
    print("Total GT objects across frames:", total_objects_in)
    print("Total object slots set in y_true:", total_objects_out)
    assert total_objects_in == total_objects_out, "Mismatch between GT boxes and assigned y_true slots!"
    print("Self-test passed.")
