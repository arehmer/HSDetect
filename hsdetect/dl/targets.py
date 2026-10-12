"""
targets.py

Converts raw ground-truth annotations (a pandas DataFrame of axis-aligned
boxes in top-left/bottom-right pixel format) into the `y_true` tensor
consumed by loss.py's YOLOv2Loss. Five stages, matching the requested
pipeline exactly:

    1. xyxy_to_center()     -- (xtl, ytl, xbr, ybr) -> (x, y, w, h) center format
    2. kmeans_anchors()     -- IoU-based k-means on (w, h) only, no radial dependency
    3. assign_boxes()       -- assign each GT box to a grid cell + anchor
    4. encode_box()         -- convert assigned boxes to tx/ty/tw/th targets
    5. build_y_true()       -- assemble the full y_true tensor

Background subtraction is intentionally NOT part of this module (per
request) -- these functions only deal with annotation -> target-tensor
conversion; the corresponding image preprocessing lives in background.py
and is applied separately.

Expected input DataFrame
-------------------------
Required columns: 'xtl', 'ytl', 'xbr', 'ybr' (pixel coordinates).
Optional columns:
    'image_id'  -- groups boxes belonging to the same frame. If absent,
                   all rows are treated as a single image.
    'class_id'  -- integer class index in [0, num_classes). If absent,
                   all boxes are assumed to be class 0 (single-class case,
                   matching the paper's "person" detector).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Stage 1: xyxy -> center format
# ---------------------------------------------------------------------------

def xyxy_to_center(df: pd.DataFrame) -> pd.DataFrame:
    """Convert (xtl, ytl, xbr, ybr) columns into (x, y, w, h) center-format
    columns, where (x, y) is the box center and (w, h) its width/height.

    Returns a new DataFrame with the added columns; original columns are
    preserved (image_id / class_id pass through untouched if present).
    """
    required = {"xtl", "ytl", "xbr", "ybr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    out = df.copy()
    out["w"] = out["xbr"] - out["xtl"]
    out["h"] = out["ybr"] - out["ytl"]
    out["x"] = out["xtl"] + out["w"] / 2.0
    out["y"] = out["ytl"] + out["h"] / 2.0

    if (out["w"] <= 0).any() or (out["h"] <= 0).any():
        raise ValueError("Found boxes with non-positive width/height -- check xtl/ytl/xbr/ybr ordering.")

    return out


# ---------------------------------------------------------------------------
# Stage 2: IoU-based k-means anchor generation (width/height only)
# ---------------------------------------------------------------------------

def _iou_wh(wh1: np.ndarray, wh2: np.ndarray) -> np.ndarray:
    """IoU between two sets of (w, h) boxes, both placed at a common origin
    (i.e. shape-only IoU, ignoring position). This is the standard distance
    basis used for YOLOv2-style anchor k-means.

    wh1: (N, 2), wh2: (K, 2) -> returns (N, K) IoU matrix.
    """
    w1, h1 = wh1[:, 0:1], wh1[:, 1:2]          # (N, 1)
    w2, h2 = wh2[None, :, 0], wh2[None, :, 1]  # (1, K)

    inter_w = np.minimum(w1, w2)
    inter_h = np.minimum(h1, h2)
    inter = inter_w * inter_h

    area1 = w1 * h1
    area2 = (w2 * h2)
    union = area1 + area2 - inter
    return inter / np.clip(union, 1e-9, None)


def kmeans_anchors(
    wh: np.ndarray,
    k: int,
    num_iters: int = 300,
    seed: Optional[int] = 0,
    tol: float = 1e-6,
) -> np.ndarray:
    """IoU-distance k-means clustering on box widths/heights only (no
    x/y position, and no radial/location dependency -- a single global
    anchor set is produced).

    Parameters
    ----------
    wh : np.ndarray, shape (N, 2)
        Ground-truth box (width, height) pairs, in pixel units.
    k : int
        Number of anchors to produce.
    num_iters : int
        Max Lloyd's-algorithm iterations.
    seed : int or None
        RNG seed for centroid initialization (k-means++ style seeding).
    tol : float
        Convergence tolerance on total centroid movement (in IoU-distance
        space) between iterations.

    Returns
    -------
    anchors : np.ndarray, shape (k, 2)
        (width, height) anchors, sorted by area ascending.
    """
    if wh.shape[0] < k:
        raise ValueError(f"Need at least k={k} boxes to cluster, got {wh.shape[0]}.")

    rng = np.random.default_rng(seed)

    # k-means++ style initialization: pick well-spread starting centroids
    # using IoU-distance instead of Euclidean distance.
    centroids = np.empty((k, 2), dtype=np.float64)
    first_idx = rng.integers(0, wh.shape[0])
    centroids[0] = wh[first_idx]
    for i in range(1, k):
        dist_to_nearest = 1.0 - _iou_wh(wh, centroids[:i]).max(axis=1)          # Distance (1-IoU) of centroid i to all given bounding boxes
        probs = dist_to_nearest / np.clip(dist_to_nearest.sum(), 1e-9, None)    # Normalize distance by dividing through sum
        next_idx = rng.choice(wh.shape[0], p=probs)                             # Pick next centroid from wh. Indexes with larger distance (~probability) are more likely to be picked
        centroids[i] = wh[next_idx]                                             # Assign picked wh to centroid

    
    # array for storing average IoU of closest centroid
    avg_iou = np.zeros((k,), dtype=np.float64)

    # Start optimization
    for _ in range(num_iters):
        ious = _iou_wh(wh, centroids)          # (N, k), determine IoU of all N instances to all k centroids
        assignments = np.argmax(ious, axis=1)  # (N,), assign instances to centroid with largest IoU

        new_centroids = centroids.copy()       # Initialize shifted centroids as centroids from previous iteration
        
        # Loop over all centroids and shift to median of members
        for cluster_idx in range(k):
            
            members = wh[assignments == cluster_idx]                            # Determine members of centroid cluster_idx
            
            # Determine average IoU of members to their centroid
            avg_iou[cluster_idx] = \
                ious[assignments == cluster_idx,cluster_idx].mean()
            
            if len(members) == 0:
                continue  # keep previous centroid if a cluster lost all members
            # Median is the standard choice here (more robust than mean to
            # outlier box sizes, and is what the original YOLOv2 k-means
            # variant uses).
            new_centroids[cluster_idx] = np.median(members, axis=0)             # Determine median of members and assign as new centroid

        shift = np.abs(new_centroids - centroids).sum()                         # Determine how much all centroids have been shifted (Manhatten distance)
        centroids = new_centroids                                               # Establish shifted centroids as new centroid
        
        # If shifted centroids barely deviate from previous iteration -> stop
        if shift < tol:
            
            # For result interpretation calculate average IoU of members to 
            # their closest centroid
            ious = _iou_wh(wh, centroids)
            assignments = np.argmax(ious, axis=1)
            
            for cluster_idx in range(k):
                
                members = wh[assignments == cluster_idx]                            # Determine members of centroid cluster_idx
                
                # Determine average IoU of members to their centroid
                avg_iou[cluster_idx] = \
                    ious[assignments == cluster_idx,cluster_idx].mean()
                    
            break
            
            
    # Sort by area for a stable, interpretable ordering.
    areas = centroids[:, 0] * centroids[:, 1]
    order = np.argsort(areas)
    return centroids[order], avg_iou[order]


# ---------------------------------------------------------------------------
# Stage 3: assign each GT box to a grid cell + anchor
# ---------------------------------------------------------------------------

@dataclass
class Assignment:
    image_id: object
    class_id: int
    cell_y: int
    cell_x: int
    anchor_idx: int
    x: float
    y: float
    w: float
    h: float


def assign_boxes(
    df_center: pd.DataFrame,
    anchors: np.ndarray,
    grid_h: int,
    grid_w: int,
    stride_y: float,
    stride_x: float,
) -> list:
    """Assign every ground-truth box to a (grid cell, anchor) pair.

    - The grid cell is the one containing the box center.
    - The anchor is the one with highest shape-only IoU against the box.

    If two boxes land on the same (cell, anchor) slot, the later one in
    the DataFrame overwrites the earlier one and a warning is printed --
    this is a rare edge case (two same-shaped, same-cell objects) and is
    handled the same way most YOLO reference implementations handle it.

    Returns a list of `Assignment` objects (one per input box).
    """
    has_image_id = "image_id" in df_center.columns
    has_class_id = "class_id" in df_center.columns

    assignments = []
    seen_slots = {}  # (image_id, cell_y, cell_x, anchor_idx) -> row index, for collision reporting

    for idx, row in df_center.iterrows():
        image_id = row["image_id"] if has_image_id else "default"
        class_id = int(row["class_id"]) if has_class_id else 0
        
        # Divide center coordinates of box by stride to obtain the coordinates
        # of the grid cell the box center falls into
        grid_x = row["x"] / stride_x
        grid_y = row["y"] / stride_y
        
        # Round down (floor) the grid coordinates full integers
        cell_x = int(np.clip(np.floor(grid_x), 0, grid_w - 1))
        cell_y = int(np.clip(np.floor(grid_y), 0, grid_h - 1))

        box_wh = np.array([[row["w"], row["h"]]])
        anchor_idx = int(np.argmax(_iou_wh(box_wh, anchors)[0]))

        slot_key = (image_id, cell_y, cell_x, anchor_idx)
        if slot_key in seen_slots:
            print(f"[targets.assign_boxes] Warning: collision at image={image_id}, "
                  f"cell=({cell_y},{cell_x}), anchor={anchor_idx} -- overwriting "
                  f"earlier box with the one at DataFrame index {idx}.")
        seen_slots[slot_key] = idx

        assignments.append(Assignment(
            image_id=image_id,
            class_id=class_id,
            cell_y=cell_y,
            cell_x=cell_x,
            anchor_idx=anchor_idx,
            x=row["x"], y=row["y"], w=row["w"], h=row["h"],
        ))

    return assignments


# ---------------------------------------------------------------------------
# Stage 4: encode assigned boxes as tx/ty/tw/th
# ---------------------------------------------------------------------------

def encode_box(
    assignment: Assignment,
    anchors: np.ndarray,
    stride_y: float,
    stride_x: float,
) -> Tuple[float, float, float, float]:
    """Encode one assigned box into (tx, ty, tw, th) targets, matching the
    encoding assumed by loss.py:

        tx_target = fractional x-offset within its cell, in [0, 1)
                    (this is what sigmoid(tx_pred) is trained to match)
        ty_target = fractional y-offset within its cell, in [0, 1)
        tw_target = log(w_gt / anchor_w)
        th_target = log(h_gt / anchor_h)
    """
    # Calculate the grid cell coordinates
    b_x = assignment.x / stride_x
    b_y = assignment.y / stride_y
    
    # The coordinate within the grid cell is the target quantity
    tx = b_x - assignment.cell_x
    ty = b_y - assignment.cell_y

    anchor_w, anchor_h = anchors[assignment.anchor_idx]
    tw = float(np.log(assignment.w / anchor_w))
    th = float(np.log(assignment.h / anchor_h))

    return float(tx), float(ty), tw, th


# ---------------------------------------------------------------------------
# Stage 5: assemble the full y_true tensor
# ---------------------------------------------------------------------------

def build_y_true(
    df: pd.DataFrame,
    anchors: np.ndarray,
    grid_h: int,
    grid_w: int,
    img_h: float,
    img_w: float,
    num_classes: int = 1,
    image_ids: Optional[list] = None,
) -> Tuple[np.ndarray, list]:
    """Run the full pipeline (stages 1, 3, 4 -- stage 2's anchors are
    passed in already computed) and assemble the y_true tensor expected
    by loss.YOLOv2Loss.

    Parameters
    ----------
    df : pd.DataFrame
        Raw annotations with columns (xtl, ytl, xbr, ybr) and optionally
        (image_id, class_id).
    anchors : np.ndarray, shape (num_anchors, 2)
        (width, height) anchors in pixel units, e.g. from kmeans_anchors().
    grid_h, grid_w : int
        Output grid size of the detector (model.grid_shape).
    img_h, img_w : float
        Input image size the model was built for (must match model's
        fixed input_shape from model.py).
    num_classes : int
        Number of explicit class channels (0 disables the class slots,
        matching loss.py's convention).
    image_ids : list or None
        Explicit ordering of image ids to lay out along the batch
        dimension of the returned tensor. If None, the sorted unique
        values of df['image_id'] are used (or a single synthetic image
        if df has no 'image_id' column).

    Returns
    -------
    y_true : np.ndarray, shape (N, grid_h, grid_w, num_anchors, 5 + num_classes)
    image_ids : list
        The image id ordering used along the batch (N) dimension, so
        callers can align this tensor with their corresponding input
        images.
    """
    num_anchors = anchors.shape[0]
    stride_y = img_h / grid_h
    stride_x = img_w / grid_w

    df_center = xyxy_to_center(df)

    has_image_id = "image_id" in df_center.columns
    if image_ids is None:
        if has_image_id:
            image_ids = sorted(df_center["image_id"].unique().tolist())
        else:
            image_ids = ["default"]
    image_id_to_batch_idx = {img_id: i for i, img_id in enumerate(image_ids)}

    assignments = assign_boxes(df_center, anchors, grid_h, grid_w, stride_y, stride_x)

    depth = 5 + num_classes
    y_true = np.zeros((len(image_ids), grid_h, grid_w, num_anchors, depth), dtype=np.float32)

    for a in assignments:
        if a.image_id not in image_id_to_batch_idx:
            # Box belongs to an image not in the requested image_ids list; skip it.
            continue
        b = image_id_to_batch_idx[a.image_id]

        tx, ty, tw, th = encode_box(a, anchors, stride_y, stride_x)

        y_true[b, a.cell_y, a.cell_x, a.anchor_idx, 0] = tx
        y_true[b, a.cell_y, a.cell_x, a.anchor_idx, 1] = ty
        y_true[b, a.cell_y, a.cell_x, a.anchor_idx, 2] = tw
        y_true[b, a.cell_y, a.cell_x, a.anchor_idx, 3] = th
        y_true[b, a.cell_y, a.cell_x, a.anchor_idx, 4] = 1.0  # objectness

        if num_classes > 0:
            class_idx = min(a.class_id, num_classes - 1)
            y_true[b, a.cell_y, a.cell_x, a.anchor_idx, 5 + class_idx] = 1.0

    return y_true, image_ids


if __name__ == "__main__":
    # Self-test with a small synthetic annotation set, mirroring a
    # (H=24, W=32) input / (grid_h=6, grid_w=8) output setup, i.e. the
    # native MLX90640 resolution and this project's model.grid_shape.

    rng = np.random.default_rng(0)
    rows = []
    for image_id in range(3):
        n_boxes = rng.integers(1, 4)
        for _ in range(n_boxes):
            w = rng.uniform(4, 10)
            h = rng.uniform(6, 14)
            cx = rng.uniform(w / 2, 32 - w / 2)
            cy = rng.uniform(h / 2, 24 - h / 2)
            rows.append({
                "image_id": image_id,
                "class_id": 0,
                "xtl": cx - w / 2, "ytl": cy - h / 2,
                "xbr": cx + w / 2, "ybr": cy + h / 2,
            })
    df = pd.DataFrame(rows)
    print("Raw annotations:\n", df, "\n")

    # Stage 1
    df_center = xyxy_to_center(df)
    print("Center-format annotations:\n", df_center[["image_id", "x", "y", "w", "h"]], "\n")

    # Stage 2
    wh = df_center[["w", "h"]].to_numpy()
    anchors = kmeans_anchors(wh, k=5, seed=0)
    print("Anchors (w, h):\n", anchors, "\n")

    # Stages 3-5
    grid_h, grid_w = 6, 8
    img_h, img_w = 24, 32
    y_true, image_ids = build_y_true(
        df, anchors, grid_h=grid_h, grid_w=grid_w, img_h=img_h, img_w=img_w, num_classes=1,
    )
    print("y_true shape:", y_true.shape)  # (3, 6, 8, 5, 6)
    print("image_ids order:", image_ids)
    print("Total assigned object slots:", int(y_true[..., 4].sum()), "== total boxes:", len(df))
