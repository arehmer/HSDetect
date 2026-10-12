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
from typing import List, NamedTuple, Tuple, Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from hspy.tparray import SensorTypes, TPArray

from hsdetect import Engine

IS_ENGINE_MODULE = True

@dataclass
class Detection:
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    conf: float
    class_id: int
    
class TargetIndex(NamedTuple):
    cell_y: int
    cell_x: int
    anchor_idx: int

    def as_slices(self):
        return tuple(slice(i, i + 1) for i in self)
    

class YOLO_EnDecode_Engine(Engine):
    """
    Engine providing methods for
        - encoding (ground truth) annotations to a YOLO target tensor 
        - decoding a single YOLO prediction (batch size 1) to an annotation
    
    Methods:
        - decode_yolo_prediction(): Decodes a single YOLO prediction tensor to
        an annotation format
        
        - encode_yolo_prediction(): Encode annotations for a single frame into
        a YOLO target tensor (matching the dimension of the output)
        
    """
    
    
    name = 'YOLO_EnDecode_Engine'
    
    def __init__(self,
                 SensorType : int,
                 yolo_anchors : np.ndarray, 
                 yolo_grid_size : tuple | np.ndarray | list,
                 yolo_n_classes : int):
        
    
        # Inititalize hidden atrributes otherwise init routine will fail
        self._yolo_n_classes = 0
        
        # Initialize public attributes
        self.SensorType = SensorType
        
        self.yolo_anchors = yolo_anchors      # anchors stored as (w,h)   
        self.yolo_n_classes = yolo_n_classes
        self.yolo_grid_size = yolo_grid_size  # (grid_h,grid_w)
        
        # Derive some quantities needed for conversion from relative offsets
        # predicted by yolo to absolute coordinates in pixels
        htpa = TPArray(SensorType = self.SensorType)
        w = htpa.width; h = htpa.height
        
        self._yolo_grid_cell_size = (h / self.yolo_grid_size[0] ,
                                     w / self.yolo_grid_size[1])
       
    @property
    def SensorType(self):
        return self._SensorType
    
    @SensorType.setter
    def SensorType(self,SensorType:int):
        self._SensorType = SensorType
    
    @property
    def yolo_anchors(self):
        return self._yolo_anchors
    
    @yolo_anchors.setter
    def yolo_anchors(self,yolo_anchors:np.ndarray):
        self._yolo_anchors = yolo_anchors
        self._yolo_n_anchors = len(yolo_anchors)
        self._n_anchor_channels = (5 + self.yolo_n_classes) * len(yolo_anchors)
        
    @property
    def yolo_grid_size(self):
        return self._yolo_grid_size
    
    @yolo_grid_size.setter
    def yolo_grid_size(self,yolo_grid_size : tuple | np.ndarray | list):
        self._yolo_grid_size = tuple(yolo_grid_size)
        
    @property
    def yolo_n_classes(self):
        return self._yolo_n_classes
    
    @yolo_n_classes.setter
    def yolo_n_classes(self,yolo_n_classes:int):
        self._yolo_n_classes = yolo_n_classes
        self._n_anchor_channels = (5 + yolo_n_classes) * self.yolo_n_anchors
        
    @property
    def yolo_n_anchors(self):
        return self._yolo_n_anchors
    @property
    def n_anchor_channels(self):
        return self._n_anchor_channels
    
    @property
    def yolo_grid_cell_size(self):
        return self._yolo_grid_cell_size
    
    @yolo_grid_cell_size.setter
    def yolo_grid_cell_size(self,yolo_grid_cell_size : Tuple[int,int]):
        self._yolo_grid_cell_size = yolo_grid_cell_size
     
    @property
    def yolo_stride(self):
        return self.yolo_grid_cell_size    # Stride is same as size of grid cells
    
    
    def unpack_anchor_channels(self,
                               yolo_output : tf.Tensor):
        """
        Takes a yolo output tensor with shape
        
            (n_batch x grid_h x grid_w x n_anchor_channels)
            
            with n_anchor_channels = n_anchors * (5 + n_classes)
            
        and reshapes it into a tensor with shape 
        
            (n_batch x grid_h x grid_w x n_anchors x n_channels)
        
            with n_channels = (5 + n_classes)
        
        Purpose: Lets the user then access channels (tx,ty,tw,th,obj_conf,class_1, ...)   
        associated with every anchor directly by slicing along the n_channels-axis
        instead of computing the offsets 
        
            anchor_idx * n_channels + channel_idx
            
        manually.
            
        Parameters
        ----------
        yolo_output : tf.Tensor
            DESCRIPTION.
        n_anchors : int
            DESCRIPTION.
        depth : int
            DESCRIPTION.
    
        Returns
        -------
        TYPE
            DESCRIPTION.
    
        """
        
        # Get shape of yolo output tensor
        shape = tf.shape(yolo_output)
        
        # Check dimensions 
        n_batch, grid_h, grid_w = shape[0], shape[1], shape[2]
        n_anchor_channels = shape[3]
        
        if (grid_h,grid_w) != self.yolo_grid:
            raise ValueError(f'Deducted grid size {(grid_h,grid_w)} but ' +\
                             ' expected {self.yolo_grid}')
        
                
        if n_anchor_channels != self.n_anchor_channels:
            raise ValueError(f'Deducted {n_anchor_channels} anchor channels ' +\
                             ' but expected {self.n_anchor_channels}')
        
        
        # Reshape tensor and return
        new_shape = (n_batch, grid_h, grid_w, self.yolo_n_anchors, 5+self.yolo_n_classes)
        
        return tf.reshape(yolo_output,new_shape)
    
    def constrain_channels(self, 
                           yolo_output_unpacked : tf.Tensor) -> tf.Tensor :
        """
        Applies sigmoid activation function to the tx, ty and conf channel.
        
        Expects an unpacked yolo output vector of shape 
        
            (n_batch x grid_h x grid_w x n_anchors x n_channels)
            
        so it can apply the activation by indexing along the n_channels-axis. 

        Parameters
        ----------
        yolo_output_unpacked : tf.Tensor
            Yolo output tensor with shape 
                (n_batch x grid_h x grid_w x n_anchors x n_channels).

        Returns
        -------
        tf.Tensor

        """
        
        # Check dimensions of tensor
        shape = tf.shape(yolo_output_unpacked)
        
        # If tensor has 4 dimensions, try to unpack it to 5
        if len(shape) == 4:
            yolo_output_unpacked = self.unpack_anchor_channels(yolo_output_unpacked)
        # If tensor has 5 dimensions, it is unpacked already, pass on
        elif len(shape) == 5:
            pass
        # Raise error in all other cases 
        else:
            raise ValueError('Expected yolo vector with 4 or 5 ' + \
                             'dimensions, but received tensor with ' + \
                              f'{len(shape)} dimensions.')
        
        # If tensor has the expected 5 dimensions, check if the last 4 
        # dimensions have the proper size
        exp_shape = self.yolo_grid + (self.yolo_n_anchors,) + \
            (self.n_anchor_channels,)
        
        if not shape[1::] == exp_shape:
            raise ValueError('Expected unpacked yolo vector with shape[1::]=' + \
                             '{exp_shape} but received tensor with ' + \
                              f'shape[1::]={shape[1::]}.')
                
        # Apply sigmoidal activation function to the tx, ty and conf channel
        channels = tf.unstack(yolo_output_unpacked, axis=-1)
        channels[0] = tf.sigmoid(channels[0])
        channels[1] = tf.sigmoid(channels[1])
        channels[4] = tf.sigmoid(channels[4])
        
        yolo_output_unpacked = tf.stack(channels, axis=-1)  
        
        # Return tensor with the constrained channels
        return yolo_output_unpacked
    
    
    def _yolo_predictions_to_bboxes(self,
                                    yolo_output : tf.Tensor,
                                    idx_to_annotID: Dict[TargetIndex,int] = {},
                                    ) -> List[dict]:
        """
        

        Parameters
        ----------
        t : tf.Tensor
            DESCRIPTION.

        Returns
        -------
        List[Detection]
            DESCRIPTION.

        """
        
        # ---- Convert the relative offsets to absolute image coordinates -----
        b = self._offset_to_absolute_coords(yolo_output)
        # Note: b has shape (1 x grid_h x grid_w x n_anchors x n_channels)
        
        # ---------- Convert the reulting tensor into a dictionary ------------
        
        # Drop the leading axis because batch size is assumed to be 1
        b = tf.squeeze(b, axis=0)  # (grid_h, grid_w, n_anchors, n_channels)
        
        # Positions of all non-zero-confidence predictions, shape (n_kept, 3),
        # each row is (cell_y, cell_x, anchor_idx)
        grid_idx = tf.where(b[..., 4] != 0)
        
        # Flatten grid_h, grid_w, n_anchors into a single "predictions" axis
        # b_flat = tf.reshape(b, (-1, (5 + self.yolo_n_classes)))
        b_flat = tf.gather_nd(b, grid_idx).numpy()
        grid_idx = grid_idx.numpy()
        
        # Note: b_flat has shape (n_preds, n_channels) 
        # with  n_channels = (5 + self.yolo_n_classes)
        
        # Channel 4 in b_flat contains the estimated confidence. To speed up
        # downstream processing, keep only non-zero confidence entries
        # keep = b_flat[:, 4] != 0
        # b_flat = b_flat[keep]
        
        
        # Obtain or calculate all annotation relevant quantities from yolo
        # output
        bx = b_flat[:,0:1]
        by = b_flat[:,1:2]
        bw = b_flat[:,2:3]
        bh = b_flat[:,3:4]
        
        xtl = (bx - bw / 2.0)
        ytl = (by - bh / 2.0)
        xbr = (bx + bw / 2.0)
        ybr = (by + bh / 2.0)
        
        conf = b_flat[:,4:5]
        class_probs = b_flat[:,5:]
        
        # Keep only predictions with non-zero confidence
        keep = b_flat[:, 4] != 0
        b_flat = b_flat[keep]
        
        columns = ['xtl','ytl','xbr','ybr','w','h','conf'] +\
            [f'class_{str(i)}' for i in range(self.yolo_n_classes)]
        
        data = np.hstack([xtl,ytl,xbr,ybr,bw,bh,conf,class_probs])
        
        # Create a list of dictionaries, each dicionary containing a row
        # of the 2d array data
        bboxes = []
        for row, (cy, cx, a) in zip(data, grid_idx):
            idx = TargetIndex(int(cy), int(cx), int(a))
            bbox = dict(zip(columns, row.tolist()))
            bbox['cell_y'], bbox['cell_x'], bbox['anchor_idx'] = idx
            bbox['id'] = idx_to_annotID.get(idx)   # None if no match
            bboxes.append(bbox)
    
        return bboxes    
        

    def _offset_to_absolute_coords(self,
                                   t : tf.Tensor) -> tf.Tensor:
        """
        

        Parameters
        ----------
        t : tf.Tensor
            Unpacked, constrained yolo output with shape
                (1 x grid_h x grid_w x n_anchors x n_channels)
            

        Returns
        -------
        tf.Tensor.

        """
        
       
        # Get predicted offset coordinates from respective output channels
        tx = t[..., 0]      # bounding box center x-offset over all grid cells and anchors 
        ty = t[..., 1]      # bounding box center y-offset over all grid cells and anchors 
        tw = t[..., 2]      # bounding box width offset over all grid cells and anchors 
        th = t[..., 3]      # bounding box height offset over all grid cells and anchors 
        
        # Note: tx, ty, tw, th have shape (1 x grid_h x grid_w x n_anchors)
        
        # -------------- Determine bounding box centers bx,by -----------------
        
        # Create a grid over the image in grid cell coordinates
        grid_h, grid_w = self.yolo_grid_size 
        cy, cx = tf.meshgrid(tf.range(grid_h), tf.range(grid_w), indexing="ij")
        
        # Note: cy and cx have shape (grid_h, grid_w) and contain the y and x 
        # coordinates of all grid cells. To make it compatible with tx, ty, 
        # tw and th a leading and a trailing axis need to be added
        cx = tf.cast(cx, tf.float32)[tf.newaxis, :, :, tf.newaxis] 
        cy = tf.cast(cy, tf.float32)[tf.newaxis, :, :, tf.newaxis] 
        
        # In order to obtain absolute image coordinates in pixels from the
        # relative offset coordinates predicted by yolo one needs to apply the
        # following mappings
        # bx = (tx + cx) * c_w , i.e. (x-offset + x-coordinate to grid cell) * x-stride
        # by = (ty + cy) * c_h , i.e. (y-offset + y-coordinate to grid cell) * y-stride        
        stride_y,stride_x = self.yolo_grid_cell_size
        
        bx = (tx + cx) * stride_x
        by = (ty + cy) * stride_y
        
        # -------------- Determine bounding box dimensions bw,bh -----------------
        
        # First obtain the prior dimensions
        pw = tf.constant(self.yolo_anchors[:, 0], dtype=tf.float32) # shape: (n_anchors,)
        ph = tf.constant(self.yolo_anchors[:, 1], dtype=tf.float32) # shape: (n_anchors,)
        
        # Note: pw and ph have shape (n_anchors,)
        
        # In order to make pw, ph align with tw, th missing dimensions are 
        # filled in
        pw = pw[tf.newaxis, tf.newaxis, tf.newaxis, :]
        ph = ph[tf.newaxis, tf.newaxis, tf.newaxis, :]
        
        # The bounding box dimensions in absolute coordinates bw, bh are 
        # obtained from the dimensions of the priors ph, pw and the scaling 
        # factors tw,th predicted by yolo via
        # bw = pw * e^(tw)
        # bh = ph * e^(th)
        bw = pw * tf.exp(tw)
        bh = ph * tf.exp(th)

        # Stack the absolute coordinates along a new last axis 
        b_xywh = tf.stack([bx, by, bw, bh], axis=-1)
        # Note: b_xywh now has shape (1 x grid_h x grid_w x n_anchors x 4)
        
        # Obtain the remaining channels from t
        b_rest = t[... , 4:]
        # Note: b_rest has shape (1 x grid_h x grid_w x n_anchors x n_anchor_channels - 4)
        
        # Concatenate b_xywh and b_rest aloong their last axis
        b = tf.concat([b_xywh, b_rest], axis=-1)
        # Note: b has shape (1 x grid_h x grid_w x n_anchors x n_anchor_channels)
        
        # Return
        return b
    
    def decode_yolo_prediction(self,
                               yolo_output : tf.Tensor,
                               idx_to_annotID: Dict[TargetIndex,int] = {}) -> pd.DataFrame:
        """
        Expects a packed (4 dimensions) yolo output tensor with shape
        
            (1 x grid_h x grid_w x n_anchor_channels)
        
        or an unpacked yolo output tensor with shape
        
            (1 x grid_h x grid_w x n_anchors x n_channels)
        
        This function is only ever applied frame-wise, i.e. to a batch of size
        1.
        
        This method calls constrain_channels() on yolo output automatically!
        
        
        Parameters
        ----------
        yolo_output : tf.Tensor
            DESCRIPTION.

        Returns
        -------
        List[Detection]
            DESCRIPTION.

        """
        
        # ------------------------- Check batch size --------------------------
        shape = tf.shape(yolo_output)
        
        # Batch size has to be 1
        n_batch = shape[0]
        if not n_batch == 1:
            raise ValueError(f'Expected batch size 1 but received {n_batch}.')
        
        # ----------------- Constrain tx, tx and conf channel -----------------
        if len(shape) == 4:
            yolo_output = self.unpack_anchor_channels(yolo_output)
        # If tensor has 5 dimensions, it is unpacked already, pass on
        elif len(shape) == 5:
            pass
        # Raise error in all other cases 
        else:
            raise ValueError('Expected yolo vector with 4 or 5 ' + \
                             'dimensions, but received tensor with ' + \
                              f'{len(shape)} dimensions.')
        

        
        
        # ------------- Convert the tensor to bounding boxes ------------------
        bboxes = self._yolo_predictions_to_bboxes(yolo_output,
                                                  idx_to_annotID)
        
        bboxes = pd.DataFrame.from_dict(bboxes)
        
        bboxes = bboxes.set_index('id', drop = True)
        
        return pd.DataFrame.from_dict(bboxes)
        
        
    def encode_yolo_target(self,
                           annotations : pd.DataFrame,
                           ) -> Tuple[np.ndarray, list] :
        
        
        # Check type
        if not isinstance(annotations, pd.DataFrame):
            e_msg = f'Expected {pd.DataFrame}, received {type(annotations)}.'
            raise TypeError(e_msg)
        
        # Check that image_id is given in annotations
        if not 'image_id' in annotations.columns:
            e_msg = "Expected annotations to have column 'image_id'"
            raise ValueError(e_msg)
        
        
        # 1: Convert annotations from (xtl,ytl,xbr,ybr) coordinates to 
        # (x,y,w,h) coordinates with x,y being the coordinate to the center
        # of the bounding box
        annot_center = annot_to_center_format(annotations)

        # 2: Assign every bounding box in annotations to a 
        # - a grid cell
        # - and the anchor box within that grid cell
        assignments = assign_boxes_to_gridcell_anchors(annot_center,
                                                       self.yolo_anchors,
                                                       self.yolo_grid_size,
                                                       self.yolo_stride)
        
        # 3: Initialize a zero yolo target vector with shape:
        #    (n_batch x grid_h x grid_w x n_anchors, n_channels)
        n_batch = len(annotations['image_id'].unique())  # batch size is equal to the number of images
        grid_h,grid_w = self.yolo_grid_size
        n_anchors = self.yolo_n_anchors
        n_channels = 5 + self.yolo_n_classes
        
        yolo_target = np.zeros((n_batch, grid_h, grid_w, n_anchors, n_channels),
                               dtype=np.float32)
        
        # 4: Create mappings        
        # Mapping 1: image_id -> batch_idx (sorted for determinism)
        image_ids = sorted(annot_center["image_id"].unique().tolist()) # get sorted uni
        image_id_to_batch_idx = {img_id: i for i, img_id in enumerate(image_ids)}
        
        # Mapping 2: batch_idx -> (cell_y, cell_x, anchor_idx) -> annotation_id
        idx_to_annotID: Dict[int:Dict[TargetIndex, int]] = \
            {image_id : {} for image_id in image_ids}
        
        # 5: Fill the yolo target vector with the annotations assigned to
        # grid_cell, anchor pairs
        for a in assignments:
            
            # Get batch_idx from mapping 
            batch_idx = image_id_to_batch_idx[a.image_id]

            # Create an index (cell_y, cell_x, anchor_idx) for 
            # that annotation in the tensor
            target_idx = TargetIndex(a.cell_y,
                                     a.cell_x,
                                     a.anchor_idx)
            
            # Store index to annotation in mapping
            idx_to_annotID[a.image_id][target_idx] = a.id
                        
            # Encode the absolute bounding box coordinates into the offsets / 
            # scaling factors predicted by YOLO
            tx, ty, tw, th = encode_box(a,
                                        self.yolo_anchors,
                                        self.yolo_stride)
            
            # Write the quantities to be predicted by YOLO into the appropriate
            # cells of the target tensor
            yolo_target[batch_idx][target_idx][:5] = (tx, ty, tw, th, 1.0)
            
            # Write the target conditional class probability
            if self.yolo_n_classes > 0:
                class_idx = min(a.class_id, self.yolo_n_classes - 1)
                yolo_target[batch_idx][target_idx][5 + class_idx] = 1.0
            
        return yolo_target, image_id_to_batch_idx, idx_to_annotID
        

    
def annot_to_center_format(df: pd.DataFrame) -> pd.DataFrame:
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


@dataclass
class Assignment:
    id : int
    image_id: object
    class_id: int
    cell_y: int
    cell_x: int
    anchor_idx: int
    x: float
    y: float
    w: float
    h: float

def assign_boxes_to_gridcell_anchors(
    df_center: pd.DataFrame,
    anchors: np.ndarray,
    grid_size: Tuple[int,int],      # (h,w)
    stride: Tuple[int,int],         # (y,x)
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
    
    grid_h, grid_w = grid_size
    stride_y, stride_x = stride
    
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
            id = idx,
            image_id=image_id,
            class_id=class_id,
            cell_y=cell_y,
            cell_x=cell_x,
            anchor_idx=anchor_idx,
            x=row["x"], y=row["y"], w=row["w"], h=row["h"],
        ))

    return assignments


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

def encode_box(
    assignment: Assignment,
    anchors: np.ndarray,
    stride: Tuple[int,int],
) -> Tuple[float, float, float, float]:
    """Encode one assigned box into (tx, ty, tw, th) targets, matching the
    encoding assumed by loss.py:

        tx_target = fractional x-offset within its cell, in [0, 1)
                    (this is what sigmoid(tx_pred) is trained to match)
        ty_target = fractional y-offset within its cell, in [0, 1)
        tw_target = log(w_gt / anchor_w)
        th_target = log(h_gt / anchor_h)
    """
    
    stride_y, stride_x = stride
    
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


if __name__ == "__main__":
    
    from pathlib import Path
    import pickle as pkl
    from hspy.tparray import SensorTypes
    
    # 1: Load test data
    test_data_dir = Path.cwd().parents[1] / 'tests' / 'data'
    
    img80_path = test_data_dir / 'Video2485236_Frame80_img.pkl'
    img500_path = test_data_dir / 'Video2485236_Frame500_img.pkl'
    annot_path = test_data_dir / 'Video2485236_annot.pkl'

    img80 = pkl.load(open(img80_path,'rb'))
    img500 = pkl.load(open(img500_path,'rb'))
    annot = pkl.load(open(annot_path,'rb'))  
    
    # 2. Create Encoding Engine
    anchors = np.array([[10,20],
                       [20,40]])
    grid_size = (15,20)
    n_classes = 0
    
    encode_engine = YOLO_EnDecode_Engine(SensorTypes['HTPA160x120DR1_L3K95_0K8'],
                                         anchors,
                                         grid_size,
                                         n_classes)
    
    # 3. Encode the annotations to a yolo target vector
    yolo_target, img_to_batch, idx_to_annotID = encode_engine.encode_yolo_target(annot)
    
    # 4. Decode the target tensor for frames 80 and 500 back to a readable
    # annotation
    idx80 = img_to_batch[80]
    yolo_target_80 = yolo_target[idx80:idx80+1,...] 
    
    idx500 = img_to_batch[500]
    yolo_target_500 = yolo_target[idx500:idx500+1,...] 
    
    img80_annot_yolo = encode_engine.decode_yolo_prediction(yolo_target_80,
                                                            idx_to_annotID[80])
    img500_annot_yolo = encode_engine.decode_yolo_prediction(yolo_target_500,
                                                             idx_to_annotID[500])
    
    # 5. Compare to the original ground truth annotations
    img80_annot = annot.loc[annot['image_id']==80]
    img500_annot = annot.loc[annot['image_id']==500]
    
    
    comp_cols = ['xtl','ytl','xbr','ybr']
    
    if not (img80_annot_yolo[comp_cols] ==  img80_annot[comp_cols]).all(None):
        raise ValueError('Test failed. Annotations are not identical.')
        
    if not (img500_annot_yolo[comp_cols] ==  img500_annot[comp_cols]).all(None):
        raise ValueError('Test failed. Annotations are not identical.')
        
        
# def decode_predictions(
#     y_pred: tf.Tensor,
#     anchors: np.ndarray,
#     img_h: int,
#     img_w: int,
#     num_classes: int = 1,
#     conf_threshold: float = 0.5,
#     nms_iou_threshold: float = 0.3,
#     max_detections: int = 100,
# ) -> List[Detection]:
#     """Decode ONE image's raw model output into pixel-space detections.

#     Parameters
#     ----------
#     y_pred : tf.Tensor, shape (grid_h, grid_w, num_anchors * (5 + num_classes))
#         Raw output for a SINGLE image (no batch dimension) -- e.g.
#         `model(x, training=False)[0]`.
#     anchors : np.ndarray, shape (num_anchors, 2)
#         (width, height) anchors in pixel units, matching what targets.py
#         used to encode the training targets.
#     img_h, img_w : int
#         Input image size the model was built for (model.input_shape[:2]).
#     num_classes : int
#         Number of explicit class channels (0 -> no class softmax/sigmoid;
#         every detection is reported as class_id=0 with confidence =
#         objectness alone).
#     conf_threshold : float
#         Minimum score (objectness, or objectness*class-confidence if
#         num_classes > 0) for a box to be kept before NMS.
#     nms_iou_threshold : float
#         IoU threshold for non-max suppression.
#     max_detections : int
#         Cap on the number of boxes returned per image.

#     Returns
#     -------
#     List[Detection], in pixel coordinates (xtl, ytl, xbr, ybr), sorted by
#     descending confidence (tf.image.non_max_suppression's own ordering).
#     """
#     anchors = np.asarray(anchors, dtype=np.float32)
#     num_anchors = anchors.shape[0]
#     depth = 5 + num_classes

#     grid_h = y_pred.shape[0]
#     grid_w = y_pred.shape[1]
#     y_pred = tf.reshape(y_pred, (grid_h, grid_w, num_anchors, depth))

#     # ---- Same decoding loss.py applies internally (Terven Fig. 7 / paper page 3) ----
#     tx = tf.sigmoid(y_pred[..., 0])
#     ty = tf.sigmoid(y_pred[..., 1])
#     tw = y_pred[..., 2]
#     th = y_pred[..., 3]
#     obj_conf = tf.sigmoid(y_pred[..., 4])
    
#     # size of grid cells, equals stride
#     stride_y = img_h / grid_h
#     stride_x = img_w / grid_w

#     # Create a grid over the image in grid cell coordinates
#     cy, cx = tf.meshgrid(tf.range(grid_h), tf.range(grid_w), indexing="ij")
#     cx = tf.cast(cx, tf.float32)[:, :, tf.newaxis]  # (grid_h, grid_w, 1), broadcasts over anchors
#     cy = tf.cast(cy, tf.float32)[:, :, tf.newaxis]

#     bx = (tx + cx) * stride_x  # pixel-space center x
#     by = (ty + cy) * stride_y  # pixel-space center y

#     anchor_w = tf.constant(anchors[:, 0], dtype=tf.float32)  # (num_anchors,)
#     anchor_h = tf.constant(anchors[:, 1], dtype=tf.float32)
#     bw = anchor_w * tf.exp(tw)
#     bh = anchor_h * tf.exp(th)

#     x1 = bx - bw / 2.0
#     y1 = by - bh / 2.0
#     x2 = bx + bw / 2.0
#     y2 = by + bh / 2.0

#     if num_classes > 0:
#         class_probs = tf.sigmoid(y_pred[..., 5:])  # matches loss.py's independent-logistic classifiers
#         class_id = tf.argmax(class_probs, axis=-1, output_type=tf.int32)
#         class_conf = tf.reduce_max(class_probs, axis=-1)
#         score = obj_conf * class_conf
#     else:
#         class_id = tf.zeros_like(obj_conf, dtype=tf.int32)
#         score = obj_conf

#     # ---- Flatten (grid_h, grid_w, num_anchors) -> a flat list of candidate boxes ----
#     boxes_yx = tf.stack([y1, x1, y2, x2], axis=-1)  # tf.image.non_max_suppression wants [y1,x1,y2,x2]
#     boxes_flat = tf.reshape(boxes_yx, (-1, 4))
#     scores_flat = tf.reshape(score, (-1,))
#     class_flat = tf.reshape(class_id, (-1,))

#     # ---- Confidence threshold BEFORE NMS (cheaper, and NMS needs a finite candidate set) ----
#     keep_mask = scores_flat >= conf_threshold
#     boxes_flat = tf.boolean_mask(boxes_flat, keep_mask)
#     scores_flat = tf.boolean_mask(scores_flat, keep_mask)
#     class_flat = tf.boolean_mask(class_flat, keep_mask)

#     if tf.shape(boxes_flat)[0] == 0:
#         return []

#     selected = tf.image.non_max_suppression(
#         boxes_flat, scores_flat, max_output_size=max_detections, iou_threshold=nms_iou_threshold
#     )

#     final_boxes = tf.gather(boxes_flat, selected).numpy()
#     final_scores = tf.gather(scores_flat, selected).numpy()
#     final_classes = tf.gather(class_flat, selected).numpy()

#     detections = []
#     for (y1_, x1_, y2_, x2_), conf, cls in zip(final_boxes, final_scores, final_classes):
#         detections.append(Detection(
#             xtl=float(x1_), ytl=float(y1_), xbr=float(x2_), ybr=float(y2_),
#             conf=float(conf), class_id=int(cls),
#         ))
#     return detections

