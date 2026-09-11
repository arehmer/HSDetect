# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 12:50:10 2025

@author: rehmer
"""

def iou_same_corner(w1, h1, w2, h2):
    """
    Computes IoU for two rectangles that share the same top-left corner.

    Parameters:
        w1, h1 : width and height of rectangle 1
        w2, h2 : width and height of rectangle 2

    Returns:
        IoU (float)
    """
    # Intersection dimensions
    inter_w = min(w1, w2)
    inter_h = min(h1, h2)
    inter_area = inter_w * inter_h

    # Areas of each rectangle
    area1 = w1 * h1
    area2 = w2 * h2

    # Union = sum of areas - intersection
    union_area = area1 + area2 - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    return inter_area / union_area