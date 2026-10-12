# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:12:43 2026

@author: rehmer
"""

"""
augment.py
 
Flip augmentation applied directly to already-ENCODED (image, y_true)
pairs -- not to AnnotatedVideo/raw annotations. This deliberately avoids
re-running background subtraction or target encoding for augmented
copies; see the accompanying explanation for the derivation and
verification of the closed-form transform used here.
 
Box width/height (tw, th) and anchor assignment are invariant under a
flip (flipping doesn't change a box's size), so only:
  - the image itself,
  - the spatial grid axis (which cell a detection lives in), and
  - the corresponding offset channel (tx or ty)
need to change. Objectness and class channels are untouched.
"""
 
import tensorflow as tf
 


def flip_horizontal(image: tf.Tensor, y_true: tf.Tensor):
    """Flip left-right. Reverses the image's width axis (1) and y_true's
    grid-width axis (1); tx -> 1 - tx.
    """
    image = tf.reverse(image, axis=[1])
    y_true = tf.reverse(y_true, axis=[1])
    tx = y_true[..., 0:1]
    rest = y_true[..., 1:]
    y_true = tf.concat([1.0 - tx, rest], axis=-1)
    return image, y_true
 
 
def flip_vertical(image: tf.Tensor, y_true: tf.Tensor):
    """Flip top-bottom. Reverses the image's height axis (0) and y_true's
    grid-height axis (0); ty -> 1 - ty.
    """
    image = tf.reverse(image, axis=[0])
    y_true = tf.reverse(y_true, axis=[0])
    ty = y_true[..., 1:2]
    y_true = tf.concat([y_true[..., 0:1], 1.0 - ty, y_true[..., 2:]], axis=-1)
    return image, y_true

def expand_with_flips(
    dataset: tf.data.Dataset,
    hflip: bool = True,
    vflip: bool = True,
    rotate: bool = False,
) -> tf.data.Dataset:
    """Deterministically build a LARGER dataset containing the original
    elements plus their flipped variants, all present simultaneously --
    as opposed to random_flip_augment(), which stochastically replaces
    each element with a (possibly) flipped version on every epoch.
 
    Parameters
    ----------
    dataset : tf.data.Dataset of (image, y_true) pairs.
    hflip : include a horizontally-flipped copy of every element.
    vflip : include a vertically-flipped copy of every element.
    rotate : also include a copy with BOTH flips applied (i.e. a
        180-degree rotation). Only takes effect if horizontal and
        vertical are both True.
 
    Returns
    -------
    tf.data.Dataset containing the original elements followed by each
    requested flipped variant, concatenated -- e.g. with horizontal=True,
    vertical=True, include_both=False, the result has 3x as many elements
    as the input (original + h-flip + v-flip), in that order.
    """
    # Initialize a list collecting the original dataset and all of its variants
    variants = [dataset]
    
    if hflip:
        # apply horizontal flip to every element in dataset and append flipped
        # dataset to list of variants
        variants.append(dataset.map(flip_horizontal, num_parallel_calls=tf.data.AUTOTUNE))
    if vflip:
        # apply horizontal flip to every element in dataset and append flipped
        # dataset to list of variants
        variants.append(dataset.map(flip_vertical, num_parallel_calls=tf.data.AUTOTUNE))
    if rotate and hflip and vflip:
        # apply horizontal and vertical flip (=180° rotation) to every element 
        # in dataset and append flipped dataset to list of variants
        variants.append(dataset.map(
            lambda image, y_true: flip_vertical(*flip_horizontal(image, y_true)),
            num_parallel_calls=tf.data.AUTOTUNE,
        ))
    
    # Concatenate all variants to one large dataset
    combined = variants[0]
    for v in variants[1:]:
        combined = combined.concatenate(v)
        
    # Return large dataset
    return combined

