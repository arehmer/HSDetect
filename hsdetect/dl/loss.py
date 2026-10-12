"""
loss.py

Configurable YOLO-style loss function, following the three-term structure
shown in Figure 5 of Terven & Cordova-Esparaza (2023):

    total loss = localization loss + confidence loss + classification loss

Each term can be independently switched on/off via constructor flags, so
you can reproduce ablations like the ones in Vandersteegen et al. Section
4.2 (e.g. training with/without particular loss components).

Adaptation note
----------------
Terven's Figure 5 shows the original YOLOv1 loss, which operates on
(x, y, sqrt(w), sqrt(h)) directly against a single predicted box per grid
cell. Vandersteegen et al.'s network is YOLOv2-style (anchor-based, 5
boxes/cell), matching the encoding described in Terven Section 5 / Figure 7:

    bx = sigmoid(tx) + cx        by = sigmoid(ty) + cy
    bw = pw * exp(tw)            bh = ph * exp(th)
    confidence = sigmoid(to)

This implementation keeps Figure 5's three-term *structure* (localization /
confidence / classification, each independently maskable and toggleable)
but applies it to YOLOv2's anchor-relative parameterization instead of
YOLOv1's grid-relative one:

  - localization loss: squared error on (sigmoid(tx), sigmoid(ty)) against
    the encoded ground-truth cell offsets, and squared error on (tw, th)
    directly in log-space against the encoded ground-truth log-ratios
    (this replaces YOLOv1's sqrt(w), sqrt(h) trick, which was specifically
    a workaround for regressing raw width/height and is unnecessary once
    width/height are already regressed in log-space).
  - confidence loss: squared error between predicted objectness
    (sigmoid(to)) and IoU-based target, for both object and no-object
    cells (weighted by lambda_coord / lambda_noobj respectively, as in
    Figure 5).
  - classification loss: squared error on class probabilities, only for
    cells that contain an object (only meaningful if the model was built
    with num_classes > 0 in model.py).

Ground-truth encoding
---------------------
`y_true` is expected to already be encoded per grid cell / anchor, with
shape (batch, grid_h, grid_w, num_anchors, 5 + num_classes):

    y_true[..., 0] = tx_target       (ground-truth sigmoid-space x offset, in [0,1])
    y_true[..., 1] = ty_target       (ground-truth sigmoid-space y offset, in [0,1])
    y_true[..., 2] = tw_target       (ground-truth log-space width ratio: log(w_gt/anchor_w))
    y_true[..., 3] = th_target       (ground-truth log-space height ratio: log(h_gt/anchor_h))
    y_true[..., 4] = objectness_mask (1 if an object is assigned to this
                                       cell/anchor, 0 otherwise)
    y_true[..., 5:] = one-hot class label (only used if num_classes > 0)

`y_pred` is the raw model output, shape
(batch, grid_h, grid_w, num_anchors * (5 + num_classes)), i.e. straight out
of model.py's DetectionHead, before any sigmoid/exp decoding -- this loss
does that decoding internally.
"""

import tensorflow as tf

class Loss:
    pass

class YOLOv2Loss(Loss):
    """Configurable YOLOv2-style detection loss.

    Parameters
    ----------
    num_anchors : int
        Number of anchors per grid cell.
    num_classes : int
        Number of explicit classes (0 to disable the classification term
        entirely, matching the paper's single implicit-class design).
    lambda_coord : float
        Weight for the localization loss and for the "object" confidence
        term (paper/Terven Figure 5 use 5.0 for localization).
    lambda_noobj : float
        Down-weighting factor for the confidence loss in cells without an
        object (paper/Terven Figure 5 use 0.5).
    use_localization, use_confidence, use_classification : bool
        Independently enable/disable each of the three loss terms. This is
        the "pick which parts of the loss you want" mechanism requested --
        each flag simply zeroes out that term's contribution to the total
        when set to False.
    """

    def __init__(
        self,
        num_anchors: int,
        num_classes: int = 0,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        use_localization: bool = True,
        use_confidence: bool = True,
        use_classification: bool = True,
    ):
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.use_localization = use_localization
        self.use_confidence = use_confidence
        self.use_classification = use_classification and num_classes > 0

    def _reshape_pred(self, y_pred):
        shape = tf.shape(y_pred)
        batch, grid_h, grid_w = shape[0], shape[1], shape[2]
        depth = 5 + self.num_classes
        return tf.reshape(y_pred, (batch, grid_h, grid_w, self.num_anchors, depth))

    def __call__(self, y_true, y_pred):
        """Compute the (scalar) total loss for one batch.

        Returns a dict with the individual terms as well as 'total', so
        callers can log each component separately.
        """
        y_pred = self._reshape_pred(y_pred)

        obj_mask = y_true[..., 4]          # (batch, gh, gw, num_anchors)
        noobj_mask = 1.0 - obj_mask

        tx_pred = tf.sigmoid(y_pred[..., 0])
        ty_pred = tf.sigmoid(y_pred[..., 1])
        tw_pred = y_pred[..., 2]
        th_pred = y_pred[..., 3]
        conf_pred = tf.sigmoid(y_pred[..., 4])

        tx_true = y_true[..., 0]
        ty_true = y_true[..., 1]
        tw_true = y_true[..., 2]
        th_true = y_true[..., 3]

        num_obj = tf.maximum(tf.reduce_sum(obj_mask), 1.0)
        num_noobj = tf.maximum(tf.reduce_sum(noobj_mask), 1.0)

        losses = {}

        # ---- Localization loss (only over cells/anchors with an object) ----
        if self.use_localization:
            xy_loss = tf.square(tx_pred - tx_true) + tf.square(ty_pred - ty_true)
            wh_loss = tf.square(tw_pred - tw_true) + tf.square(th_pred - th_true)
            loc_loss = obj_mask * (xy_loss + wh_loss)
            loc_loss = self.lambda_coord * tf.reduce_sum(loc_loss) / num_obj
        else:
            loc_loss = tf.constant(0.0, dtype=y_pred.dtype)
        losses["localization"] = loc_loss

        # ---- Confidence loss (object cells weighted differently from
        #      no-object cells, as in Terven Figure 5) ----
        if self.use_confidence:
            # Ground-truth confidence target: 1 for object cells (the
            # standard, simplest target; swap in an IoU(pred, gt) value
            # here if you want IoU-aware objectness targets instead).
            conf_target = obj_mask

            obj_conf_loss = obj_mask * tf.square(conf_pred - conf_target)
            noobj_conf_loss = noobj_mask * tf.square(conf_pred - 0.0)

            conf_loss = (
                self.lambda_coord * tf.reduce_sum(obj_conf_loss) / num_obj
                + self.lambda_noobj * tf.reduce_sum(noobj_conf_loss) / num_noobj
            )
        else:
            conf_loss = tf.constant(0.0, dtype=y_pred.dtype)
        losses["confidence"] = conf_loss

        # ---- Classification loss (only over cells/anchors with an object) ----
        if self.use_classification:
            class_pred = tf.sigmoid(y_pred[..., 5:])  # independent logistic
                                                       # classifiers, as in
                                                       # YOLOv3 (Terven Sec 6);
                                                       # fine for single-class too.
            class_true = y_true[..., 5:]
            class_loss = tf.reduce_sum(tf.square(class_pred - class_true), axis=-1)
            class_loss = tf.reduce_sum(obj_mask * class_loss) / num_obj
        else:
            class_loss = tf.constant(0.0, dtype=y_pred.dtype)
        losses["classification"] = class_loss

        losses["total"] = losses["localization"] + losses["confidence"] + losses["classification"]
        return losses

class YOLOv2Loss_v2(Loss):
    """Configurable YOLOv2-style detection loss.
 
    Parameters
    ----------
    num_anchors : int
        Number of anchors per grid cell.
    num_classes : int
        Number of explicit classes (0 to disable the classification term
        entirely, matching the paper's single implicit-class design).
    lambda_coord : float
        Weight for the localization loss and for the "object" confidence
        term (paper/Terven Figure 5 use 5.0 for localization).
    lambda_noobj : float
        Down-weighting factor for the confidence loss in cells without an
        object (paper/Terven Figure 5 use 0.5).
    loss_type : 'squared_error' | 'bce' | 'focal'
        Loss function applied to the CONFIDENCE and CLASSIFICATION terms
        only -- localization stays squared error regardless, since
        tx/ty/tw/th are continuous regression targets, not probabilities,
        and BCE/focal don't apply to them conceptually.
 
        'squared_error' (default): (sigmoid(logit) - target)^2, matching
            the original YOLOv1/v2 paper. Known weakness: gradient w.r.t.
            the raw logit vanishes as the prediction saturates (p->0 or
            p->1), so a confidently-WRONG prediction gets almost no
            corrective signal -- this is what produces persistent
            high-confidence false positives that don't improve with more
            training.
        'bce': standard binary cross-entropy computed directly from raw
            logits (tf.nn.sigmoid_cross_entropy_with_logits, numerically
            stable). Gradient w.r.t. the logit is exactly (p - y) --
            full strength even when the prediction is confidently wrong,
            directly fixing the squared-error saturation problem above.
        'focal': BCE further reweighted by a (1 - p_t)^gamma modulating
            factor (Lin et al., RetinaNet) that suppresses the
            contribution of easy, already-correct examples so gradient
            budget concentrates on hard/misclassified ones. Addresses a
            DIFFERENT problem than 'bce' does: severe class imbalance
            (typically hundreds of easy "no object" cells per handful of
            real objects) diluting the average gradient, rather than the
            saturation issue itself (which 'bce' already fixes on its
            own). Useful in addition to 'bce', not a replacement for it.
    focal_gamma, focal_alpha : float
        Only used when loss_type='focal'. `gamma` controls how strongly
        easy examples are down-weighted (paper default 2.0); `alpha`
        rebalances positive vs. negative examples overall (paper default
        0.25 for the positive class).
    use_localization, use_confidence, use_classification : bool
        Independently enable/disable each of the three loss terms. This is
        the "pick which parts of the loss you want" mechanism requested --
        each flag simply zeroes out that term's contribution to the total
        when set to False.
    """
 
    def __init__(
        self,
        num_anchors: int,
        num_classes: int = 0,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        loss_type: str = "squared_error",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        use_localization: bool = True,
        use_confidence: bool = True,
        use_classification: bool = True,
    ):
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        if loss_type not in ("squared_error", "bce", "focal"):
            raise ValueError(f"loss_type must be 'squared_error', 'bce', or 'focal', got {loss_type!r}")
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.use_localization = use_localization
        self.use_confidence = use_confidence
        self.use_classification = use_classification and num_classes > 0
 
    def _binary_loss(self, logits, target, is_positive_class):
        """Dispatch to squared_error/bce/focal, all elementwise, all
        operating on RAW logits (numerically stable -- no separately
        computed sigmoid() is unwound/reused here). Used for BOTH the
        confidence and classification terms.
 
        `is_positive_class` selects which per-example alpha weight to use
        for focal loss (alpha for target=1 cells, 1-alpha for target=0
        cells) -- irrelevant for the other two loss types.
        """
        if self.loss_type == "squared_error":
            p = tf.sigmoid(logits)
            return tf.square(p - target)
 
        # BCE from logits: numerically stable, avoids ever forming log(0)
        # or log(1) explicitly even as logits grow large in magnitude.
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)
 
        if self.loss_type == "bce":
            return bce
 
        # focal: reweight BCE by (1 - p_t)^gamma, where p_t is the model's
        # probability assigned to the TRUE class (large p_t = easy example).
        p = tf.sigmoid(logits)
        p_t = target * p + (1.0 - target) * (1.0 - p)
        modulating_factor = tf.pow(1.0 - p_t, self.focal_gamma)
        alpha_factor = target * self.focal_alpha + (1.0 - target) * (1.0 - self.focal_alpha)
        return alpha_factor * modulating_factor * bce
 
    def _reshape_pred(self, y_pred):
        shape = tf.shape(y_pred)
        batch, grid_h, grid_w = shape[0], shape[1], shape[2]
        depth = 5 + self.num_classes
        return tf.reshape(y_pred, (batch, grid_h, grid_w, self.num_anchors, depth))
 
    def __call__(self, y_true, y_pred):
        """Compute the (scalar) total loss for one batch.
 
        Returns a dict with the individual terms as well as 'total', so
        callers can log each component separately.
        """
        y_pred = self._reshape_pred(y_pred)
 
        obj_mask = y_true[..., 4]          # (batch, gh, gw, num_anchors)
        noobj_mask = 1.0 - obj_mask
 
        tx_pred = tf.sigmoid(y_pred[..., 0])
        ty_pred = tf.sigmoid(y_pred[..., 1])
        tw_pred = y_pred[..., 2]
        th_pred = y_pred[..., 3]
 
        tx_true = y_true[..., 0]
        ty_true = y_true[..., 1]
        tw_true = y_true[..., 2]
        th_true = y_true[..., 3]
 
        num_obj = tf.maximum(tf.reduce_sum(obj_mask), 1.0)
        num_noobj = tf.maximum(tf.reduce_sum(noobj_mask), 1.0)
 
        losses = {}
 
        # ---- Localization loss (only over cells/anchors with an object) ----
        if self.use_localization:
            xy_loss = tf.square(tx_pred - tx_true) + tf.square(ty_pred - ty_true)
            wh_loss = tf.square(tw_pred - tw_true) + tf.square(th_pred - th_true)
            loc_loss = obj_mask * (xy_loss + wh_loss)
            loc_loss = self.lambda_coord * tf.reduce_sum(loc_loss) / num_obj
        else:
            loc_loss = tf.constant(0.0, dtype=y_pred.dtype)
        losses["localization"] = loc_loss
 
        # ---- Confidence loss (object cells weighted differently from
        #      no-object cells, as in Terven Figure 5) ----
        if self.use_confidence:
            conf_logits = y_pred[..., 4]
            # Ground-truth confidence target: 1 for object cells (the
            # standard, simplest target; swap in an IoU(pred, gt) value
            # here if you want IoU-aware objectness targets instead).
            conf_target = obj_mask
 
            obj_conf_loss = obj_mask * self._binary_loss(conf_logits, conf_target, is_positive_class=True)
            noobj_conf_loss = noobj_mask * self._binary_loss(conf_logits, tf.zeros_like(conf_logits), is_positive_class=False)
 
            conf_loss = (
                self.lambda_coord * tf.reduce_sum(obj_conf_loss) / num_obj
                + self.lambda_noobj * tf.reduce_sum(noobj_conf_loss) / num_noobj
            )
        else:
            conf_loss = tf.constant(0.0, dtype=y_pred.dtype)
        losses["confidence"] = conf_loss
 
        # ---- Classification loss (only over cells/anchors with an object) ----
        if self.use_classification:
            class_logits = y_pred[..., 5:]  # independent logistic
                                             # classifiers, as in
                                             # YOLOv3 (Terven Sec 6);
                                             # fine for single-class too.
            class_true = y_true[..., 5:]
            class_loss = tf.reduce_sum(self._binary_loss(class_logits, class_true, is_positive_class=True), axis=-1)
            class_loss = tf.reduce_sum(obj_mask * class_loss) / num_obj
        else:
            class_loss = tf.constant(0.0, dtype=y_pred.dtype)
        losses["classification"] = class_loss
 
        losses["total"] = losses["localization"] + losses["confidence"] + losses["classification"]
        return losses

if __name__ == "__main__":
    # Self-test with synthetic tensors, exercising every on/off combination.
    batch, grid_h, grid_w, num_anchors, num_classes = 2, 6, 8, 5, 1
    depth = 5 + num_classes

    y_pred = tf.random.normal((batch, grid_h, grid_w, num_anchors * depth))

    y_true = tf.zeros((batch, grid_h, grid_w, num_anchors, depth))
    # Manually mark one object at (row=2, col=3, anchor=0) in batch 0.
    y_true = y_true.numpy()
    y_true[0, 2, 3, 0, 0] = 0.5   # tx target
    y_true[0, 2, 3, 0, 1] = 0.5   # ty target
    y_true[0, 2, 3, 0, 2] = 0.1   # tw target
    y_true[0, 2, 3, 0, 3] = -0.2  # th target
    y_true[0, 2, 3, 0, 4] = 1.0   # objectness
    y_true[0, 2, 3, 0, 5] = 1.0   # class 0 (person)
    y_true = tf.constant(y_true, dtype=tf.float32)

    print("All terms enabled:")
    loss_fn = YOLOv2Loss(num_anchors=num_anchors, num_classes=num_classes)
    print(loss_fn(y_true, y_pred))

    print("\nLocalization + confidence only (no classification):")
    loss_fn2 = YOLOv2Loss(num_anchors=num_anchors, num_classes=num_classes,
                           use_classification=False)
    print(loss_fn2(y_true, y_pred))

    print("\nConfidence only:")
    loss_fn3 = YOLOv2Loss(num_anchors=num_anchors, num_classes=num_classes,
                           use_localization=False, use_classification=False)
    print(loss_fn3(y_true, y_pred))
