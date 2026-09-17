"""
train.py
 
Orchestrates training of YOLOv2Tiny (model.py) against the configurable
YOLOv2Loss (loss.py), using cached per-sequence tf.data.Datasets written
by video_dataset.save_sequence_cache().
 
Pipeline:
  1. Discover every cached sequence under `data_dir` (each must have been
     built with build_video_dataset(..., sequence_id=N) so elements carry
     (image, y_true, sequence_id)).
  2. Split sequences into train/val/test -- by sequence, not by frame (see
     split_by_sequence_id in video_dataset.py), either via explicit id
     lists or an automatic random split by fraction.
  3. Train with a manual tf.GradientTape loop (not model.fit()) so the
     paper-style warmup + reduce-on-plateau LR schedule and the
     configurable loss terms are easy to express.
  4. Track validation loss each epoch; checkpoint whenever it improves;
     reduce LR on a validation plateau; stop early after sustained
     non-improvement.
  5. Restore the best checkpoint and report final test-set loss.
 
Checkpointing / resuming:
  In addition to the "best model so far" checkpoint, the Trainer persists
  its FULL state (model weights, optimizer state incl. momentum slot
  variables and the current learning rate, epoch number, plateau/early-
  stopping counters, base_lr, best_val_loss, and the train/val/test
  sequence-id split) to `checkpoint_dir` after every epoch. If training is
  interrupted (crash, forced shutdown, ...), simply constructing a new
  Trainer against the same `checkpoint_dir` (or calling `Trainer.restore`)
  will automatically pick the optimization procedure back up exactly where
  it left off -- see `Trainer.save_checkpoint`, `Trainer.restore`, and the
  `auto_resume` constructor argument.
 
Usage as a class:
    trainer = Trainer(data_dir="cache/", input_shape=(24, 32, 1))
    trainer.fit(epochs=200)
    test_losses = trainer.test()
 
    # Resuming after an interruption (same checkpoint_dir):
    trainer = Trainer(data_dir="cache/", model=..., loss=..., checkpoint_dir="checkpoints")
    trainer.fit(epochs=200)  # continues from the last completed epoch
 
Usage as a script: see the __main__ block / `python train.py --help`.
"""
 
import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple
 
 
import numpy as np
import tensorflow as tf
 
import tqdm
 
from .models import YOLOv2Tiny
from .loss import Loss
from .augment import expand_with_flips
 
 
def find_sequence_tfdata(data_dir: str, compression: Optional[str] = None) -> Dict[int, str]:
    """Find every cached sequence directory under `data_dir` and map it to
    the sequence_id embedded in its data (peeked from the first element,
    NOT inferred from the directory name -- more robust to arbitrary
    naming conventions).
 
    Every subdirectory of `data_dir` is assumed to be one
    td.data.Dataset.save() output, built from a dataset that carries
    (image, y_true, sequence_id) triples (i.e. build_video_dataset() was
    called with an explicit sequence_id).
 
    Returns
    -------
    dict mapping sequence_id -> cache directory path.
    """
    subdirs = sorted(d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d))
    if not subdirs:
        raise FileNotFoundError(f"No sequence cache subdirectories found under {data_dir}")
 
    seq_paths = {}
    for subdir in subdirs:
        ds = tf.data.Dataset.load(subdir, compression=compression)
        try:
            first_element = next(iter(ds))
        except StopIteration:
            raise ValueError(f"Cached sequence at {subdir} is empty.")
        if len(first_element) != 3:
            raise ValueError(
                f"Cached sequence at {subdir} has element_spec without a sequence_id "
                f"(got {len(first_element)}-tuple elements). Rebuild it with "
                f"build_video_dataset(..., sequence_id=N) before using Trainer."
            )
        seq_id = int(first_element[2].numpy())
        seq_paths[seq_id] = subdir
    return seq_paths
 
 
def split_sequence_tfdata(
    sequence_ids: Sequence[int],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """Randomly partition sequence ids into train/val/test by fraction.
 
    Splitting happens at the SEQUENCE level (not per-frame), per the
    earlier recommendation to keep validation/test sequences from
    different physical recording locations than training sequences.
    """
    ids = list(sequence_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
 
    n = len(ids)
    n_val = max(1, round(n * val_frac)) if n > 1 else 0
    n_test = max(1, round(n * test_frac)) if n > 2 else 0
    n_val = min(n_val, n - 1) if n > 1 else 0
    n_test = min(n_test, n - n_val - 1) if n - n_val > 1 else 0
 
    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test + n_val]
    train_ids = ids[n_test + n_val:]
 
    if len(train_ids) == 0:
        raise ValueError(
            f"auto_split_sequence_ids produced zero training sequences from "
            f"{n} total -- too few sequences for the requested val_frac/test_frac. "
            f"Pass explicit train_ids/val_ids/test_ids instead."
        )
    return train_ids, val_ids, test_ids
 
def split_by_sequence_id(
    dataset: tf.data.Dataset,
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    test_ids: Sequence[int],
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Split a combined (image, y_true, sequence_id) dataset into
    train/val/test by sequence_id, per the "split by sequence, not by
    frame" recommendation discussed earlier.
 
    Requires the dataset's elements to include a sequence_id (i.e. it must
    have been built/combined from datasets created with
    build_video_dataset(..., sequence_id=N)).
    """
 
    # Create a function that maps every dataset element to True or False
    # _make_filter takes the sequence id's of elements that should remain in
    # in the dataset as input
    def _filter_by_sequence_id(ids):
        """
        Function that maps every dataset element to True or False
 
        Parameters
        ----------
        ids : list
            list of sequence_ids that should remain in the dataset.
 
        """
 
        # Convert the list of sequence ids to keep in dataset to tf.constant
        ids_tensor = tf.constant(list(ids), dtype=tf.int64)
 
        # Define actual filter function
        def _filter_fn(image, y_true, seq_id):
 
            # tf.equal(seq_id, ids_tensor) broadcasts the scalar seq_id
            # against every entry of ids_tensor and produces a boolean array
            # the same length as ids_tensor
 
            # tf.reduce_any(...) — collapses that boolean array down to a
            # single True/False: True if any entry matched.
            return tf.reduce_any(tf.equal(seq_id, ids_tensor,
                                          name = 'matching_sequence_id'))
 
        return _filter_fn
 
    # dataset.filter(fn) keeps only the elements where fn(...) returns True
    train_ds = dataset.filter(_filter_by_sequence_id(train_ids), name = 'train_ds')
    val_ds = dataset.filter(_filter_by_sequence_id(val_ids), name = 'val_ds')
    test_ds = dataset.filter(_filter_by_sequence_id(test_ids), name = 'test_ds')
 
    return train_ds, val_ds, test_ds
 
 
class Trainer:
    """Loads cached sequence datasets, splits by sequence, and trains
    YOLOv2Tiny to minimize validation loss, then reports test loss.
 
    Parameters
    ----------
    data_dir : str
        Directory containing one subdirectory per cached sequence (see
        discover_sequence_caches()).
    input_shape, num_anchors, num_classes : optional
        Passed to YOLOv2Tiny. If left as None, all three are inferred
        from the first training example's shape (image shape -> input_shape;
        y_true's last two dims -> num_anchors, num_classes).
    compression : 'GZIP' or None
        Must match what save_sequence_cache() used for every sequence.
    train_ids, val_ids, test_ids : optional
        Explicit sequence-id assignment. If any is None, all three are
        derived automatically via auto_split_sequence_ids() -- UNLESS an
        existing checkpoint is found under `checkpoint_dir` and
        `auto_resume` is True, in which case the split saved by that
        checkpoint is reused (see `auto_resume` below).
    loss_kwargs : dict
        Extra kwargs forwarded to YOLOv2Loss (e.g. use_classification=False,
        lambda_coord=5.0, ...).
    base_lr : float
        Learning rate reached at the end of warmup and used as the
        starting point for the reduce-on-plateau schedule.
    warmup_steps : int
        Number of training steps (mini-batches) over which LR ramps
        linearly from `base_lr/100` up to `base_lr`. The paper describes
        an "exponential" warmup without giving an exact formula; a linear
        ramp is used here as a documented, simple stand-in -- swap
        `_warmup_lr()` for a different curve if you need to match a
        specific schedule.
    lr_decay_factor : float
        Multiplicative LR reduction applied on a validation plateau.
    lr_patience : int
        Epochs without validation improvement before reducing LR.
    early_stopping_patience : int
        Epochs without validation improvement before stopping training.
    checkpoint_dir : str
        Where the best-so-far model weights are saved, AND where the
        full resumable trainer/optimizer state is saved after every
        epoch (see `save_checkpoint` / `restore`).
    auto_resume : bool
        If True (default) and `checkpoint_dir` already contains state
        written by a previous run (i.e. `save_checkpoint()` ran at least
        once, whether via `fit()`'s per-epoch checkpointing or manually),
        the new Trainer instance automatically restores the model
        weights, optimizer state (including the current learning rate),
        epoch number, and plateau/early-stopping counters from it, and
        `fit()` will continue from the epoch right after the last one
        that was checkpointed. Pass `auto_resume=False` to always start
        fresh (this will overwrite the existing checkpoint as training
        proceeds).
    """
 
    # Filename (inside checkpoint_dir) that stores everything needed to
    # resume training that ISN'T already captured by the TensorFlow
    # checkpoint (model weights + optimizer variables) -- i.e. plain
    # Python/JSON-serializable trainer state.
    STATE_FILENAME = "trainer_state.json"
 
    def __init__(
        self,
        data_dir : str,
        compression : str,
        model : tf.keras.Model,
        loss : Loss,
        batch_size: int = None,
        optimizer : str = 'SGD',
        shuffle_buffer: int = 1024,
        train_ids: Optional[Sequence[int]] = None,
        val_ids: Optional[Sequence[int]] = None,
        test_ids: Optional[Sequence[int]] = None,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        split_seed: int = 0,
        loss_kwargs: Optional[dict] = None,
        base_lr: float = 1e-3,
        momentum: float = 0.9,
        warmup_epochs: int = 10,
        lr_decay_factor: float = 0.1,
        lr_patience: int = 5,
        early_stopping_patience: int = 15,
        # min_delta: float = 1e-4,
        checkpoint_dir: str = "checkpoints",
        auto_resume: bool = True,
        **kwargs
    ):
        self.data_dir = data_dir
        self.compression = compression
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
 
        self.base_lr = base_lr
        self.momentum = momentum
        self.warmup_epochs = max(1, warmup_epochs)
        self.lr_decay_factor = lr_decay_factor
        self.lr_patience = lr_patience
        self.early_stopping_patience = early_stopping_patience
        # self.min_delta = min_delta
        self.checkpoint_dir = checkpoint_dir
 
        self.hflip = kwargs.pop('hflip',False)
        self.vflip = kwargs.pop('vflip',False)
 
        # ---- Optimization-procedure state (also persisted/restored by
        # save_checkpoint()/restore() -- see those methods) ----
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_losses: Optional[Dict[str, float]] = None
        self.epochs_since_improvement = 0
        self.epochs_since_lr_reduction = 0
 
        # ---- If a previous run already checkpointed state here, pick up
        # its train/val/test split so resuming lines up with the data the
        # in-progress optimization procedure was actually trained on. ----
        os.makedirs(checkpoint_dir, exist_ok=True)
        state_path = os.path.join(checkpoint_dir, self.STATE_FILENAME)
        resumed_state = None
        if auto_resume and os.path.exists(state_path):
            with open(state_path, "r") as f:
                resumed_state = json.load(f)
            if train_ids is None:
                train_ids = resumed_state["train_ids"]
            if val_ids is None:
                val_ids = resumed_state["val_ids"]
            if test_ids is None:
                test_ids = resumed_state["test_ids"]
            print(f"[Trainer] found existing checkpoint state at {state_path} "
                  f"(last completed epoch={resumed_state['epoch'] - 1}) -- reusing its "
                  f"train/val/test split so training can resume consistently. "
                  f"Pass auto_resume=False to ignore it and start fresh.")
 
        # --------- Discover all tf sequences contained in data_dir  ----------
        seq_paths = find_sequence_tfdata(data_dir, compression=compression)
        all_ids = list(seq_paths.keys())
 
        # ---------- If no data split is provided: split randomly -------------
        if train_ids is None or val_ids is None or test_ids is None:
            train_ids, val_ids, test_ids = split_sequence_tfdata(
                all_ids, val_frac=val_frac, test_frac=test_frac, seed=split_seed
            )
 
        # Assign the sequence ids of the split data to the respective class
        # attributes
        self.train_ids, self.val_ids, self.test_ids = \
            list(train_ids), list(val_ids), list(test_ids)
        print(f"[Trainer] {len(all_ids)} sequences found -> "
              f"train={self.train_ids} val={self.val_ids} test={self.test_ids}")
 
        # Combine all sequences to one large dataset
        combined = self._combine(seq_paths, all_ids)
 
        # Split the elements in the combined dataset by sequence_id into
        # training, validation and test data
        train_raw, val_raw, test_raw = split_by_sequence_id(
            combined, self.train_ids, self.val_ids, self.test_ids
        )
 
        # Drop sequence_id for actual training/eval -- model/loss only need (image, y_true).
        self.train_ds = self._make_pipeline(train_raw, shuffle=True,
                                            hflip = self.hflip,
                                            vflip = self.vflip)
 
        self.val_ds = self._make_pipeline(val_raw, shuffle=False,
                                          hflip = self.hflip,
                                          vflip = self.vflip)
 
        self.test_ds = self._make_pipeline(test_raw, shuffle=False,
                                           hflip = self.hflip,
                                           vflip = self.vflip)
 
        # ----- Output some information regarding the provided data --------
        len_train_ds = sum(1 for _ in self.train_ds) * self.batch_size
        len_val_ds = sum(1 for _ in self.val_ds) * self.batch_size
        len_test_ds = sum(1 for _ in self.test_ds) * self.batch_size
 
        print(f"[Trainer] Training data - Number of elements: {len_train_ds}\n"
              f"[Trainer] Validation data - Number of elements: {len_val_ds}\n"
              f"[Trainer] Test data - Number of elements: {len_test_ds}\n")
 
    
        print(sum(float(tf.reduce_sum(y[..., 4])) for _, y in self.train_ds))
        
        # ---- Model, loss, optimizer ----
        self.model = model # YOLOv2Tiny(input_shape=input_shape, num_anchors=num_anchors, num_classes=num_classes)
        self.loss_fn = loss # YOLOv2Loss(num_anchors=num_anchors, num_classes=num_classes, **(loss_kwargs or {}))
 
        # Force the model's weights to exist before the optimizer is built
        # and before any checkpoint restore is attempted: a freshly
        # constructed Keras model whose weights are only created lazily
        # (on first call) has nothing for tf.train.Checkpoint to
        # save/restore yet.
        self._ensure_model_built()
        
        # Initialize optimizer
        if optimizer == 'SGD':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr / 100.0,
                                                      momentum=momentum)
        elif optimizer == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(base_lr)
            
 
        # Eagerly create the optimizer's slot variables (momentum
        # accumulators, iteration counter, ...) instead of letting them be
        # created lazily on the first apply_gradients() call. Without this,
        # a checkpoint saved/restored before the first training step would
        # silently miss the optimizer state entirely.
        if hasattr(self.optimizer, "build"):
            self.optimizer.build(self.model.trainable_variables)
 
        # "Best" checkpoint: model weights only, saved whenever validation
        # loss improves, used to restore the best model for test()/deployment.
        self.best_checkpoint = tf.train.Checkpoint(model=self.model)
        self.best_ckpt_manager = tf.train.CheckpointManager(
            self.best_checkpoint, directory=os.path.join(checkpoint_dir, "best"), max_to_keep=1
        )
 
        # "Latest"/resume checkpoint: model weights AND optimizer state
        # (learning rate, momentum accumulators, step count, ...), saved
        # after EVERY epoch so training can resume exactly where it
        # stopped after a crash or forced shutdown.
        self.full_checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.latest_ckpt_manager = tf.train.CheckpointManager(
            self.full_checkpoint, directory=os.path.join(checkpoint_dir, "latest"), max_to_keep=2
        )
 
        if resumed_state is not None:
            self._load_state(resumed_state)
 
    # ---- setup helpers ----------------------------------------------
 
    def _combine(self, seq_paths: Dict[int, str], ids: Sequence[int]) -> tf.data.Dataset:
        datasets = [tf.data.Dataset.load(seq_paths[i], compression=self.compression) for i in ids]
        combined = datasets[0]
        for ds in datasets[1:]:
            combined = combined.concatenate(ds)
        return combined
 
    def _ensure_model_built(self) -> None:
        """Run a single training batch through the model if its weights
        don't exist yet, so they can be checkpointed/restored regardless
        of whether any real training step has happened."""
        if getattr(self.model, "built", True) and self.model.trainable_variables:
            return
        for images, _ in self.train_ds.take(1):
            self.model(images, training=False)
            break
 
    def _make_pipeline(self,
                       dataset: tf.data.Dataset,
                       shuffle: bool,
                       **kwargs) -> tf.data.Dataset:
 
        hflip = kwargs.pop('hflip',False)   # Flag for augmenting dataset with horizontally flipped version
        vflip = kwargs.pop('vflip',False)   # Flag for augmenting dataset with vertically flipped version
 
        rotate = hflip and vflip            # Flag for augmenting dataset with 180° rotated version
 
        # Drop the sequence_id from the data element triplet (image, y_true,sequence_id)
        ds = dataset.map(lambda image, y_true, seq_id: (image, y_true),
                         num_parallel_calls=tf.data.AUTOTUNE)
 
        # Expand dataset with flipped and rotated versions as specified
        ds = expand_with_flips(ds,
                               hflip,
                               vflip,
                               rotate)
 
        # Shuffle the data, so the temporarilly highly correlated sequence
        # data is uncorrelated when split into batches
        if shuffle:
            ds = ds.shuffle(self.shuffle_buffer)
 
        # Divide the dataset into batches
        if self.batch_size is None:
            # If no batch_size is provided, put all elements in one batch
            # Get number of elements by iterating through full dataset
            len_ds = sum(1 for _ in ds)
            ds = ds.batch(len_ds)
        else:
            # Else use specified batch_size
            ds = ds.batch(self.batch_size)
 
        # prefetch lets tensorflow prepare the next batch
        # (i.e. do drop sequence id -> shuffle -> batch overhead) on CPU
        # while the GPU is still busy processing the current batch
        ds = ds.prefetch(tf.data.AUTOTUNE)
 
        # Return dataset
        return ds
 
    # ---- LR schedule ---------------------------------------------------
 
    def _update_lr(self, epoch: int) -> float:
        """Exponential warmup ramp from start_lr to base_lr over warmup_epochs.
        See class docstring note on the paper's unspecified exponential
        warmup formula.
        """
 
        if epoch > self.warmup_epochs:
            # if warm-up stage is over, return base_lr
            return self.base_lr
 
        else:
            # # else increase learning rate exponentially from start_lr to
            # # base_lr
            # # Exponential function: lr = a*e^(k*epoch)
            # start_lr = self.base_lr / 1E9
 
            # a = start_lr
            # k = np.log(self.base_lr / start_lr) / self.warmup_epochs
 
            # new_lr = a * np.exp(k*epoch)
 
            # linear warm up schedule
            start_lr = self.base_lr / 1E3
            fraction = (self.base_lr - start_lr)/self.warmup_epochs
            new_lr = start_lr + fraction * epoch
 
            return new_lr
 
    # ---- train / eval steps --------------------------------------------
 
    @tf.function
    def _train_step(self, images, y_true):
 
        # Open a "tape" that records every differentiable operation performed
        with tf.GradientTape() as tape:
            # Evaluate the model on the input data (images)
            y_pred = self.model(images, training=True)
 
            # Pass the model prediction to the loss function to determine the
            # differentiable loss
            losses = self.loss_fn(y_true, y_pred)
 
        # Differentiate the loss w.r.t. the model parameters
        grads = tape.gradient(losses["total"], self.model.trainable_variables)
 
        # Pass the gradient to the optimizer, which performs the parameter
        # update
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return losses
 
    @tf.function
    def _eval_step(self, images, y_true):
        y_pred = self.model(images, training=False)
        return self.loss_fn(y_true, y_pred)
 
    def _run_epoch(self, dataset: tf.data.Dataset, training: bool) -> Dict[str, float]:
        totals = {"localization": 0.0, "confidence": 0.0, "classification": 0.0, "total": 0.0}
        batch_counter = 0
 
        # Iterate over the dataset. For dataset construction see _make_pipeline
        for images, y_true in dataset:
            if training:
                # Retrieve the current learning rate
 
                losses = self._train_step(images, y_true)
            else:
                losses = self._eval_step(images, y_true)
            for k in totals:
                totals[k] += float(losses[k])
            batch_counter += 1
 
 
        if batch_counter == 0:
            raise ValueError("Encountered an empty dataset split during an epoch -- check your train/val/test split.")
        return {k: v / batch_counter for k, v in totals.items()}
 
    # ---- checkpointing / restoring the trainer & optimization state ----
 
    def _state_dict(self) -> dict:
        """Everything needed to resume the optimization procedure that
        ISN'T already captured by the TensorFlow checkpoint (model
        weights + optimizer variables, saved separately by
        `save_checkpoint` via `self.latest_ckpt_manager`)."""
        return {
            # optimization-procedure progress
            "epoch": self.epoch,
            "base_lr": self.base_lr,
            "current_lr": float(tf.keras.backend.get_value(self.optimizer.learning_rate)),
            "best_val_loss": self.best_val_loss,
            "best_val_losses": self.best_val_losses,
            "epochs_since_improvement": self.epochs_since_improvement,
            "epochs_since_lr_reduction": self.epochs_since_lr_reduction,
            # enough of the constructor configuration to reconstruct an
            # identical Trainer (data split, pipeline, and LR-schedule
            # hyperparameters) around the restored model/optimizer.
            "data_dir": self.data_dir,
            "compression": self.compression,
            "batch_size": self.batch_size,
            "shuffle_buffer": self.shuffle_buffer,
            "train_ids": self.train_ids,
            "val_ids": self.val_ids,
            "test_ids": self.test_ids,
            "hflip": self.hflip,
            "vflip": self.vflip,
            "momentum": self.momentum,
            "warmup_epochs": self.warmup_epochs,
            "lr_decay_factor": self.lr_decay_factor,
            "lr_patience": self.lr_patience,
            "early_stopping_patience": self.early_stopping_patience,
            "checkpoint_dir": self.checkpoint_dir,
        }
 
    def save_checkpoint(self) -> str:
        """Persist the FULL trainer/optimization state -- model weights,
        optimizer state (learning rate, momentum accumulators, step
        count, ...), current epoch, and plateau/early-stopping counters
        -- to `self.checkpoint_dir`, so training can resume from exactly
        this point later (see `restore` / the `auto_resume` constructor
        argument).
 
        Called automatically at the end of every epoch inside `fit()`,
        but safe to call manually at any other time too.
        """
        save_path = self.latest_ckpt_manager.save()
        state_path = os.path.join(self.checkpoint_dir, self.STATE_FILENAME)
        with open(state_path, "w") as f:
            json.dump(self._state_dict(), f, indent=2)
        return save_path
 
    # Backwards/forwards-compatible alias.
    save = save_checkpoint
 
    def _load_state(self, state: dict) -> None:
        """Restore model weights, optimizer state, and trainer/
        optimization-procedure progress from a state dict produced by
        `_state_dict()`, plus the matching TensorFlow checkpoint under
        `self.checkpoint_dir/latest`."""
        restore_path = self.latest_ckpt_manager.latest_checkpoint
        if restore_path is None:
            raise FileNotFoundError(
                f"{self.STATE_FILENAME} was found but no TensorFlow checkpoint exists under "
                f"{os.path.join(self.checkpoint_dir, 'latest')} -- cannot restore model/optimizer weights."
            )
        # expect_partial(): the model/optimizer objects were freshly
        # constructed by this __init__ call (not loaded from this
        # checkpoint), so asserting full consumption here is unnecessary
        # and would also trip on harmless bookkeeping variables that
        # differ across TF versions. The values that DO match are still
        # restored correctly.
        status = self.full_checkpoint.restore(restore_path)
        status.expect_partial()
 
        self.epoch = state["epoch"]
        self.base_lr = state["base_lr"]
        self.best_val_loss = state["best_val_loss"]
        self.best_val_losses = state["best_val_losses"]
        self.epochs_since_improvement = state["epochs_since_improvement"]
        self.epochs_since_lr_reduction = state["epochs_since_lr_reduction"]
 
        print(f"[Trainer] restored checkpoint from {restore_path} -- resuming at epoch {self.epoch} "
              f"(base_lr={self.base_lr:.2e}, best_val_loss={self.best_val_loss:.4f})")
 
    @classmethod
    def restore(
        cls,
        checkpoint_dir: str,
        model: tf.keras.Model,
        loss: Loss,
        **overrides,
    ) -> "Trainer":
        """Reconstruct a Trainer from `checkpoint_dir`, picking the
        optimization procedure -- epoch, learning-rate schedule state,
        optimizer state, model weights, plateau/early-stopping counters,
        and the original train/val/test sequence split -- back up exactly
        where it left off.
 
        `model` and `loss` must be freshly constructed instances with the
        same architecture/config as whatever produced the checkpoint
        (arbitrary Python objects aren't serialized by save_checkpoint());
        everything else needed is read back from `checkpoint_dir`.
 
        Any other constructor keyword argument (e.g. `batch_size` or
        `checkpoint_dir` itself, to copy a checkpoint to a new location)
        can be overridden via `overrides`.
 
        Equivalent to constructing `Trainer(..., checkpoint_dir=checkpoint_dir)`
        with `auto_resume=True` (the default), except `data_dir`/
        `compression`/the rest of the split & schedule configuration are
        recovered from the checkpoint instead of having to be supplied
        again by hand.
        """
        state_path = os.path.join(checkpoint_dir, cls.STATE_FILENAME)
        if not os.path.exists(state_path):
            raise FileNotFoundError(
                f"No trainer state found at {state_path} -- nothing to restore. "
                f"(Did training ever run and call save_checkpoint()/fit()?)"
            )
        with open(state_path, "r") as f:
            state = json.load(f)
 
        init_kwargs = dict(
            data_dir=state["data_dir"],
            compression=state["compression"],
            model=model,
            loss=loss,
            batch_size=state["batch_size"],
            optimizer=state["optimizer"],
            shuffle_buffer=state["shuffle_buffer"],
            train_ids=state["train_ids"],
            val_ids=state["val_ids"],
            test_ids=state["test_ids"],
            base_lr=state["base_lr"],
            momentum=state["momentum"],
            warmup_epochs=state["warmup_epochs"],
            lr_decay_factor=state["lr_decay_factor"],
            lr_patience=state["lr_patience"],
            early_stopping_patience=state["early_stopping_patience"],
            checkpoint_dir=checkpoint_dir,
            hflip=state["hflip"],
            vflip=state["vflip"],
            auto_resume=True,
        )
        init_kwargs.update(overrides)
        # __init__ itself performs the restore (it detects the existing
        # checkpoint state at `checkpoint_dir` and calls _load_state()).
        return cls(**init_kwargs)
 
    def _print_loss(self,losses:dict, mode : str):
        
        epoch = self.epoch
        
        loc_loss = losses['localization']
        conf_loss = losses['confidence']
        class_loss = losses['classification']
        total_loss = losses['total']
        
        print(f"[{mode}][epoch {epoch:4d}] lr={float(self.optimizer.learning_rate.numpy()):.2e}  "
              f"loc_loss={loc_loss:.4f} " + \
              f"conf_loss={conf_loss:.4f} " + \
              f"class_loss={class_loss:.4f} " + \
              f"total_loss={total_loss:.4f} ")
        
 
    # ---- public API ------------------------------------------------
 
    def fit(self, epochs: int = 200) -> Dict[str, float]:
        """Train to minimize validation loss, with warmup + reduce-on-
        plateau LR scheduling, checkpointing the best model, and early
        stopping. Returns the best validation loss dict achieved.
 
        If this Trainer was constructed with `auto_resume=True` (the
        default) against a `checkpoint_dir` that already contains state
        from a previous run -- e.g. after a forced shutdown -- training
        resumes from the epoch right after the last one that was
        checkpointed, with the learning-rate schedule, optimizer state,
        and plateau/early-stopping counters all picked back up as they
        were.
        """
 
        start_epoch = self.epoch
        if start_epoch > 0:
            print(f"[Trainer] resuming training at epoch {start_epoch} "
                  f"(best_val_loss so far={self.best_val_loss:.4f})")
        else:
            print('Starting training...')
 
        for epoch in range(start_epoch, epochs + 1):
 
            # Apply warm-up strategy if still in warm-up stage
            # self.optimizer.learning_rate = self._update_lr(epoch)
 
            # Evaluate the model on the training and validation dataset
            train_losses = self._run_epoch(self.train_ds, training=True)
            val_losses = self._run_epoch(self.val_ds, training=False)
            
            self._print_loss(train_losses,'train')
            self._print_loss(val_losses,'val')
            
            # # Reduce-on-plateau learning-rate scheduling (skipped while
            # # still in the warm-up phase).
            # if self.best_val_losses is None or val_losses["total"] < self.best_val_loss: # - self.min_delta:
            #     # If a new validation loss minimum has been found, reset
            #     # counters and store new best model
            #     self.best_val_loss = val_losses["total"]
            #     self.best_val_losses = val_losses
            #     self.epochs_since_improvement = 0
            #     self.epochs_since_lr_reduction = 0
            #     self.best_ckpt_manager.save()
            #     print(f"           -> new best val_total={self.best_val_loss:.4f}, checkpoint saved")
            # else:
            #     # If no validation loss improvement has been achieved, increase
            #     # counters
            #     self.epochs_since_improvement += 1
            #     self.epochs_since_lr_reduction += 1
 
            # if (self.epochs_since_lr_reduction >= self.lr_patience) and \
            #     (epoch > self.warmup_epochs):
            #     # If a plateau has been reached, reduce the learning rate
            #     new_base_lr = self.base_lr * self.lr_decay_factor
            #     print(f"           -> validation plateau: reducing base_lr {self.base_lr:.2e} -> {new_base_lr:.2e}")
            #     self.base_lr = new_base_lr
            #     self.epochs_since_lr_reduction = 0
 
            # Persist the full trainer + optimization-procedure state
            # (model, optimizer incl. learning rate, epoch, counters, ...)
            # after EVERY epoch, so training can resume exactly here if
            # interrupted by a crash or forced shutdown.
            self.epoch = epoch + 1
            self.save_checkpoint()
 
            if epoch > self.warmup_epochs and self.epochs_since_improvement >= self.early_stopping_patience:
                print(f"[Trainer] no val improvement for {self.early_stopping_patience} epochs -> early stopping")
                break
 
        # Restore best weights before returning / before test() is called.
        if self.best_ckpt_manager.latest_checkpoint is not None:
            self.best_checkpoint.restore(self.best_ckpt_manager.latest_checkpoint)
            print(f"[Trainer] training done, restored best checkpoint (val_total={self.best_val_loss:.4f})")
        return self.best_val_losses
 
    def test(self) -> Dict[str, float]:
        """Evaluate the (best-checkpoint) model on the held-out test split."""
        test_losses = self._run_epoch(self.test_ds, training=False)

        self._print_loss(test_losses,'test')
        return test_losses
 


