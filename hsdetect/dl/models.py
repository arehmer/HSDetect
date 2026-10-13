"""
model.py

Assembles the layer classes from layers.py into the full ultra-lightweight
CNN + YOLOv2 detection head described in Vandersteegen et al. (2023),
Section 3.2 / Figure 4.

The network is built as a `tf.keras.Model` subclass. Input spatial size is
fixed at construction time (`input_shape=(H, W, 1)`) -- it is NOT a variable
"None" shape -- but you can build separate instances of this class for
different fixed resolutions (e.g. one model for the 32x24 MLX90640 sensor,
another for a 160x120 sensor), since the network is fully convolutional and
has no dense layers.

Channel progression (matches Figure 4, and yields the paper's ~1.26M
parameter uncompressed model before pruning/quantization):

    Input                                   (H,   W,   1)
    Conv 3x3, 16 ch, stride 1               (H,   W,   16)
    DWSep 3x3, 32 ch,  stride 1             (H,   W,   32)
    DWSep 3x3, 64 ch,  stride 2             (H/2, W/2, 64)
    DWSep 3x3, 128 ch, stride 1             (H/2, W/2, 128)
    DWSep 3x3, 256 ch, stride 2             (H/4, W/4, 256)
    DWSep 3x3, 512 ch, stride 1             (H/4, W/4, 512)
    DWSep 3x3, 1024 ch, stride 1            (H/4, W/4, 1024)
    DWSep 3x3, 512 ch, stride 1             (H/4, W/4, 512)
    Conv 1x1 head, num_anchors*(5+C) ch     (H/4, W/4, num_anchors*(5+C))

This gives the overall x4 downsampling factor stated in the paper (as
opposed to the x32 used by higher-resolution detectors), preserving enough
spatial resolution for the ultra low-res thermal input.
"""
from pathlib import Path

import tensorflow as tf

from .layers import ConvBNAct, DepthwiseSeparableConv, DetectionHead
import keras


""" Dummy parent class for YOLOv2Tiny-like models """
@keras.saving.register_keras_serializable()
class YOLOv2Tiny(tf.keras.Model):
    
    def __init__(self,
                 name : str,
                 save_path : str,
                 input_shape = None,
                 num_anchors = None,
                 num_classes = None,
                  **kwargs,
                 ):
        super(YOLOv2Tiny,self).__init__(name=name)
        
        
        self.save_path = save_path
        self.input_shape = input_shape
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        self.build_model()

    @property
    def save_path(self):
        """Path that is passed to tf.keras.Model.save()"""
        return self._save_path
    @save_path.setter
    def save_path(self,save_path : str):
        if isinstance(save_path,Path):
            save_path = save_path.as_posix()
        if not isinstance(save_path,str):
            raise TypeError(f'Expected {str}, received {type(save_path)}')
        self._save_path = save_path
        
    @property
    def input_shape(self):
        """(H, W, C) this model instance was built for."""
        return self._input_shape
    @input_shape.setter
    def input_shape(self,input_shape:tuple):
        if input_shape is not None:
            input_shape = tuple(input_shape)
        self._input_shape = input_shape

    @property
    def num_anchors(self):
        return self._num_anchors     
    @num_anchors.setter
    def num_anchors(self,num_anchors:int):
        if num_anchors is not None:
            num_anchors = int(num_anchors)
        self._num_anchors = num_anchors

    @property
    def num_classes(self):
        return self._num_classes
    @num_classes.setter
    def num_classes(self,num_classes:int):
        if num_classes is not None:
            num_classes = int(num_classes)
        self._num_classes = num_classes
        
    @property
    def grid_shape(self):
        """(grid_h, grid_w) of the backbone's output feature map, computed by
        running a dummy input through the backbone and reading its shape.
        """
        dummy = tf.zeros((1,) + self.input_shape)
        x = dummy
        for layer in self.backbone:
            x = layer(x, training=False)
        _, h, w, _ = x.shape
        return (h, w)

    @property
    def num_output_channels(self):
        return self.num_anchors * (5 + self.num_classes)
    
    @property
    def output_shape(self):
        
        # n_batch x grid_h x grid_w x num_anchors * (5 + self.num_classes)
        output_shape = (n_batch,) + self.grid_shape + (self.num_output_channels,)
        
        return output_shape

   
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape" : self.input_shape,
            "num_anchors" : self.num_anchors,
            "num_classes" : self.num_classes,
            "save_path" : self.save_path
            })
        return config
    
    def call(self, x, training=False):
        for layer in self.backbone:
            x = layer(x, training=training)
        x = self.head(x, training=training)
        return x
    
    def save(self):
        """
        Wrapper for tf.keras.Model.save(), which calls tf.keras.Model.save()
        with self.path / self.name as argument.

        Returns
        -------
        None.

        """
        # Convert str to Path
        save_path = Path(self.save_path)
        
        path = ( save_path / (self.name + '.keras')).as_posix()
        super(YOLOv2Tiny,self).save(path)
        
        return path
        

@keras.saving.register_keras_serializable()
class YOLOv2Tiny4X(YOLOv2Tiny):
    """Ultra-lightweight YOLOv2-tiny-style person detector.

    Parameters
    ----------
    input_shape : tuple(int, int, int)
        Fixed (H, W, C) input shape this model instance is built for, e.g.
        (24, 32, 1) for the MLX90640 sensor, or (120, 160, 1) for a larger
        thermal sensor. C is typically 1 (single-channel background-
        subtracted thermal image).
    num_anchors : int
        Number of anchor boxes per grid cell (paper uses 5).
    num_classes : int
        Number of explicit class-logit channels per anchor. The paper's
        single-class ("person") model folds the class into the objectness
        score and uses no separate class channel -- pass num_classes=0 to
        reproduce that exact 25-channel head (5 anchors x 5 values). Pass
        num_classes=1 (default here) if you'd rather have an explicit,
        separately-supervised class channel usable with the classification
        term of the configurable loss in loss.py.
    """
        
    def build_model(self):
        """
        Build the model's layers.

        Returns
        -------
        None.

        """
        # Model can only be built if all attributes are set
        build_possible = all([a is not None for a in [self.input_shape,
                                                      self.num_anchors,
                                                      self.num_classes]])
        
        if build_possible:
        
            self.conv1 = ConvBNAct(filters=16, kernel_size=3, strides=1, name="conv1_3x3")
    
            self.dwsep1 = DepthwiseSeparableConv(filters=32, strides=1, name="dwsep1")
            self.dwsep2 = DepthwiseSeparableConv(filters=64, strides=2, name="dwsep2_down")
            self.dwsep3 = DepthwiseSeparableConv(filters=128, strides=1, name="dwsep3")
            self.dwsep4 = DepthwiseSeparableConv(filters=256, strides=2, name="dwsep4_down")
            self.dwsep5 = DepthwiseSeparableConv(filters=512, strides=1, name="dwsep5")
            self.dwsep6 = DepthwiseSeparableConv(filters=1024, strides=1, name="dwsep6")
            self.dwsep7 = DepthwiseSeparableConv(filters=512, strides=1, name="dwsep7")
    
            # Organize all backbone layers in a list. used both by
            # call() and by grid_shape below 
            self.backbone = [self.conv1, self.dwsep1, self.dwsep2, 
                             self.dwsep3, self.dwsep4, self.dwsep5, 
                             self.dwsep6, self.dwsep7]
    
            self.head = DetectionHead(num_anchors=self.num_anchors,
                                      num_classes=self.num_classes,
                                      name="detection_head")
        
            # Build the model's variables now, at a known fixed input shape,
            # rather than lazily on first call. Running a dummy forward pass
            # is the reliable way to build a subclassed tf.keras.Model with
            # nested custom layers (a plain .build() call does not recurse
            # into custom Layer.call-based sublayers).
            dummy = tf.zeros((1,) + self.input_shape)
            self(dummy, training=False)
            
        else:
            print('Model not built because not all properties set.')



@keras.saving.register_keras_serializable()
class YOLOv2Tiny8X(YOLOv2Tiny):
    """
    Derived from YOLOv2Tiny4X. Only difference is an additional downsampling
    (implemented in layer dwsep7 of the backbone), which results in a total
    downsampling factor of 8.

    Parameters
    ----------
    input_shape : tuple(int, int, int)
        Fixed (H, W, C) input shape this model instance is built for, e.g.
        (24, 32, 1) for the MLX90640 sensor, or (120, 160, 1) for a larger
        thermal sensor. C is typically 1 (single-channel background-
        subtracted thermal image).
    num_anchors : int
        Number of anchor boxes per grid cell (paper uses 5).
    num_classes : int
        Number of explicit class-logit channels per anchor. The paper's
        single-class ("person") model folds the class into the objectness
        score and uses no separate class channel -- pass num_classes=0 to
        reproduce that exact 25-channel head (5 anchors x 5 values). Pass
        num_classes=1 (default here) if you'd rather have an explicit,
        separately-supervised class channel usable with the classification
        term of the configurable loss in loss.py.
    """
    
    def build_model(self):
        """
        Build the model's layers.

        Returns
        -------
        None.

        """
        # Model can only be built if all attributes are set
        build_possible = all([a is not None for a in [self.input_shape,
                                                      self.num_anchors,
                                                      self.num_classes]])
        
        if build_possible:
            
            self.conv1 = ConvBNAct(filters=16, kernel_size=3, strides=1, name="conv1_3x3")
    
            self.dwsep1 = DepthwiseSeparableConv(filters=32, strides=1, name="dwsep1")
            self.dwsep2 = DepthwiseSeparableConv(filters=64, strides=2, name="dwsep2_down")
            self.dwsep3 = DepthwiseSeparableConv(filters=128, strides=1, name="dwsep3")
            self.dwsep4 = DepthwiseSeparableConv(filters=256, strides=2, name="dwsep4_down")
            self.dwsep5 = DepthwiseSeparableConv(filters=512, strides=1, name="dwsep5")
            self.dwsep6 = DepthwiseSeparableConv(filters=1024, strides=2, name="dwsep6_down")
            self.dwsep7 = DepthwiseSeparableConv(filters=512, strides=1, name="dwsep7")
    
            # Organize all backbone layers in a list. used both by
            # call() and by grid_shape below 
            self.backbone = [self.conv1, self.dwsep1, self.dwsep2, 
                             self.dwsep3, self.dwsep4, self.dwsep5, 
                             self.dwsep6, self.dwsep7]
    
            self.head = DetectionHead(num_anchors=self.num_anchors,
                                      num_classes=self.num_classes,
                                      name="detection_head")   
            
            # Build the model's variables now, at a known fixed input shape,
            # rather than lazily on first call. Running a dummy forward pass
            # is the reliable way to build a subclassed tf.keras.Model with
            # nested custom layers (a plain .build() call does not recurse
            # into custom Layer.call-based sublayers).
            batch_len1 = tf.zeros((1,) + self.input_shape)
            self(batch_len1, training=False)
            
        else:
            print('Model not built because not all properties set.')

if __name__ == "__main__":
    # Self-test with two different fixed input resolutions, demonstrating
    # that separate model instances can be built for different sizes.

    # Native MLX90640 resolution used in the paper (32x24, given as H=24, W=32).
    model_small = YOLOv2Tiny(input_shape=(24, 32, 1), num_anchors=5, num_classes=0)
    dummy_small = tf.random.normal((2, 24, 32, 1))
    out_small = model_small(dummy_small, training=False)
    print("Small-input model output shape:", out_small.shape)  # (2, 6, 8, 25)
    print("Small-input grid shape:", model_small.grid_shape)   # (6, 8)
    print("Small-input param count:", model_small.count_params())

    # A larger hypothetical thermal sensor (e.g. 160x120).
    model_large = YOLOv2Tiny(input_shape=(120, 160, 1), num_anchors=5, num_classes=1)
    dummy_large = tf.random.normal((1, 120, 160, 1))
    out_large = model_large(dummy_large, training=False)
    print("Large-input model output shape:", out_large.shape)  # (1, 30, 40, 30)
    print("Large-input grid shape:", model_large.grid_shape)   # (30, 40)
    print("Large-input param count:", model_large.count_params())
