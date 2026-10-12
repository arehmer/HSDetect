"""
layers.py

Custom tf.keras.layers.Layer building blocks used to assemble the
ultra-lightweight YOLOv2-tiny-style person detector described in:

    Vandersteegen et al., "Person Detection Using an Ultra Low-resolution
    Thermal Imager on a Low-cost MCU" (2023).

Per Section 3.2 / Figure 4 of that paper, the backbone consists of:
  - a single 3x3 standard convolution
  - 7 depthwise-separable convolutions (some with stride 2 for downsampling)
  - a final 1x1 convolution "head" (linear activation) producing the
    YOLOv2-style bounding-box tensor

ReLU6 is used everywhere in the backbone (chosen by the authors because it
can be fused into the preceding convolution at deployment time), and every
conv (except the final 1x1 head) is followed by BatchNorm.
"""

import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class ConvBNAct(tf.keras.layers.Layer):
    """Standard Conv2D -> BatchNorm -> Activation block.

    Used for the very first 3x3 convolution of the backbone (Figure 4,
    top-left "Conv 3x3" box).
    """

    def __init__(self, filters, kernel_size=3, strides=1, activation="relu6",
                 use_bn=True, padding="same", name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bn = use_bn
        self.activation_name = activation

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=not use_bn,  # bias is redundant when followed by BN
        )
        self.bn = tf.keras.layers.BatchNormalization() if use_bn else None
        self.act = _get_activation(activation)

    def call(self, x, training=False):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x, training=training)
        if self.act is not None:
            x = self.act(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "use_bn": self.use_bn,
            "activation": self.activation_name,
        })
        return config

@keras.saving.register_keras_serializable()
class DepthwiseSeparableConv(tf.keras.layers.Layer):
    """Depthwise-separable convolution block:

        DepthwiseConv(3x3, stride) -> BN -> ReLU6
        -> Conv(1x1) -> BN -> ReLU6

    This corresponds to the "DW sep 3x3" boxes in Figure 4, whose internal
    structure is shown in the block legend on the right of that figure
    ("DW Conv 3x3 /X" -> BN -> ReLU6 -> "Conv 1x1" -> BN -> ReLU6).

    `filters` sets the number of output (pointwise) channels;
    `strides` is applied on the depthwise 3x3 convolution and controls
    spatial downsampling (the paper uses stride 2 twice to achieve the
    overall x4 downsampling factor of the network).
    """

    def __init__(self, filters, strides=1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.strides = strides

        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=strides,
            padding="same",
            use_bias=False,
        )
        self.dw_bn = tf.keras.layers.BatchNormalization(momentum=0.1)
        self.dw_act = _get_activation("relu6")

        self.pw_conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
        )
        self.pw_bn = tf.keras.layers.BatchNormalization(momentum=0.1)
        self.pw_act = _get_activation("relu6")
        print('Set momentum to 0.1 for debugging!')
    def call(self, x, training=False):
        x = self.dw_conv(x)
        x = self.dw_bn(x, training=training)
        x = self.dw_act(x)

        x = self.pw_conv(x)
        x = self.pw_bn(x, training=training)
        x = self.pw_act(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "strides": self.strides})
        return config

@keras.saving.register_keras_serializable()
class DetectionHead(tf.keras.layers.Layer):
    """Final 1x1 convolution producing the raw YOLOv2-style output tensor.

    No BatchNorm and a linear activation are used here, since this layer
    directly predicts box offsets/confidence/(optional) class logits which
    are decoded downstream (see loss.py / postprocessing), matching the
    "Conv 1x1" -> "YOLOv2 post process" path in Figure 4.
    """

    def __init__(self, num_anchors, num_classes=1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # 5 = (tx, ty, tw, th, objectness). +num_classes for explicit class
        # logits. The original paper only ever has a single "person" class
        # and folds it into the objectness score, giving exactly 5 values
        # per anchor (25 total channels for 5 anchors). Set num_classes=0
        # to reproduce that exact 25-channel head.
        out_channels = num_anchors * (5 + num_classes)
        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=1,
            strides=1,
            padding="same",
            activation=None,
            use_bias=True,
        )

    def call(self, x, training=False):
        return self.conv(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_anchors": self.num_anchors,
            "num_classes": self.num_classes,
        })
        return config

@keras.saving.register_keras_serializable()
def _get_activation(name):
    if name is None:
        return None
    if name == "relu6":
        return tf.keras.layers.ReLU(max_value=6.0)
    return tf.keras.layers.Activation(name)


if __name__ == "__main__":
    # Minimal self-test / smoke test for each layer type.
    dummy = tf.random.normal((2, 24, 32, 1))

    conv = ConvBNAct(filters=16, kernel_size=3, strides=1)
    out = conv(dummy, training=True)
    print("ConvBNAct output shape:", out.shape)  # (2, 24, 32, 16)

    dwsep_stride1 = DepthwiseSeparableConv(filters=32, strides=1)
    out2 = dwsep_stride1(out, training=True)
    print("DepthwiseSeparableConv (stride 1) output shape:", out2.shape)  # (2, 24, 32, 32)

    dwsep_stride2 = DepthwiseSeparableConv(filters=64, strides=2)
    out3 = dwsep_stride2(out2, training=True)
    print("DepthwiseSeparableConv (stride 2) output shape:", out3.shape)  # (2, 12, 16, 64)

    head = DetectionHead(num_anchors=5, num_classes=0)
    out4 = head(out3, training=False)
    print("DetectionHead output shape:", out4.shape)  # (2, 12, 16, 25)
