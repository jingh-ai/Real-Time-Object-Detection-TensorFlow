import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

print(tf.__file__)

def Conv_bn(filters, kernel_size, strides=1, padding='same', activation='relu'):
    """Convolutional layer followed by Batch Normalization and ReLU activation."""
    return Sequential([
        layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation(activation)
    ])


def conv_bn_act(
    inputs,
    filters,
    kernel_size,
    strides,
    padding="same",
    zero_pad=False,
    activation="leaky",
):
    """
    Convolutional layer followed by Batch Normalization and ReLU activation.

    Args:
        inputs (tf.Tensor): 4D (N,H,W,C) input tensor
        filters (int): Number of convolutional filters
        kernel_size (int): Size of the convolutional kernel
        strides (int): Strides used for the convolution
        padding (str): Type of padding used in the convolution
        zero_pad (bool): If true, will zero-pad the input
        activation (string): Activation layer. Can be "mish" or "leaky_relu", or linear otherwise

    Returns:
        tf.Tensor: 4D (N,H,W,C) output tensor
    """
    if zero_pad:
        inputs = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)

    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    if activation == "leaky_relu":
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    elif activation == "mish":
        x = tf.activations.siL(x)

    return x