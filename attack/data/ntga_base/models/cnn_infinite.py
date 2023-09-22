from neural_tangents import stax

def ConvBlock(channels, W_std, b_std, strides=(1, 1)):
    return stax.serial(stax.Conv(out_chan=channels, filter_shape=(5, 5), strides=strides, 
                                 padding='SAME', W_std=W_std, b_std=b_std), 
                       stax.Relu(do_backprop=True))

def ConvGroup(n, channels, stride, W_std, b_std):
    """
    :param n: int. Number of layers.
    :param channels: int. Number of channels in each layer (finite-width case).
            For the infinite-width neuron network, this parameter is meaningless.
    :param: stride: (int, int). Stride number used in the convolution operation.
    :param W_std: float. Standard deviation of weights at initialization.
    :param b_std: float. Standard deviation of biases at initialization.
    :return: a callable convolutional neural network.
    """
    blocks = []
    for i in range(n):
        blocks += [ConvBlock(channels, W_std, b_std, stride)]
    return stax.serial(*blocks)
