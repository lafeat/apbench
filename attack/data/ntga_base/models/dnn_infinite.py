from neural_tangents import stax

def DenseBlock(neurons, W_std, b_std):
    return stax.serial(stax.Dense(neurons, W_std, b_std), 
                       stax.Erf())

def DenseGroup(n, neurons, W_std, b_std):
    """
    :param n: int. Number of layers.
    :param neurons: int. Number of neurons in each layer (finite-width case).
            For the infinite-width neuron network, this parameter is meaningless.
    :param W_std: float. Standard deviation of weights at initialization.
    :param b_std: float. Standard deviation of biases at initialization.
    :return: a callable fully-connected neural network.
    """
    blocks = []
    for _ in range(n):
        blocks += [DenseBlock(neurons, W_std, b_std)]
    return stax.serial(*blocks)
