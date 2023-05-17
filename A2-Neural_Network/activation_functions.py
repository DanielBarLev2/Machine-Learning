import numpy as np

def softmax(z: np.array) -> tuple[np.array, np.array]:
    """
    Input:
    Z – the linear component of the activation function

    Output:
    activation – the activations of the layer
    activations_cache – returns Z, which will be useful for the backpropagation
    """

    activation = np.exp(z) / np.sum(z, axis=0)
    activations_cache = z

    return activation, activations_cache


def relu(z: np.array) -> tuple[np.array, np.array]:
    """
    Input:
    Z – the linear component of the activation function

    Output:
    activation – the activation of the layer
    activation_cache – returns Z, which will be useful for the backpropagation
    """
    activation = np.maximum(0, z)
    activations_cache = z

    return activation, activations_cache
