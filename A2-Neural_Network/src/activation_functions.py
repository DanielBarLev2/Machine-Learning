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


def relu_backward(da: any, activation_cache: any) -> np.array:
    """
    Description: Implements backward propagation for a ReLU unit

    Input:
    dA – the post-activation gradient
    activation_cache – contains z (stored during the forward propagation)

    Output:
    dz – gradient of the cost with respect to Z
    """

    z = activation_cache
    dz = np.array(da, copy=True)
    dz[z <= 0] = 0

    return dz


def softmax_backward(da: any, activation_cache: tuple) -> np.array:
    """
    Description: Implements backward propagation for a softmax unit

    Input:
    dA – the post-activation gradient
    activation_cache – contains Z (stored during the forward propagation)

    Output:
    dZ – gradient of the cost with respect to Z
    """

    z = activation_cache

    p = softmax(z=z)
