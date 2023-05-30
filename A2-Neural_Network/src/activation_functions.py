import numpy as np

def sigmoid(z: np.ndarray) -> tuple[np.array, np.ndarray]:
    """
    Input:
    Z – the linear component of the activation function

    Output:
    activation – the activations of the layer
    activations_cache – returns Z, which will be useful for the backpropagation
    """

    activation = 1 / (1 + np.exp(-z))
    activations_cache = z

    return activation, activations_cache


def relu(z: np.array) -> tuple[np.array, np.ndarray]:
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


def relu_backward(da: np.ndarray, activation_cache: np.ndarray) -> np.array:
    """
    Description: Implements backward propagation for a ReLU unit

    Input:
    dA – the post-activation gradient
    activation_cache – contains z (stored during the forward propagation)

    Output:
    dz – gradient of the cost with respect to Z
    """

    z = activation_cache
    a, z = relu(z=z)
    dz = np.multiply(da, np.int64(a > 0))

    return dz


def sigmoid_backward(da: np.ndarray, activation_cache: np.ndarray) -> np.array:
    """
    Description: Implements backward propagation for a sigmoid unit

    Input:
    dA – the post-activation gradient
    activation_cache – contains Z (stored during the forward propagation)

    Output:
    dZ – gradient of the cost with respect to Z
    """

    z = activation_cache
    a, z = sigmoid(z=z)

    dz = da * sigmoid(a) * sigmoid(1 - a)

    return dz
