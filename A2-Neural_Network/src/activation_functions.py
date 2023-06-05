import numpy as np

def relu(z: np.array) -> tuple[np.array, np.ndarray]:
    """
    Description: Implements forward propagation for a relu unit.

    Input:
    Z – the linear component of the activation function.

    Output:
    activation – the activation of the layer.
    activation_cache – returns Z, which will be useful for the backpropagation.
    """
    activation = np.maximum(0, z)
    activations_cache = z

    return activation, activations_cache


def relu_backward(da: np.ndarray, activation_cache: np.ndarray) -> np.array:
    """
    Description: Implements backward propagation for a ReLU unit.

    Input:
    dA – the post-activation gradient.
    activation_cache – contains z (stored during the forward propagation).

    Output:
    dz – gradient of the cost with respect to Z.
    """

    z = activation_cache
    a, z = relu(z=z)
    dz = np.multiply(da, np.int64(a > 0))

    return dz


def softmax(z: np.ndarray) -> tuple[np.array, np.ndarray]:
    """
    Description: Implements forward propagation for a Softmax unit.

    Input:
    Z – the linear component of the activation function.

    Output:
    activation – the activations of the layer.
    activations_cache – returns Z, which will be useful for the backpropagation.
    """

    activation = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
    activations_cache = z

    return activation, activations_cache


def softmax_backward(da, activation_cache):
    """
    Description: Implements backward propagation for a Softmax unit.

    Arguments:
    da -- true labels, numpy array of shape (n_classes, m)
    cache -- input Z stored during forward propagation

    Returns:
    dz -- gradient of the cost with respect to Z
    """

    z = activation_cache
    a, z = softmax(z=z)
    dz = da * a * (1-a)

    return dz