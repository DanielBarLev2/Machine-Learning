import numpy as np

# forward propagation
def initialize_parameters(layer_dims: np.array) -> dict:
    """
    input: an array of the dimensions of each layer in the network.
    (layer 0 is the size of the flaened input, layer L is the output softmax)

    output: a diconary containing the inialized W and b parameters of each layer
    (W1…WL, b1…bL).
    """

    parameters = {}
    l = len(layer_dims)

    for layer in range(1, len(layer_dims)):
        parameters[f'W{layer}'] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) * 0.01
        parameters[f'b{layer}'] = np.zeros((layer_dims[layer]), 1)

    return parameters


def linear_forward(A, W: np.array, b: np.array) -> tuple(int, dict):
    """
    Description: Implement the linear part of a layer's forward propagaon

    input:
    A – the activations of the previous layer
    W – the weight matrix of the current layer (of shape [size of current layer, size of previous layer])
    B – the bias vector of the current layer (of shape [size of current layer, 1])

    Output:
    Z – the linear component of the activations function (i.e., the value before applying the non-linear funcon)
    linear_cache – a diconary containing A, W, b (stored for making the backpropagaon easier to compute)
    """

    Z = np.dot(W, A) + b
    linear_cache = {A: A, W: W, b: b}

    return Z, linear_cache


def softmax(Z: np.array) -> tuple(np.array, np.array):
    """
    Input:
    Z – the linear component of the activation function

    Output:
    A – the activations of the layer
    activations_cache – returns Z, which will be useful for the backpropagation
    """

    A = np.exp(Z) / np.sum(Z, axis=0)
    activations_cache = Z

    return A, activations_cache


def relu(Z: np.array) -> tuple(np.array, np.array):
    """
    Input:
    Z – the linear component of the activation function

    Output:
    A – the activation of the layer
    activation_cache – returns Z, which will be useful for the backpropagation
    """
    A = np.maximum(0, Z)
    activations_cache = Z

    return A, activations_cache


def linear_activation_forward(A_prev: np.array, W: np.array, B: np.array, activation_fn: str) -> tuple(np.array,
                                                                                                     np.array):
    """
    Description: Implement the forward propagaon for the LINEAR -> ACTIVATION layer

    Input:
    A_prev – activation of the previous layer
    W – the weights matrix of the current layer
    B – the bias vector of the current layer
    activation – the activation function to be used (a string, either “softmax” or “relu”)

    Output:
    A – the activation of the current layer
    cache – a joint dictionary containing both linear_cache and activation_cache
    """

    if activation_fn.__eq__("softmax"):
        Z, linear_cache = linear_forward(A=A_prev, W=W, b=b)
        A, activation_cache = softmax(Z=Z)

    elif activation_fn.__eq__("relu"):
        Z, linear_cache = linear_forward(A=A_prev, W=W, b=b)
        A, activation_cache = relu(Z=Z)

    cache = [linear_cache, activation_cache]

    return A, cache
