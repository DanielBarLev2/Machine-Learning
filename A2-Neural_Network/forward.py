import numpy as np
import activation_functions

# forward propagation
def initialize_parameters(layer_dims: np.array) -> dict:
    """
    input: an array of the dimensions of each layer in the network.
    (layer 0 is the size of the flaened input, layer L is the output softmax)

    output: a diconary containing the inialized w and b parameters of each layer
    (W1…WL, b1…bL).
    """

    parameters = {}
    l = len(layer_dims)

    for layer in range(1, len(layer_dims)):
        parameters[f'w{layer}'] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) * 0.01
        parameters[f'b{layer}'] = np.zeros((layer_dims[layer]), 1)

    return parameters


def linear_forward(activation, w: np.array, b: np.array) -> tuple[int, dict]:
    """
    Description: Implement the linear part of a layer's forward propagaon

    input:
    activation – the activations of the previous layer
    w – the weight matrix of the current layer (of shape [size of current layer, size of previous layer])
    B – the bias vector of the current layer (of shape [size of current layer, 1])

    Output:
    Z – the linear component of the activations function (i.e., the value before applying the non-linear funcon)
    linear_cache – a diconary containing activation, w, b (stored for making the backpropagaon easier to compute)
    """

    z = np.dot(w, activation) + b
    linear_cache = {activation: activation, w: w, b: b}

    return z, linear_cache


def linear_activation_forward(activation_prev: np.array, w: np.array,
                              B: np.array, activation_fn: str) -> tuple[np.array, np.array]:
    """
    Description: Implement the forward propagaon for the LINEAR -> ACTIVATION layer

    Input:
    activation_prev – activation of the previous layer
    w – the weights matrix of the current layer
    B – the bias vector of the current layer
    activation – the activation function to be used (a string, either “softmax” or “relu”)

    Output:
    activation – the activation of the current layer
    cache – a joint dictionary containing both linear_cache and activation_cache
    """

    if activation_fn.__eq__("softmax"):
        Z, linear_cache = linear_forward(activation=activation_prev, w=w, b=b)
        activation, activation_cache = activation_functions.softmax(Z=Z)

    elif activation_fn.__eq__("relu"):
        Z, linear_cache = linear_forward(activation=activation_prev, w=w, b=b)
        activation, activation_cache = activation_functions.relu(Z=Z)

    cache = [linear_cache, activation_cache]

    return activation, cache
