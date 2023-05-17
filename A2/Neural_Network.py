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
    Descripon: Implement the linear part of a layer's forward propagaon

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


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z


def tanh(Z):
    A = np.tanh(Z)
    return A, Z


def relu(Z):
    A = np.maximum(0, Z)
    return A, Z


def leaky_relu(Z):
    A = np.maximum(0.1 * Z, Z)
    return A, Z