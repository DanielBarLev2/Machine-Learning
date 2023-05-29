import numpy as np
import activation_functions

def initialize_parameters(layer_dims: np.array) -> dict:
    """
    input: an array of the dimensions of each layer in the network.
    (layer 0 is the size of the flattened input, layer L is the output softmax)

    output: a dictionary containing the initialized w and b parameters of each layer
    (W1…WL, b1…bL).
    """

    parameters = {}

    for layer in range(1, len(layer_dims)):
        parameters[f'w{layer}'] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) * 0.01
        parameters[f'b{layer}'] = np.zeros((layer_dims[layer]), 1)

    return parameters


def linear_forward(activation, w: np.array, b: np.array) -> tuple[int, tuple]:
    """
    Description: Implement the linear part of a layer's forward propagation

    input:
    activation – the activations of the previous layer
    w – the weight matrix of the current layer (of shape [size of current layer, size of previous layer])
    B – the bias vector of the current layer (of shape [size of current layer, 1])

    Output:
    Z – the linear component of the activations function (i.e., the value before applying the non-linear function)
    cache – a dictionary containing activation, w, b (stored for making the backpropagation easier to compute)
    """

    z = np.dot(w, activation) + b
    cache = (activation, w, b)

    return z, cache


def linear_activation_forward(activation_prev: np.array, w: np.array,
                              b: np.array, activation_fn: str) -> tuple[np.array, tuple]:
    """
    Description: Implement the forward propagation for the LINEAR -> ACTIVATION layer

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
        z, linear_cache = linear_forward(activation=activation_prev, w=w, b=b)
        activation, activation_cache = activation_functions.softmax(z=z)

        cache = (linear_cache, activation_cache)

        return activation, cache

    elif activation_fn.__eq__("relu"):
        z, linear_cache = linear_forward(activation=activation_prev, w=w, b=b)
        activation, activation_cache = activation_functions.relu(z=z)

        cache = (linear_cache, activation_cache)

        return activation, cache


def l_model_forward(data_x: np.array, parameters: dict) -> tuple[np.array, list]:
    """
    Description: Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation

    Input:
    x – the data, numpy array of shape (input size, number of examples)
    parameters – the initialized W and b parameters of each layer
    (note that this option needs to be set to “false” in Section 3 and “true” in Second 4).

    Output:
    activation_last – the last post-activation value
    caches – a list of all the cache objects generated by the linear_forward function
    """

    activation = data_x
    caches = []

    for layer in range(1, (len(parameters) // 2)):
        activation_prev = activation
        activation, cache = linear_activation_forward(activation_prev=activation_prev, w=parameters[f'w{layer}'],
                                                      b=parameters[f'b{layer}'], activation_fn="relu")
        caches.append(cache)

    activation_last, cache = linear_activation_forward(activation_prev=activation, w=parameters[f'w{layer}'],
                                                      b=parameters[f'b{layer}'], activation_fn="softmax")

    caches.append(cache)

    return activation_last, caches


def compute_cost(activation_last, y: np.array):
    """
    Description: Implement the cost function defined by equation. The requested cost function is categorical
    cross-entropy loss

    Input:
    activation_last – probability vector corresponding to your label predictions, shape (num_of_classes,
    number of examples)
    y – the labels vector (i.e. the ground truth)

    Output:
    cost – the cross-entropy cost
    """

    m = y.shape[1]

    cost = - (1 / m) * np.sum(np.multiply(y, np.log(activation_last)) + np.multiply(1 - y, np.log(1 - activation_last)))

    return cost