from activation_functions import sigmoid_backward, relu_backward
import numpy as np

def linear_backward(dz: any, cache: tuple) -> tuple[np.array, any, any]:
    """
    description: Implements the linear part of the backward propagation process for a single layer

    Input:
    dz – the gradient of the cost with respect to the linear output of the current layer (layer l)
    cache – tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Output:
    da_prev - gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dw - gradient of the cost with respect to W (current layer l), same shape as W
    db - gradient of the cost with respect to b (current layer l), same shape as b
    """

    activation_prev, w, b = cache
    m = activation_prev.shape[1]

    # dW(l) = dL/db(l)
    dw = (1 / m) * np.dot(dz, activation_prev.T)

    # db(l) = dL/db(l)
    db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

    # dA(l-1) = dL/dA(l-1)
    da_prev = np.dot(w.T, dz)

    return da_prev, dw, db


def linear_activation_backward(da: any, cache: tuple, activation_function: str) -> tuple:
    """
    Description: Implements the backward propagation for the LINEAR->ACTIVATION layer. The function first computes
    dZ and then applies the linear_backward function.

    Input:
    da – post activation gradient of the current layer
    cache – contains both the linear cache and the activation cache

    Output:
    da_prev – Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW – Gradient of the cost with respect to W (current layer l), same shape as W
    db – Gradient of the cost with respect to b (current layer l), same shape as b
    """

    linear_cache, activation_cache = cache

    if activation_function == "sigmoid":
        dz = sigmoid_backward(da=da, activation_cache=activation_cache)
        da_prev, dw, dx = linear_backward(dz=dz, cache=linear_cache)

    elif activation_function == "relu":
        dz = relu_backward(da=da, activation_cache=activation_cache)
        da_prev, dw, dx = linear_backward(dz=dz, cache=linear_cache)

    else:
        da_prev, dw, dx = (0, 0, 0)

    return da_prev, dw, dx


def l_model_backward(last_activation: np.array, y_train: np.array, caches: list) -> dict:
    """
    Description: Implement the backward propagation process for the enre network.
    The backpropagation for the sigmoid function should be done only once as only the
    output layers uses it and the RELU should be done iteratively over all the remaining
    layers of the network.

    Input:
    last_activation - the probabilities vector, the output of the forward propagation (L_model_forward)
    y_train - the true labels vector (the "ground truth" - true classifications)
    Caches - list of caches containing for each layer: the linear cache; b the activation cache

    Output: Grads - a dictionary with the gradients
    """
    y_train = y_train.reshape(last_activation.shape)
    layer = len(caches)
    gradients = {}

    d_la = np.divide(last_activation - y_train, np.multiply(last_activation, 1 - last_activation))

    # at the beginning layer, activates the sigmoid layer
    gradients[f'dA{layer - 1}'], gradients[f'dW{layer}'], gradients[f'db{layer}'] =\
        linear_activation_backward(da=d_la,cache=caches[layer - 1], activation_function="sigmoid")

    # back propagate the linear activation using relu
    for layer in range(layer - 1, 0, -1):
        current_cache = caches[layer - 1]

        gradients[f'dA{layer - 1}'], gradients[f'dW{layer}'], gradients[f'db{layer}'] = \
            linear_activation_backward(da=gradients[f'dA{layer}'], cache=current_cache, activation_function="relu")

    return gradients


def update_parameters(parameters, grads, learning_rate):
    """
    Description: Updates parameters using gradient descent

    Input:
    parameters – a python dictionary containing the DNN architecture’s parameters
    grads – a python dictionary containing the gradients (generated by L_model_backward)
    learning_rate – the learning rate used to update the parameters (the “alpha”)

    Output:
    parameters – the updated values of the parameters object provided as input
    """

    l = len(parameters) // 2

    for l in range(1, l + 1):
        parameters[f'W{l}'] = parameters[f'W{l}'] - learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] = parameters[f'b{l}'] - learning_rate * grads[f'db{l}']

    return parameters
