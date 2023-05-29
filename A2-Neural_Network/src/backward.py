from activation_functions import softmax_backward, relu_backward
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


def linear_activation_backward(da, cache, activation):
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

    if activation == "softmax":
        dz = softmax_backward(da=da, activation_cache=activation_cache)
        da_prev, dw, dx = linear_backward(dz=dz, cache=linear_cache)

    elif activation == "relu":
        dz = relu_backward(da=da, activation_cache=activation_cache)
        da_prev, dw, dx = linear_backward(dz=dz, cache=linear_cache)

    else:
        da_prev, dw, dx = (0, 0, 0)

    return da_prev, dw, dx



