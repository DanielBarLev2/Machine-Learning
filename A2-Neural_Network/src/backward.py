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


def linear_activation_backward(da: any, cache: tuple, activation: str) -> tuple:
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


def l_model_backward(al: np.array, y: np.array, caches: list) -> dict:
    """
    Description: Implement the backward propagation process for the enre network.
    The backpropagation for the softmax function should be done only once as only the
    output layers uses it and the RELU should be done iteratively over all the remaining
    layers of the network.

    Input:
    AL - the probabilities vector, the output of the forward propagation (L_model_forward)
    Y - the true labels vector (the "ground truth" - true classifications)
    Caches - list of caches containing for each layer: a) the linear cache; b) the activation cache

    Output: Grads - a dictionary with the gradients
    """
    y = y.reshape(al.shape)
    l = len(caches)
    grads = {}

    dal = np.divide(al - y, np.multiply(al, 1 - al))

    grads[f'dA + {l - 1}'], grads[f'dW + {l}'], grads[f'db + {l}'] =\
        linear_activation_backward(da=dal,cache=caches[l - 1], activation="softmax")

    for l in range(l - 1, 0, -1):
        current_cache = caches[l - 1]

        grads[f'dA + {l - 1}'], grads[f'dW + {l}'], grads[f'db + {l}'] = \
            linear_activation_backward(da=grads[f'dA + {l}'], cache=current_cache, activation="relu")

    return grads
