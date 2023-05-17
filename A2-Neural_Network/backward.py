import numpy as np

def linear_backward(dz, cache):
    """
    description: Implements the linear part of the backward propagation process for a single layer

    Input:
    dz – the gradient of the cost with respect to the linear output of the current layer (layer l)
    cache – tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Output:
    da_prev - gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- gradient of the cost with respect to W (current layer l), same shape as W
    db -- gradient of the cost with respect to b (current layer l), same shape as b
    """

    activation_prev, w, b = cache
    m = activation_prev.shape[1]

    # dW(l) = dL/db(l)
    dw = (1 / m) * np.dot(dz, activation_prev.T)

    # db(l) = dl/db(l)
    db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

    # dA(l-1) = dL/dA(l-1)
    da_prev = np.dot(w.T, dz)

    return da_prev, dw, db