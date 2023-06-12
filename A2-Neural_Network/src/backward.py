from activation_functions import relu_backward, softmax_backward
import numpy as np

def linear_backward(dz: np.ndarray, cache: tuple) -> tuple[np.array, np.ndarray, np.ndarray]:
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
    dw = np.dot(dz, activation_prev.T) / m

    # db(l) = dL/db(l)
    db = np.sum(dz, axis=1, keepdims=True) / m

    # dA(l-1) = dL/dA(l-1)
    da_prev = np.dot(w.T, dz)

    return da_prev, dw, db


def linear_activation_backward(da: np.ndarray, cache: dict, activation_function: str, layer: int) -> tuple:
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

    linear_cache = cache.get(f'x{layer}'), cache.get(f'W{layer}'), cache.get(f'b{layer}')
    activation_cache =  cache.get(f'z{layer}')

    if activation_function == "softmax":
        dz = softmax_backward(da=da, activation_cache=activation_cache)
        da_prev, dw, dx = linear_backward(dz=dz, cache=linear_cache)

    elif activation_function == "relu":
        dz = relu_backward(da=da, activation_cache=activation_cache)
        da_prev, dw, dx = linear_backward(dz=dz, cache=linear_cache)

    else:
        raise Exception('Non-supported activation function')

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

    # _train = y_train.reshape(last_activation.shape)
    layer = len(caches)
    gradients = {}

    # dl/da
    d_cost = cost_backward(last_activation=last_activation, y_train=y_train)

    # at the beginning layer, activates the softmax layer
    gradients[f'dA{layer - 1}'], gradients[f'dW{layer}'], gradients[f'db{layer}'] =\
        linear_activation_backward(da=d_cost,cache=caches[layer - 1], activation_function="softmax", layer=layer)

    # back propagate the linear activation using relu
    for layer in range(layer - 1, 0, -1):
        current_cache = caches[layer - 1]

        gradients[f'dA{layer - 1}'], gradients[f'dW{layer}'], gradients[f'db{layer}'] = \
            linear_activation_backward(da=gradients[f'dA{layer}'], cache=current_cache, activation_function="relu",
                                       layer=layer)

    return gradients


def cost_backward(last_activation: np.ndarray, y_train: np.array) ->  np.array:
    """
    Description: Implement the cost function defined by equation.
    The requested cost function is categorical cross-entropy loss.

    Input:
    y_train (ndarray of shape (n, m)) – A one-hot encoding of the true class labels.
    Each row constitutes a training example, and each column is a different class.
    last_activation (ndarray of shape (n, m)) – The network predictions for the probability of each of m class labels
    on each of n examples in a batch.

    Output:
    grad (ndarray of shape (n, m)) – The gradient of the cross-entropy loss with respect to the input to the softmax
    function.
    """

    d_cost = last_activation - y_train

    return d_cost
