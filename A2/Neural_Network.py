import numpy as np

def initialize_parameters(layer_dims: np.array, x,y) -> dict:
    """
    input: an array of the dimensions of each layer in the network.
    (layer 0 is the size of the flaened input, layer L is the output softmax)

    output: a diconary containing the inialized W and b parameters of each layer
    (W1…WL, b1…bL).
    """

    parameters = {}
    l = len(layer_dims)

    for layer in range(1, len(layer_dims)):
        parameters[f'W{layer}'] = np.randn(layer_dims[layer], layer_dims[layer - 1])
        parameters[f'b{layer}'] = np.zeros(layer_dims[layer], 1)

    return parameters

# test:
