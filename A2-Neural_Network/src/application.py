from forward import initialize_parameters, l_model_forward, cross_entropy_loss
from backward import l_model_backward, update_parameters, compute_cost_gradient
import numpy as np

def l_layer_model(x_train: np.ndarray, y_train: np.ndarray, layer_dims: np.ndarray, learning_rate: float,
                  num_iterations: int) -> dict:
    """
    Description: Implements L-layer neural network. All layers but the last should have the ReLU
    activation function, and the final layer will apply the sigmoid activation function.
    The size of the output layer should be equal to the number of labels in the data.
    Batch size is selected such that it enables the code to run well.

    Input:
    x_data – the input data, a numpy array of shape (height*width , number_of_examples).
    y_train – the “real” labels of the data, a vector of shape (num_of_classes, number of examples).
    Layer_dims – a list containing the dimensions of each layer, including the input.
    batch_size – the number of examples in a single training batch.
    learning_rate - by how amount to change the weight matrices.
    num_iterations - the learning cap or limit.

    Output:
    parameters – the parameters learnt by the system during the training.
    (the same parameters that were updated in the update_parameters function).
    costs – the values of the cost function (calculated by the compute_cost function).
    One value is to be saved after each 100 training iteration.
    """

    # initialize parameters
    parameters = initialize_parameters(layer_dims=layer_dims)

    # initialize cost list
    cost_list = []

    for i in range(num_iterations):
        # iterate over l layers to get the final last last_activation and the cache
        last_activation, caches = l_model_forward(x_data=x_train, parameters=parameters)

        cost = cross_entropy_loss(last_activation=last_activation, y_train=y_train)

        # iterate over L-layers backward to get gradients
        gradients = l_model_backward(last_activation=last_activation, y_train=y_train, caches=caches)

        # update parameters
        parameters = update_parameters(parameters, gradients, learning_rate)

        print(i)
        if i % 10 == 0 and i != 0:
            cost_list.append(cost)
            print(f'The cost after {i} iterations is: {cost}')

    return parameters


def predict(x_test: np.ndarray, y_test: np.ndarray, parameters: dict) -> str:
    """
    Description: The function receives an input data and the true labels and calculates the accuracy of
    the trained neural network on the data.

    Input:
    X – the input data, a numpy array of shape (height*width, number_of_examples)
    Y – the “real” labels of the data, a vector of shape (num_of_classes, number of
    examples)
    Parameters – a python dictionary containing the DNN architecture’s parameters

    Output:
    accuracy – the accuracy measure of the neural net on the provided data (i.e. the
    percentage of the samples for which the correct label receives the highest confidence
    score). Use the somax function to normalize the output values.
    """

    predictions, caches = l_model_forward(x_data=x_test, parameters=parameters)

    accuracy = np.mean(predictions == y_test, axis=0) / len(y_test)

    return f"The accuracy rate is: {accuracy:.2f}%."

