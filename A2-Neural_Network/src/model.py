from forward import initialize_parameters, l_model_forward, cost_forward
from backward import l_model_backward, update_parameters
from activation_functions import softmax
import numpy as np
import matplotlib.pyplot as plt

def l_layer_model(x_train: np.ndarray, y_train: np.ndarray, layer_dims: np.ndarray, learning_rate: float,
                  num_iterations: int, batch_size: int) -> dict:
    """
    Description: Implements L-layer neural network. All layers but the last should have the ReLU
    activation function, and the final layer will apply the sigmoid activation function.
    The size of the output layer should be equal to the number of labels in the data.
    Batch size is selected such that it enables the code to run well.

    Input:
    x_input – the input data, a numpy array of shape (height*width , number_of_examples).
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

    x_train_batch, y_train_batch, num_batches  = create_batches(x_data=x_train, y_data=y_train, batch_size=batch_size)
    batch_index = 0

    for i in range(num_iterations):

        if batch_index == num_batches:
            batch_index = 0

        # iterate over l layers to get the final last last_activation and the cache
        last_activation, caches = l_model_forward(x_input=x_train_batch[batch_index], parameters=parameters)

        cost = cost_forward(last_activation=last_activation, y_train=y_train_batch[batch_index])

        # iterate over L-layers backward to get gradients
        gradients = l_model_backward(last_activation=last_activation, y_train=y_train_batch[batch_index], caches=caches)

        # update parameters
        parameters = update_parameters(parameters, gradients, learning_rate)

        batch_index += 1

        cost_list.append(cost)
        if i % 10 == 0 and i != 0:
            print(f'The cost after {i} iterations is: {cost}')

    plt.plot(cost_list)
    plt.show()

    return parameters


def create_batches(x_data: np.ndarray, y_data: np.ndarray, batch_size: int) -> tuple[list, list, int]:
    num_samples = x_data.shape[1]
    num_batches = num_samples // batch_size
    batches_x = []
    batches_y = []

    # Create full-sized batches
    for i in range(num_batches):
        batch_x = x_data[:, i * batch_size: (i + 1) * batch_size]
        batch_y = y_data[:, i * batch_size: (i + 1) * batch_size]
        batches_x.append(batch_x)
        batches_y.append(batch_y)

    # Create the last smaller batch, if necessary
    if num_samples % batch_size != 0:
        batch_x = x_data[:, num_batches * batch_size:]
        batch_y = y_data[:, num_batches * batch_size:]
        batches_x.append(batch_x)
        batches_y.append(batch_y)

    return batches_x, batches_y, num_batches


def predict(x_test: np.ndarray, y_test: np.ndarray, parameters: dict) -> str:
    """
    Description: The function receives an input data and the true labels and calculates the accuracy of
    the trained neural network on the data.

    Input:
    x_input – the input data, a numpy array of shape (height*width, number_of_examples)
    Y – the “real” labels of the data, a vector of shape (num_of_classes, number of
    examples)
    Parameters – a python dictionary containing the DNN architecture’s parameters

    Output:
    accuracy – the accuracy measure of the neural net on the provided data (i.e. the
    percentage of the samples for which the correct label receives the highest confidence
    score). Use the somax function to normalize the output values.
    """

    probs, caches = l_model_forward(x_input=x_test, parameters=parameters)

    probs = softmax(probs)

    predictions, _ = np.argmax(probs, axis=1)

    accuracy = np.sum(predictions == y_test, axis=0) / len(y_test)

    return f"The accuracy rate is: {accuracy}%."

