from forward import l_model_forward, cost_forward
from keras.src.utils import to_categorical
from backward import l_model_backward
import numpy as np


class NeuralNetwork:
    def __init__(self, x_input: np.ndarray, y_label: np.ndarray, layer_dims: np.ndarray,
                 learning_rate: float, epoch: int, batch_size: int):

        self.x_input = x_input
        self.y_label = y_label
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.parameters = {}

    def divide_to_folds(self, n_folds=5):
        """
        Description: divides input training data and labels into randomized fold.
        In addition, prepares the data by transposing and vectorised.
        """
        size = len(self.x_input)
        train_size = size * (n_folds - 1) // n_folds

        x_train = self._prepare_x_inputs(x_input=self.x_input)
        y_train = self._prepare_y_labels(y_label=self.y_label)

        ordered_pairs = list(zip(x_train.T, y_train.T))
        np.random.shuffle(ordered_pairs)

        x_fold, y_fold = zip(*ordered_pairs)

        x_train_fold = np.array(x_fold)[0:train_size].T
        y_train_fold = np.array(y_fold)[0:train_size].T
        x_test_fold = np.array(x_fold)[train_size:size].T
        y_test_fold = np.array(y_fold)[train_size:size].T

        return x_train_fold, y_train_fold, x_test_fold, y_test_fold

    def evaluate(self, n_folds=5):

        print("training...")

        accuracies = []

        for fold in range(n_folds):
            x_train_fold, y_train_fold, x_test_fold, y_test_fold = self.divide_to_folds()

            self.train(x_train=x_train_fold, y_train=y_train_fold)

            accuracy = self.predict(x_input=x_test_fold, y_label=y_test_fold)

            accuracies.append(accuracy)

        print(f'Average success rate is: {round(sum(accuracies)/len(accuracies) * 100, 4)}%')

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Description: Implements L-layer neural network. All layers but the last should have the ReLU
        activation function, and the final layer will apply the softmax activation function.
        The size of the output layer should be equal to the number of labels in the data.

        Input:
        x_input – the input data, a numpy array of shape (height*width , number_of_examples).
        y_train – the “real” labels of the data, a vector of shape (num_of_classes, number of examples).
        Layer_dims – a list containing the dimensions of each layer, including the input.
        batch_size – the number of examples in a single training batch.
        learning_rate - by how amount to change the weight matrices.
        epoch - the learning cap or limit.

        Output:
        parameters – the parameters learnt by the system during the training.
        costs – the values of the cost function (calculated by the compute_cost function).
        One value is to be saved after each 100 training iteration.
        """

        self.initialize_parameters()

        cost_list = []

        # divides to batches
        x_train_batch, y_train_batch, num_batches = \
            self._create_batches(x_data=x_train, y_data=y_train, batch_size=self.batch_size)

        batch_index = 0

        for i in range(self.epoch):

            # return to first batch when finished
            if batch_index == num_batches:
                batch_index = 0

            last_activation, caches = l_model_forward(x_input=x_train_batch[batch_index], parameters=self.parameters)

            cost = cost_forward(last_activation=last_activation, y_train=y_train_batch[batch_index])

            gradients = l_model_backward(last_activation=last_activation, y_train=y_train_batch[batch_index],
                                         caches=caches)

            self.update_parameters(gradients=gradients)

            batch_index += 1

            cost_list.append(cost)

        # plt.plot(cost_list)
        # plt.show()

    def initialize_parameters(self):
        """
        Description: initialize weight matrices with random values and bias vectors with zeros.
        layer_dims: an array of the dimensions of each layer in the network.

        output: parameters - a dictionary containing the initialized w and b parameters.
        """

        for layer in range(1, len(self.layer_dims)):
            self.parameters[f'W{layer}'] = np.random.randn(self.layer_dims[layer], self.layer_dims[layer - 1]) * 0.1
            self.parameters[f'b{layer}'] = np.zeros((self.layer_dims[layer], 1))

    def update_parameters(self, gradients: dict):
        """
        Description: Updates parameters using gradient descent

        Input:
        parameters – a python dictionary containing the DNN architecture’s parameters
        gradients – a python dictionary containing the gradients (generated by L_model_backward)
        learning_rate – the learning rate used to update the parameters (the “alpha”)

        Output:
        parameters – the updated values of the parameters object provided as input
        """

        length = len(self.parameters) // 2

        for layer in range(1, length + 1):
            self.parameters[f'W{layer}'] = self.parameters[f'W{layer}'] - self.learning_rate * gradients[f'dW{layer}']
            self.parameters[f'b{layer}'] = self.parameters[f'b{layer}'] - self.learning_rate * gradients[f'db{layer}']

    def predict(self, x_input: np.ndarray, y_label: np.ndarray, test=False) -> float:
        """

        """
        if test:
            x_input = self._prepare_x_inputs(x_input=x_input)
            y_label = self._prepare_y_labels(y_label=y_label)

        probs, _ = l_model_forward(x_input=x_input, parameters=self.parameters)
        predictions = np.argmax(probs, axis=0)
        predictions = self._prepare_y_labels(y_label=predictions)
        accuracy = (predictions == y_label).sum() / (len(y_label[1]) * 10)
        return accuracy

    @staticmethod
    def _prepare_x_inputs(x_input: np.ndarray) -> np.ndarray:
        """
        Description: prepare inputs by vectorization, transposing and normalizing input data.
        return: input in shape [num of features in [0,1], num of samples]
        """
        x_input = x_input.reshape((x_input.shape[0], 28 * 28))
        x_input = x_input.T
        x_input = x_input / 255

        return x_input

    @staticmethod
    def _prepare_y_labels(y_label: np.ndarray) -> np.ndarray:
        """
        Description: one-hot encoding
        return: output labels in [ground truth, num of samples]
        """
        y_label = to_categorical(y_label, num_classes=10)
        y_label = y_label.T

        return y_label

    @staticmethod
    def _create_batches(x_data: np.ndarray, y_data: np.ndarray, batch_size: int) -> tuple[list, list, int]:
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
