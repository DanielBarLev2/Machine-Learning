from application import l_layer_model, predict
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Description: load mnist dataset. then, vectorize it and encode y data as one-hot.

    Output:
    x_train: [num of examples, features].
    y_train: [num of examples, true label].
    x_test: [num of examples, features].
    y_test: [num of examples, true label].
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # vectorization
    x_train = x_train.reshape((x_train.shape[0], 28*28))
    x_test = x_test.reshape((x_test.shape[0], 28*28))

    # one-hot encoding
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


def normalize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Description: scale and normalise data.

    Input:
    x_train -[num of examples, features].
    x_test - [num of examples, features].

    Output:
    x_train_norm, x_test_norm - normalised to contain values between [0,1].
    """

    x_train_norm = x_train.astype('float32')
    x_test_norm = x_test.astype('float32')

    x_train_norm = x_train_norm / 255
    x_test_norm = x_test_norm / 255

    return x_train_norm, x_test_norm


def prepares_folds(x_data: np.ndarray, y_data: np.ndarray, n_folds=5) -> dict:
     """
     Description: prepares cross validation by dividing to five folds.

     Input:
     x_data - [num of examples, features].
     y_data - [num of examples, true label].

     Output:
     fold - dictionary of five folds: [x_fold_train, x_fold_test, y_fold_train, y_fold_test].
     """

     k_fold = KFold(n_folds, shuffle=True, random_state=1)
     folds = {}

     # enumerate splits
     for index, (x_train_i, x_test_i) in enumerate(k_fold.split(x_data)):

         # split the folds and transpose the input vectors
         x_fold_train, x_fold_test = x_data[x_train_i].T, x_data[x_test_i].T
         y_fold_train, y_fold_test = y_data[x_train_i].T, y_data[x_test_i].T

         folds[f'fold{index}'] = x_fold_train, x_fold_test, y_fold_train, y_fold_test

     return folds


def evaluate_model(folds: dict, num_iterations=100, learning_rate=0.009) -> dict:
    """
    Description: evaluates model using five folds and a cross validation test samples.

    Input:
    folds - [x_fold_train, x_fold_test, y_fold_train, y_fold_test].

    Output:
    parameters - dict filled with weight and biases.
    """
    parameters = {}

    for k_fold in folds:
        x_fold_train, x_fold_test, y_fold_train, y_fold_test = folds[k_fold]

        # set up dimensions
        layers_dims = np.array([x_fold_train.shape[0], 20, 7, 5, 10])

        # train the network by folds
        parameters = l_layer_model(x_train=x_fold_train, y_train=y_fold_train, layer_dims=layers_dims,
                                   learning_rate=learning_rate, num_iterations=num_iterations)

        accuracy = predict(x_test=x_fold_test, y_test=y_fold_test, parameters=parameters)
        print(f'fold {k_fold} accuracy on validation set is {accuracy} %')

    return parameters

def run():
    x_train, y_train, x_test, y_test = load_dataset()

    x_train, x_test = normalize(x_train=x_train, x_test=x_test)

    folds = prepares_folds(x_data=x_train, y_data=y_train)

    parameters = evaluate_model(folds=folds)

    accuracy = predict(x_test=x_test, y_test=y_test, parameters=parameters)

    print(f'accuracy on test set is {accuracy} %')

